# METIS_NodeND 到 Supernode Tree：当前实现、1000 DOF 上限与工程改造设计

> 日期：2026-10-11。依据当前工作树源码，包含尚未提交的修改。
> 本文是技术评估与开发设计；本次只新增文档，没有实施求解器改造。
> 本文中的 DOF 指矩阵行列索引。必须区分节点自身索引数、上层更新索引数、前沿规模和投影维数。

## 1. 直接回答

1. **`METIS_NodeND` 本身只返回排列 `perm/iperm`，不会直接返回分割树、标量消去树或超节点树。** 它在内部进行 nested dissection；若要精确保留它内部的分隔器层次，需要另一个能输出层次的接口或对其实现做额外记录，不能仅凭最终排列唯一恢复内部递归历史。
2. **当前默认 `--partition nd` 是“标量消去树经尺寸聚合、单子链收缩后得到的分割块树”。** 它具有消去树来源及祖先耦合性质，代码称聚合块为 `Sup`，但没有进行严格超节点的列模式判定。它不是标准符号 Cholesky 意义下的 exact/fundamental supernode tree，也不是 METIS 内部原样输出的 nested-dissection separator tree。
3. `--partition bisect` 是递归分割树；`--partition external` 是外部注入的块树，其语义由文件内容决定。目前 external 也受单根、禁止单子节点等校验限制，不能直接接受一般超节点森林。
4. **从 NodeND 得到标准超节点树，需要在该排列下做符号分析，得到标量 etree、填充后列计数/模式，再划分 exact supernodes、构建块父子关系。** relaxed amalgamation 是可选优化，不是必需步骤；不必先做数值分解。
5. **超节点树通常是不规则多叉树，可能有大量叶子、单子链及很不均匀的深度。** 它没有满二叉树、完全二叉树或平衡树保证。结构连通的对称图，其标准标量消去树有一个根；多个真正的根对应结构不连通的分量，不能把“很多子树”理解成“很多互不依赖的根”。
6. **自然超节点没有 `1000 DOF` 上界。** 可把最终计算块的自身索引数强制限制为 `<=1000`：保留单子链，禁止超限合并，对超大的 exact supernode 按连续消去列拆成链。拆后应称为 bounded supernode blocks 或受限消去块，不再声称每块都是最大的自然超节点。
7. **`每块 <=1000` 不等于所有计算规模都 <=1000，也不等于百万/千万 DOF 会快。** 上层更新宽度、填充量、链长、跨级耦合、局部保留谱数量和最终投影维数必须同时控制。当前稠密投影求解以及每节点至少保留一阶的默认策略，是细粒度树落地的主要额外障碍。

## 2. 当前工程实际做了什么

### 2.1 默认入口与构建流水线

源码定位采用 1-based 行号。

| 位置 | 当前行为 | 含义 |
|---|---|---|
| [`solver.hpp:23`](../src/amls/solver.hpp#L23) | `partitionMethod = "nd"` | 默认运行 ND 路线 |
| [`metis_partitioner.cpp:254`](../src/partition/metis_partitioner.cpp#L254) | 一次 `METIS_NodeND` | 得到排序，不是树 |
| [`metis_partitioner.cpp:259`](../src/partition/metis_partitioner.cpp#L259) | Liu 风格标量消去树算法 | 由重排后的图建立 `parent` |
| [`metis_partitioner.cpp:318`](../src/partition/metis_partitioner.cpp#L318) | 后序累积 `bag`，达到 `leafSize` 后切出块 | 按尺寸聚合 etree 连通部分，没有匹配填充列模式 |
| [`metis_partitioner.cpp:378`](../src/partition/metis_partitioner.cpp#L378) | 多个 etree 根的顶层索引并入单一根 | 不保留原始森林结构；根规模可能增大 |
| [`metis_partitioner.cpp:393`](../src/partition/metis_partitioner.cpp#L393) | 反复合并所有单子节点 | 没有尺寸上限，深链可能收缩成大块 |
| [`metis_partitioner.cpp:409`](../src/partition/metis_partitioner.cpp#L409) | 输出 `SeparatorTree` | 无子块为 Leaf，其余为 Separator/Root |
| [`metis_partitioner.cpp:429`](../src/partition/metis_partitioner.cpp#L429) | 后序拼接每块原始索引，重新生成 `res.perm` | 最终工程排列不保证等于 METIS 原始 `mperm` |

这条路线可以概括为：

```mermaid
flowchart LR
    A[对称结构图] --> B[METIS_NodeND 排列]
    B --> C[标量消去树]
    C --> D[按 leafSize 累袋聚合]
    D --> E[多个根合并]
    E --> F[全部单子链收缩]
    F --> G[SeparatorTree 和最终排列]
```

`Sup` 的命名不构成严格超节点证明。当前没有 `ColCount`、填充列集合相等判定或符号 supernodal pattern 后端。

### 2.2 为什么 leafSize 不是上限

切块发生在所有孩子的剩余 `bag` 汇入当前节点以后，条件是 `bag.size() >= leafSize`。因此它是**切出触发阈值**，不是 `bag.size() <= leafSize` 的保证。

随后还有两次可能扩大的操作：多个分量顶块合并；所有单子链向上合并。后者没有任何 `maxNodeSize` 检查。即使把 `leafSize` 调成 3、100 或 500，也没有所有分割块都小于 1000 的保证。

一个极端例子是路径图：标量 etree 可以是长链。当前禁止单子节点的策略可能把这类链全部合成一个叶块。说明“图规模大但局部很稀疏”不代表当前构树会产生小块。

历史对比文档记录过 `cant_20` 的 `leafSize=500` 树：40 块、最大块 1458、叶均 640，见 [`AMS对比PKU_CCM.md`](abaqus_ams/AMS对比PKU_CCM.md)。这是旧运行记录，不是本次重跑；它与当前实现没有硬上限这一事实一致。

### 2.3 树接口已经支持什么

`children` 是向量，下游很多操作已遍历所有孩子，因此**不需要把整个调度器从二叉重写为多叉**。主要障碍是现有契约：

- [`separator_tree.cpp:81`](../src/amls/separator_tree.cpp#L81)：恰好一个根。
- [`separator_tree.cpp:92`](../src/amls/separator_tree.cpp#L92)：非 Leaf 节点至少两个孩子，拒绝真正的单子链。
- `postOrder`/`descendants` 采用递归遍历：小块拆分造成深链后，存在调用栈深度问题。
- 下游依赖后序节点编号、每块重排索引连续升序，以及非零耦合块的双方在祖先关系上可比。
- [`original_blocks.cpp:139`](../src/amls/original_blocks.cpp#L139) 附近拒绝不可比节点之间的 K/M 耦合。这是应继续保留的数学不变量。

## 3. 四种“树/块”不能混为一谈

| 对象 | 一个节点代表什么 | 从排序得到它还需要什么 | 一般形状 |
|---|---|---|---|
| Nested-dissection separator tree | 分隔器集合或停止分割的叶集合 | 保留分割过程；排序本身不唯一确定该历史 | 常有二分来源，但不保证满二叉 |
| Scalar elimination tree | 单个消去列/索引 | 重排结构的 etree 符号算法 | 多叉、单子链、可能森林 |
| Exact supernode tree | 若干连续消去列，填充后共享下方行模式 | etree + filled column counts/pattern | 标量树的合法收缩，通常不规则 |
| Bounded AMLS block tree | 适合局部计算、尺寸受限的索引块 | exact supernode 的拆分/受限聚合及契约验证 | 可以比自然超节点更深，也可以保留一子节点 |

Exact supernode 的定义依赖**填充后的消去结构**，不是原矩阵中相邻列看起来相似，更不是相邻行非零数恰好差 1。

工程实现还常区别 fundamental supernodes 与 relaxed supernodes。CHOLMOD 的 fundamental 构造会在分支处额外设边界；relaxed amalgamation 可以把模式不完全相同的块合并并引入显式零。本文建议首版使用 exact 分组并在标量树分支处保守设边界，关闭 relaxed 合并，以便验证与解释。

## 4. 从 METIS_NodeND 到受限 Supernode Tree

### 4.1 构造矩阵对的结构图

建议使用

\[
G = \operatorname{pattern}(K) \cup \operatorname{pattern}(M)
\]

的无向对称结构，去掉自环、重复边和明确的显式零；以集合并集表达，避免数值相加导致抵消。显式零处理必须与数值块抽取保持一致：统一清除 K/M 中的显式零，或让 `OriginalBlocks` 按相同规则跳过零值。当前抽取会对存储条目执行祖先可比检查，仅从结构图忽略零边却保留数值条目，可能错误拒绝合法输入。

现工程下游要求 K、M 块耦合都只出现在祖先可比节点之间。仅对 `pattern(K)` 建树，若 M 另有结构边，可能破坏这一要求。对并集图做符号分析得到的是保守消去结构：它可能比 K 单独的符号填充更多，得到的是**并集模式对应的超节点**，不能宣传成“K 单独的最大 exact supernodes”。当 M 为单位阵或纯对角阵时，二者结构一致。

这里仅分析模式，不给 `pattern(K) ∪ pattern(M)` 的 0/1 矩阵做数值 Cholesky，也不把它当成原特征值问题的数值算子。

### 4.2 排列方向与重排

采用 METIS 和工程已有约定：

```text
perm[new] = old
iperm[old] = new
Aperm(i,j) = A(perm[i], perm[j])
x_original[perm[i]] = x_permuted[i]
```

依据 vendored [`ometis.c:37`](../third_party/metis/libmetis/ometis.c#L37)。两边都要重排：`Kp = P K Pᵀ`、`Mp = P M Pᵀ`。

**符号分析不要求先物理复制数值 K/M。** 可以将原始模式及 `perm` 交给符号后端，或先构造重排后的 pattern CSC。只在得到最终采用的排列后，对数值矩阵重排一次。否则在千万级会出现不必要的矩阵副本峰值。

如果符号后端额外做 etree postorder，应使用并验证后端最终返回的排列。若 `q[new2] = old_position`，组合为 `permFinal[new2] = permND[q[new2]]`。绝不能分析一种顺序，却把另一种顺序的列范围/父节点交给数值阶段。

### 4.3 符号分析需要输出的元数据

最少输出：

```text
permFinal, ipermFinal
parentColumn[j]        // 标量 etree，-1 表示根
columnCount[j]         // L(:,j) 含对角的结构非零数
scalarPostorder
componentRoots
```

对 exact supernode 的定义，若 `C_j` 是 filled L 第 j 列的行集合，则连续列 j 和 j+1 的基本相容条件是：

\[
\operatorname{parent}(j)=j+1,\qquad
C_j\setminus\{j\}=C_{j+1}.
\]

在**确实由同一结构、同一排列得到的 Cholesky etree 与列计数**前提下，可用

\[
\operatorname{parent}(j)=j+1,\qquad
\operatorname{columnCount}(j)=\operatorname{columnCount}(j+1)+1
\]

判定模式相容；父列包含关系与相等计数使集合相等。单独检查列计数下降 1 不够。

首版再加 `scalarChildCount[j+1] <= 1` 的保守限制，形成不跨标量树分支的 fundamental 风格分组。这个限制影响块粒度，不应与前述模式相容的数学必要性混淆。

### 4.4 推荐符号后端：CHOLMOD，只分析、不数值分解

本机已确认存在：

```text
D:/msys2/mingw64/include/suitesparse/cholmod.h
D:/msys2/mingw64/lib/libcholmod.a
D:/msys2/mingw64/lib/libcholmod.dll.a
```

当前 [`CMakeLists.txt:37`](../CMakeLists.txt#L37) 只有注释预留，没有链接 CHOLMOD。本机文件存在不等于依赖已经完整配置或链接验证通过。

建议新增 `ISymbolicAnalyzer` 和 `CholmodSymbolicAnalyzer`，使用 long/64-bit API。本机头文件提供 `cholmod_l_analyze_ordering`，其输入为 A、指定的 GIVEN 排列，显式输出 `Parent/Post/ColCount`，适合首版只提取符号元数据。必须封装调用与内存管理，并核对固定 SuiteSparse 版本的函数契约。

实现约束：

- A 是合法压缩 CSC 模式矩阵；按 CHOLMOD 的对称单三角规则设置 `stype`，不能误设 `stype=0`，否则会分析 `A Aᵀ` 路线。
- 使用 `CHOLMOD_GIVEN`，不让默认策略重新选择 AMD/METIS。已经调用一次 NodeND 后，不应因后端默认配置隐式再排序。
- 明确是否采用 `Post` 重排；首版可固定 NodeND 顺序，将 `Post` 用于遍历/辅助分析，不另改列序。
- 64-bit API、`idx_t`、工程 `int64_t`、`size_t` 的长度转换必须检查。
- symbolic simplicial factor 不能假定有 `L->p/i` 的完整 filled pattern；该形式主要提供 `Perm/ColCount`。
- 如要 front/update rows 的精确行集合，另做受控的符号 reach、supernodal symbolic extraction 或块级耦合模式分析；不要靠列计数猜这些行号。
- CHOLMOD relaxed 参数 `nrelax/zrelax` 不是 1000 的硬上限：无新增零的合并以及低零比例条件可以不受该尺寸限制。硬上限必须在本工程建块阶段再次落实。

备用方向是自己实现符号列计数与 reach，再与 CHOLMOD 交叉验证。现有 Liu etree 可复用作为参考，但仅有 etree 不能证明列模式相同。首版不推荐同时自研完整符号分解与改造 AMLS 全链路。

### 4.5 exact 分组、硬拆分及块树构建

首版步骤：

1. 按上述相容条件扫描连续消去列，得到 exact/fundamental 风格超节点区间。
2. 关闭 relaxed 合并。后续只有在合并后索引数 `<=maxBlockDofs` 且满足合法连通收缩条件时，才允许受限合并。
3. 若自然块宽 `w>1000`，按连续消去列切成宽 `<=1000` 的区间；块内原消去顺序保持不变。
4. 建立 `blockOfColumn[j]`；对每个最终块，由标量父边确定外部块父关系，并验证外部父关系唯一。对拆分出的 exact 单链块，低编号块指向下一个高编号块，最后一个块指向原超节点父块。
5. 原子树按标量父边重新挂接到包含目标列的拆分块，不应一律挂到拆后最顶块；所有外部耦合必须重新映射和校验。
6. 保留单子节点；输出根列表及迭代后序，后序重编号后仍满足祖先 id 大于后代 id。
7. 每个计算块的索引在最终排列中为连续升序范围；保持 DOF 全覆盖、无重复。
8. 最终断言 `1 <= n_i <= 1000`。该断言应放在所有合并、根处理和排列变换之后。

简化伪代码：

```text
graph = symmetricPatternUnion(K, M)
permND = METIS_NodeND(graph)
symbolic = analyzeGivenOrdering(graph, permND)
exactRanges = findExactRanges(symbolic.parent, symbolic.columnCount,
                              preserveBranchBoundaries = true)
boundedRanges = splitContiguous(exactRanges, maxBlockDofs = 1000)
blockForest = contractScalarTree(symbolic.parent, boundedRanges)
validateCoverageAndAncestorEdges(blockForest, graph)
finalizePostorderAndContinuousRanges(blockForest, permND)
assert max(blockForest.ownedDofs) <= 1000
```

这不是将一个任意分隔器数组每 1000 个元素随便切开。正确拆分依据连续消去列和合法父边，拆后块间非零耦合必须在祖先链上。

### 4.6 拆分例子：大团不消失，只变成多块链

若排序后末端有 5000 列的稠密团，自然 exact supernode 可以宽 5000。拆为五块：

```text
B0: 列 [   0,1000)  → B1
B1: 列 [1000,2000)  → B2
B2: 列 [2000,3000)  → B3
B3: 列 [3000,4000)  → B4
B4: 列 [4000,5000)  → 原父块或根
```

B0 自身只有 1000 DOF，但需要更新后面的 4000 DOF。这里可能存在 B0 到 B2/B3/B4 的跨级耦合，并非只有父子块耦合。

在不做谱截断时，这种拆分对应相同列序的分块精确消去，可保持代数等价。**AMLS 会在每块进行局部谱截断，因此把一块变成五块会改变近似子空间与误差；不能据此宣称截断后的结果完全不变。**

## 5. 节点 DOF、小块与大规模性能

### 5.1 必须同时记录四种尺寸

设节点 s 的自身消去列集合为 `I_s`，更新到后续列的行集合为 `U_s`：

\[
n_s=|I_s|,\quad u_s=|U_s|,\quad f_s=n_s+u_s,\quad k_s=\text{保留局部谱数},\quad p=\sum_s k_s.
\]

| 指标 | 1000 上限能否保证 | 对工程的意义 |
|---|---|---|
| `n_s`：节点自身 DOF | 可以，靠最终拆分与禁止超限合并 | 限制局部对角块、局部稠密谱问题规模 |
| `u_s`：更新行/祖先耦合列总宽度 | 不能 | Ψ、Mtemp、Schur 更新仍可能大 |
| `f_s`：前沿 DOF | 不能 | 若走 frontal 存储，局部工作区仍可能大 |
| `p`：全局投影维数 | 不能 | 决定最终稠密/稀疏特征求解的成本 |

双精度 `1000×1000` 矩阵约 8 MB，局部 K/M、因子、向量和工作区的总内存会更多；并发节点数也影响峰值。即使自身索引数受限，也应给每个并行任务估算工作区并限制在途内存。

### 5.2 百万/千万规模没有自然小节点保证

对典型规则模型，在固定每个空间点未知数数目的前提下：

- 2D nested dissection 的顶层 separator 规模通常为 `O(N^(1/2))`。
- 3D 通常为 `O(N^(2/3))`；`N=10^6` 与 `10^7` 的数量级分别约 `10^4` 与 `4.6×10^4`，常数和图结构会改变实际数值。

这是典型 separator 渐近量级，不是每个 supernode 的预测值，也不是所有稀疏图的定理。但它说明：大规模三维问题的高层稠密耦合可能非常宽，不能以“小叶块”推出“小高层成本”。

稀疏 Cholesky 的自然块会因排序、结构等价列、根部团和 relaxed 合并而变大。硬拆分可以控制 `n_s`，不会消除整体填充，也不会自动降低总算术复杂度。拆得过细还会增加链长、同步和容器开销，减少 BLAS3 效率。

推荐第一轮比较 `maxBlockDofs=256/512/1000`，报告实际规模、时间、内存与谱误差，避免预设越细一定越快。1000 是可配置的工程边界，不是普适性能最优值。

### 5.3 树里有很多子树，但并行度不均匀

树的每个非叶节点下面都有子树。大图经良好排序通常有很多可以并行的分支，但靠近根仍可能出现长单子链或共享祖先，从而形成关键路径。

必须分别记录叶数、每层节点数、最大扇入、最长父链、单子节点比例、加权关键路径与节点内核时间。树更深不必然有更均匀的并行度，超节点更小也不保证根部依赖消失。

## 6. 下游工程必须修改或审计的地方

### 6.1 构树与接口

| 文件/模块 | 当前约束或行为 | 建议开发内容 |
|---|---|---|
| `src/partition/partitioner.hpp` | 返回 `SeparatorTree + perm` | 添加 tree kind、符号统计与构建策略；现有 nd/bisect 行为保留用于比较 |
| `src/partition/metis_partitioner.cpp` | NodeND、etree、尺寸聚合混在一个函数 | 分离 ordering、symbolic analysis、exact grouping、bounded blocking；新增 `supernode` 路线 |
| 新增 `src/partition/symbolic_analyzer.*` | 尚无 | 统一 `Parent/Post/ColCount` 元数据，封装 CHOLMOD long API |
| 新增 `src/partition/supernode_builder.*` | 尚无 | exact 判定、受限拆分、块父映射、规模断言、树统计 |
| `src/amls/separator_tree.hpp/.cpp` | 单根、非叶至少两子、递归遍历 | 用语义明确的 BlockEliminationTree/Forest 或 tree-kind 校验；支持单子节点，迭代遍历，缓存 depth/postorder/tin/tout |
| `src/partition/external_tree.*` | 单根块树格式；`perm:` 模式另生成 ND 树 | 扩展版本化导入；支持真实链/森林；给定外部排列必须按此排列重新符号分析与构树 |
| `src/amls/solver.hpp/.cpp` | 默认 nd，根/索引/后序规范化 | 接入新策略，保留明确的最终排列，增加资源预算与前置拒绝 |
| `src/apps/amls_solve.cpp`, `amls_verify.cpp` | 分割选项与参数校验 | 接入新参数、tree-only 诊断及比较选项 |
| `CMakeLists.txt` | CHOLMOD 未链接 | 可选后端开关、固定版本/依赖发现、64-bit API 链接及 CI/本机构建 |

建议新增的配置名仅是设计，当前命令行不存在：

```text
--partition supernode
--symbolic-backend cholmod
--max-block-dofs 1000
--supernode-amalgamation exact
--tree-only
--tree-stats <json-file>
--projected-solver auto|dense|sparse
--memory-budget-mb <budget>
--local-min-modes <count>
```

`leafSize` 继续服务旧分割策略；新策略用 `maxBlockDofs` 表达明确上限，不复用一个参数表达相反语义。

特别需要修正 external 的纯排列入口：[`external_tree.cpp:91`](../src/partition/external_tree.cpp#L91) 附近先用内部 ND 生成树，再只覆盖 `res.perm`。这不是“由外部排列生成消去树”。任意排列会与块索引连续性及符号元数据不一致；正确做法是 `analyzeGivenOrdering(A, externalPerm)`，再按该结果建块树。

### 6.2 森林：不能靠拼成大数值根绕过问题

推荐目标结构显式保存 `roots[]`，所有树遍历、祖先查询和后序编号覆盖各个分量；普通根仍是非空计算节点，不增设空 DOF 的数值根。

当前根相关查询与 [`temp_block_manager.cpp`](../src/amls/temp_block_manager.cpp) 的最终释放判定需要改为全森林完成条件。调度队列应能接收各分量的叶节点。

首版若范围受限，可以仅支持结构连通输入，发现多根就明确拒绝并报告；完整版本再支持森林。不能无条件把多个根的 DOF 合起来，因为合并后的数值根可能再次超过 1000。空 DOF 虚根只能作调度元节点，不能送进现有 `K.rows()>0` 的局部谱求解路径。

### 6.3 消去、跨级耦合与临时存储

| 模块 | 需要保持的约束 | 改造/审计重点 |
|---|---|---|
| `OriginalBlocks` | 每个 DOF 只属于一块；K/M 非零边双方祖先可比 | 对新树做全边校验，支持连续范围索引映射，拒绝符号/数值排列不一致 |
| `GaussEliminationModule` | 所有祖先方向耦合与 Schur 更新正确 | 保留多叉循环，验证单子链；不能只更新父块；取消对“全部祖先都有块”的过度预建 |
| `TempBlockManager` | μ/Mtemp 的发布、投影、传播、释放严格依赖消费者完成 | 验证深链、多根及零谱中间节点；避免按所有后代×祖先路径批量复制键 |
| `NodeTaskExecutor` | 非叶必须等待所有孩子相关任务；每任务只执行一次 | 单子节点的就绪/倒计数应正常；加入按内存预算限制在途任务的机制 |
| `EigenPairSaver` | 谱数/特征向量形状一致，局部 M-正交 | 小块/零谱的形状、偏移和投影必须端到端验证 |
| `SubspaceMassSaver` | 合法跨级块方向与投影对称性 | 将全祖先对预建改为符号活跃块集；新增稀疏/矩阵操作形式 |
| `solver` 回代 | 原索引恢复、全祖先 Ψ 折叠 | 用已缓存 depth，支持单子链/森林；输出按列批次恢复，减少峰值 |

目前 [`subspace_mass_saver.hpp:38`](../src/amls/subspace_mass_saver.hpp#L38) 为每个节点的所有祖先对预建 key。深链下总量为 `Σ depth(i)`，最坏可二次增长；消去阶段还可能枚举祖先对，单节点成本达 `O(depth(i)^2)` 的元数据/循环量级，另有后代传播开销。必须依照保守活跃耦合集减少不必要的键，而非仅把树节点改小。

**质量变换的活跃集不能直接等于 CHOLMOD 的 L 非零模式。** 必须从原始 K/M 块和 Ψ 模式出发，对 `Mtemp`、质量更新及 `μ(d,a) += μ(d,x) Ψ(x,a)` 的乘加逐步构造符号闭包，保留所有可能生成非零块的依赖。例如即使 K 为三对角、M 为单位阵，完整消去同余变换后的 M 也可能出现远祖先耦合，链上甚至稠密。零谱投影消掉的最终块与仍需传播的中间块应分别处理。首版继续使用完整祖先对作为正确性基线，再以小矩阵显式同余结果验证活跃集裁剪，不能按 L 模式直接丢弃质量块。

祖先查询建议用预先算好的 `tin/tout` 做 O(1) 可比性判定。不能为千万个标量节点预存每条祖先链；对最终计算块也应尽量采用按需迭代与经过闭包验证的耦合表。

### 6.4 小块谱截断：必须先解决默认至少一阶与空谱回退

当前 [`node_task_executor.hpp:206`](../src/amls/node_task_executor.hpp#L206) 的 `minExtractOrder=1` 被 solver 默认使用。因此每个非空计算块至少保留一阶，通常有

\[
p \ge N_{\text{blocks}}.
\]

如果 DOF 总数 N 被分成自身宽度 `<=1000` 的块，则块数至少为 `ceil(N/1000)`。千万 DOF 即使采用尽可能大的 1000 块，也至少有一万块；自然小块可能使块数远大于此。

现有 saver 初始化允许 0 谱，部分临时块路径也支持零尺寸矩阵，且已有零谱叶节点测试。但这不等于零谱中间节点已得到数学与端到端验证。不能直接把 `minExtractOrder` 改成 0 当成性能补丁。

本次代码检查还发现一个明确的待修复风险：[`node_task_executor.cpp:157`](../src/amls/node_task_executor.cpp#L157) 的区间空回退，直接对区间求解后同一 K/M 调用 `solveIndex`；而 [`dense_eigensolver.cpp:78`](../src/linalg/dense_eigensolver.cpp#L78) 的 LAPACK 调用会覆盖输入。回退应重新加载原始/更新后的局部 K/M 或保留未被覆盖的副本。小块较多时，此分支更值得专门验证。

开发顺序：先修复回退输入问题，保留至少一阶策略建立数值基线；再独立验证零谱叶/中间/根节点、全祖先传播、释放及回代，依据误差与残差决定默认策略。已有注释关于“零谱使质量耦合链断裂”不能当成数学证明，需要测试定位具体行为。

### 6.5 最终投影必须有真实的稀疏求解路线

当前 [`solver.cpp:249`](../src/amls/solver.cpp#L249) 组装 dense reduced K/M，`denseProjectedMax=5000` 仅告警，随后仍调用稠密 `solveInterval`。该内核还分配 `p×p` 的 Z。仅前三张双精度方阵的大致内存为：

\[
3\times8p^2\ \text{bytes}.
\]

| 投影维数 p | 单张 dense 方阵 | K、M、Z 三张下界 |
|---:|---:|---:|
| 5,000 | 0.20 GB | 0.60 GB |
| 10,000 | 0.80 GB | 2.40 GB |
| 29,488 | 6.96 GB | 20.87 GB |
| 100,000 | 80 GB | 240 GB |

这里用十进制 GB，只计算这三张矩阵，不含副本、局部块、LAPACK 工作区、回代向量或线程并发峰值。p=29,488 是说明细粒度块数量的算例，不表示已经得到一棵合法新超节点树。

必须开发：

- reduced K 保持对角向量，避免稠密化；reduced M 依据真实块占用率选择 SparseMatrix 或 block/matrix-free matvec。**Mred 不保证稀疏**：其活跃集须包含质量变换的符号闭包；若近于稠密，改用 SparseMatrix 不能消除平方存储，应评估隐式算子、谱策略或重新聚合。
- 接入真正的广义稀疏低端谱/区间求解器。已有 Spectra 参考工具可供复用算子与经验，但不能把参考工具直接当作已接入的 projected solver。
- 移位反演须使用 `Kred - σ Mred` 及合适分解/预条件；σ=0 且 Kred 有零谱时不能假定可逆。
- 说明“给定频段全谱”与“前 nev 阶”的差别：迭代法只得到指定阶数，若要保证频段完整，还需计数/自适应扩阶与收敛验证。
- dense/sparse 切换由估算内存、p、谱目标与稀疏度共同决定；未实现的路线应在分配前明确拒绝，不再仅打印警告。
- 验证 Mred 对称正定、投影 Ritz 残差、原问题回代残差及 M-正交性。不能将已有 B 不正定失败未经定位就归咎于“对角填 1”；局部 M-正交成立时单位对角本身是合理约定。

### 6.6 千万规模的索引与输入/输出预算

工程 `DefaultTraits` 和本机 METIS 已为 64-bit。仍需审计所有非零偏移、Eigen/METIS/CHOLMOD 长度转换和二进制缓存格式；现 benchmark 旧缓存示例有 i32 列号，列号与 nnz 偏移的容量要分别论证。

`readMatrixMarket` 当前采用 triplet 展开与 `setFromTriplets`，大规模时会同时持有 triplets 与压缩矩阵。应提供二进制 CSC/CSR 直接读入和受控峰值加载，避免给符号步骤再加多个数值副本。

仅恢复 N×200 个双精度特征向量：百万 DOF 约 1.6 GB，千万 DOF 约 16 GB；当前恢复阶段还会持有节点行块再拼全局矩阵。必须支持不恢复、按特征向量批次恢复及流式保存。1000 DOF 的局部块上限不会降低这个输出下界。

## 7. 建议的数据模型

首版复用现有 tree API 但增加 tree kind 与校验策略，避免对全部模块立刻机械改名；后续可把通用数值结构正式命名为 `BlockEliminationForest`，分割树作为一种构建策略。

```cpp
struct SymbolicAnalysis {
    std::vector<int64_t> perm;
    std::vector<int64_t> inversePerm;
    std::vector<int64_t> parentColumn;
    std::vector<int64_t> columnCount;
    std::vector<int64_t> scalarPostorder;
};

struct EliminationBlock {
    int64_t begin;             // 最终重排列区间 [begin, end)
    int64_t end;
    int64_t parent;            // -1 表示真实根
    std::vector<int64_t> children;
};

struct BlockTreeMetadata {
    std::vector<int64_t> roots;
    std::vector<int64_t> postorder;
    std::vector<int64_t> depth;
    std::vector<int64_t> tin, tout;
    int64_t maxOwnedDofs;
};
```

精确更新行集、活跃 block-pair 集及符号填充预算另存为紧凑 CSR 风格结构。不要在节点上默认塞满 `ancestors/descendants` 展开列表。示例为接口草案，尚未编译实现。

## 8. 分阶段开发工作与验收

下表为单人熟悉本工程、已有正常构建环境情况下的粗估，包含实现与有针对性的测试，不是交付承诺。精度问题和稀疏谱收敛可能延长工期。

| 阶段 | 工作内容 | 可审查交付与验收 | 粗估 |
|---|---|---|---:|
| A：符号与树诊断 | CHOLMOD 64-bit 后端、exact/fundamental 分组、连续拆分、tree-only 统计；不跑 AMLS | 小矩阵与独立 filled-pattern oracle 一致；所有块 <=1000；覆盖、父边、祖先可比、排列正确 | 3–5 人日 |
| B：新树接入 | 单子链、迭代遍历、后序/范围接口；external 给定排列重建；连通图 AMLS；修复空谱回退输入 | 单节点、真一子链、三叉树、超限根拆分，全谱保持时与直接法一致；旧 nd 路线回归 | 4–7 人日 |
| C：森林与零谱策略 | roots[] 生命周期；零谱中间/根节点全链路；截断误差验证 | 多分量精确谱；0 谱不破坏投影/释放；串并行一致、残差与 M-正交合格 | 3–6 人日 |
| D：细树可扩展存储与投影 | 活跃耦合键、稀疏 Mred、对角 Kred、projected 迭代求解及内存前置检查 | 小规模 dense/sparse 一致；10万数据稳定完成；记录求解目标完整性和收敛 | 6–12 人日 |
| E：大规模 I/O、回代与基准 | 二进制输入、批次恢复、内存调度、百万数据测试与千万预算 | 峰值内存受预算控制；每块<=1000；按目标谱量与误差比较；千万级未实测时明确标注 | 3–6 人日 |

从 ordering 到尺寸受限树的首版约一周量级；让细粒度树在整个 AMLS 工程中稳定且适合百万/千万规模，粗估共 19–36 人日。不能把前者的完成等同于后者。

首轮默认选 exact、`maxBlockDofs=1000`，保留旧 nd 基线。不要立即开启 relaxed 合并、零谱默认或宣称性能收益；先有构树正确性、代数一致性，再测精度和实际瓶颈。

### 8.1 必需测试矩阵

1. 路径/三对角 SPD：真单子链，防止再次无上限收缩。
2. 稠密 SPD 团：强制自然超节点超限，拆成链；含全祖先耦合。
3. 星形/多叉图：验证 1、2、3 个及更多孩子，无二叉假设。
4. 多个连通分量加孤立索引：森林、多根、单点根与空工作队列。
5. K/M 结构不相同：并集图能通过；只用 K 的非法边应被明确拒绝。
6. 满足列计数差 1 但父列/模式不相容的小矩阵：不得误合并。
7. 局部区间空、零谱叶/中间/根节点：未覆盖 K/M 的正确回退，零维投影与传播。
8. 2D/3D 小规则图：用独立符号消去 oracle 核对填充、树、块模式与更新范围。

对小矩阵显式构造变换 T，核对 `Tᵀ Kp T` 的块对角性、`Tᵀ Mp T` 与存储块一致，以及全保留谱/回代与直接广义特征解一致。仅 `tree.validate()` 通过不足以证明数值正确。

### 8.2 实际构建/测试入口

当前 CMake 定义的是 `unit_tests` 和 `regression_tests`，不是 README 旧例子中的分目录可执行名。本机 build cache 为 Ninja/Release/MinGW。

```bash
cmake --build build --target unit_tests
ctest --test-dir build --output-on-failure -R '^unit$'
```

开发过程中先运行相关 Catch2 标签与新增小矩阵用例；部分已有全谱管线用例规模较大，不应每次符号调试都重跑。阶段完成后执行完整 unit 与适用 regression，回归数据缺失要明确记录。本文未改求解器，因此本次没有重建或重跑数值测试。

### 8.3 大规模比较应输出什么

以用户当前约 10 万 DOF 输入作为第一批实际数据；百万和千万应分别取得对应数据再测，不能从 10 万运行时间线性外推。

每次保存：

```text
input rows/nnz、结构图分量数、排列及符号版本
natural exact blocks、最终 bounded blocks、split 次数
owned DOF min/p50/p90/p99/max
update width/front width 的可得统计与估计口径
depth/max child count/单子节点比例/每层宽度
symbolic nnz(L) 估计、活跃 block-pair 数
每节点 k_i 分布、最终 p、projected nnz
ordering/symbolic/build/eliminate/local-eigen/projected/recover 时间
峰值进程内存、在途任务工作区
特征值误差、原问题残差、M-正交误差、失败原因
```

硬验收：所有非空计算块自身 DOF `<=1000`；排列/全覆盖/祖先耦合检查无异常；全保留小例子代数等价；截断后达到既定精度；内存预算不足时在大分配前拒绝并说明原因。

## 9. 关于现有 Abaqus 小块统计的证据限制

仓库 [`cant_31_消去树.md`](abaqus_ams/cant_31_消去树.md) 记录 98,304 DOF、29,488 块、块宽 3–24、362 个根。其生成脚本 [`abaqus_stf_tree.py:33`](../tools/abaqus_stf_tree.py#L33) 用“相邻行非零数下降 1”识别块，再以最小更高列推父块。

**这些是该解析方法得到的候选块统计，不能直接当成已验证的 Cholesky exact supernode tree。** 行非零数差 1 没有验证列集合相同，私有 dump 也未证明就是完整 filled U=Lᵀ 模式。

本次对现有 `bench_abaqus/cant_31_stf` 做了只读校验，沿用该脚本的解析与候选划块。输入大小 48,912,660 bytes，SHA-256 为 `758a76a17db6e6f025a2cbc373c5df15c5ad0d1834b5a5bdc371360596328bfd`。相邻模式检查覆盖候选块内全部相邻行；祖先检查由候选父森林的迭代 DFS 入/出时间区间判定，不只比较编号或层数：

| 检查项 | 本次结果 |
|---|---:|
| 解析行数 / 非零数 | 98,304 / 3,688,476 |
| 原脚本候选块 / 候选根 | 29,488 / 362 |
| 块内相邻行满足“首列为下一行，去掉该列后模式等于下一行” | 65,662 对 |
| 块内相邻行不满足上述条件 | 3,154 对 |
| 候选块高向耦合数 | 357,552 |
| 不落在候选 parent 链祖先上的高向耦合 | 242,357 |

若将该 dump 当作完整 filled U，这些失败不符合 exact block/etree 的解释；若它是省略、编码或另一阶段的算子，则需要先解释其格式和语义。**本次结果不证明 Abaqus 算法有问题，只证明现有候选树重建还不足以支撑“任意层节点都只有 3–24 DOF”或“362 个真实独立根”的结论。** 也不能据此预测百万/千万规模的超节点大小。

新设计以公开符号算法和当前原始 K/M 为依据。Abaqus 对比保留为外部研究，后续若取得原始排列、官方 SuperNodeSizes/Parents/FrontNodes 或可验证 factor pattern，再做独立交叉核验。此次没有改写旧报告或解析工具。

## 10. 参考依据与设计边界

- 本仓库当前 [`metis_partitioner.cpp`](../src/partition/metis_partitioner.cpp)、[`separator_tree.cpp`](../src/amls/separator_tree.cpp)、[`solver.cpp`](../src/amls/solver.cpp)、[`node_task_executor.cpp`](../src/amls/node_task_executor.cpp)：当前行为的直接证据。
- Vendored [`METIS ometis.c`](../third_party/metis/libmetis/ometis.c)：NodeND 输出与排列方向。
- 本机 SuiteSparse `cholmod.h`：`analyze_ordering` 的 Parent/Post/ColCount 输出；symbolic factor 存储字段；relaxed amalgamation 条件。该依赖尚未在本工程链接验证。
- [CHOLMOD 官方用户指南](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Doc/CHOLMOD_UserGuide.pdf)。
- [CHOLMOD 官方 supernode symbolic 实现](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Supernodal/cholmod_super_symbolic.c)：fundamental 判据与 relaxed 合并规则；开发时应固定版本，不能依赖 dev 分支行号不变。
- Davis & Hager, *Dynamic Supernodes in Sparse Cholesky Update/Downdate and Triangular Solves*, ACM TOMS：exact/fundamental/relaxed supernodes 的算法背景。

最终建议是：**新增“NodeND 指定排序 → 符号分析 → exact/fundamental 风格分组 → 最大 1000 的连续拆分 → 保留单子链的消去块树”路线；并同步补齐细树下的谱截断验证、稀疏投影与内存预算。** 自身 DOF 上限可以严格验收；运行速度、填充量与精度必须用实际输入测量，不能由树名称或块宽上限承诺。
