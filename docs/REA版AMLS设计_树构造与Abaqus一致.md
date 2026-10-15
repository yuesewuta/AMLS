# REA 版 AMLS 设计：树构造与 Abaqus AMS 完全一致 + 蒸馏求解

> 2026-10-13。本文档基于对 Abaqus AMS（PAMS）的完整二进制逆向
> （`docs/abaqus_ams/AMS求解流程与参数机制逆向分析.md`），设计一个
> **树构造与 Abaqus 完全一致**的 AMLS 实现（代号 **REA-AMLS**），
> 参考当前 PKU 版架构（`src/partition/` + `src/amls/`）复用其成熟模块。
> 证据标注：[R]=逆向反编译，[A]=探针实测，[P]=本仓库 PKU 实现。

---

## 1. 背景：为什么必须"树构造完全一致"

### 1.1 PKU 现状与实测差距 [P][A]

同一模型 cant_20（27783 DOF，C3D8 悬臂块）：
- **Abaqus 实际产物**（stf 探针重建，[A]）：**29488 块（cant_31 98k DOF）/ 7898 块（cant_20）**，
  块大小 **3/6/9/12/18/21/24 DOF**，超节点块内行度递减 1（梯形），单子链 167 级最深。
- **PKU supernode 路径**（maxBlockDofs=24，实测 [P]）：**26108 块 / 全部 3 DOF / 24009 单子链**，
  求解 `std::bad_alloc` —— 3-DOF 块 + 超长单子链导致每块独立存储 Ψ/局部矩阵，内存爆炸。

**根因**：PKU `buildBlockForest` 只有"exact 分组 + 拆大（maxBlockDofs）"，**没有聚合"小"**；
Abaqus 有完整的**自底向上聚合链**（Collapse2 折叠 + AggregateLeaves 叶子聚合 +
大小上限），把小超节点合成 3–24 DOF 的实用块，既保稀疏又控规模。

### 1.2 Abaqus 树构造的完整链路（逆向确认）[R]

```
┌─ 1. 排序 ────────────────────────────────────────────────────────────┐
│ std_OrderingHandler::GenerateOrder → slv_SolverOptimizer::NewOrder    │
│   （内部 METIS_NodeND / ParMETIS，受 ABA_FORCE_PARMETIS 开关）        │
│ 输出: 排列 perm[new]=old                                              │
│ ★ 网格节点图收缩: NodeND 在网格节点图（每节点 dp=3 DOF 收缩为一个顶点） │
│   上跑，DOF 排列 = 节点排列按 dp 展开 —— fca 精确嵌套块全为 3 的倍数   │
│   （cant_20: 3:9143/6:48/9:2/12:4，合并率 ~0.6%）即为证据              │
├─ 2. 标量消去树 etree ────────────────────────────────────────────────┤
│ std_BuildTreeAfterMetis（std_pamsDriver 阶段 "BUILD TREE AFTER METIS"）│
│   用 Liu 经典算法对排序后邻接图建 etree（prev[] 链归并）[R: 13638-67] │
│   处理保留/残差节点标记；统计 numberOfTrees/numComponents（森林）      │
├─ 3. 符号分解 + 基础超节点 ───────────────────────────────────────────┤
│ pme_SymbolicFactor::BuildTree（再建 etree）+ CreateFundamentalSuperNodes│
│   exact 判据: parent(j)=j+1 && 列模式相同 → 同一基础超节点             │
│   ★ 以网格节点为原子：节点层符号分析（A_n = Rᵀ A R，dp×dp dof 块求和， │
│   R 为节点→dof 聚合，SPD 保持），基础超节点以节点为单位再展开为 dof 块  │
│   若在 dof 层做消除分析，导出 mtx 丢失的组内小耦合会让三重奏被拆开      │
│   （cant_20 实测: 1495/8820 节点缺 (0,1)/(0,2) 对、7 组全缺）          │
├─ 4. 聚合（本设计的核心引入项）───────────────────────────────────────┤
│ Collapse2:      子数==1 且总大小<LEAF_SN_LIMIT(120) 的节点折叠进父     │
│                 （ABA_PAMS_SN_COLLAPSE=2 控制）[R: Collapse2 101821]  │
│ AggregateLeaves: 子全为叶 且 聚合大小<LEAF_AGGR_LIMIT(20) → 聚合成叶   │
│                 （ABA_PAMS_LEAF_AGGR_LIMIT 默认 0x14=20）[R: 54412]   │
├─ 5. 大小限制 ────────────────────────────────────────────────────────┤
│ UPPER_CUTOFF=4096 / LOWER_CUTOFF=1000 / LOWER_CUTOFF_LEAF=1000 /      │
│ UPPER_CUTOFF_LEAF=1800 / MIN_SN_SIZE=80 / LEAF_SN_LIMIT=120           │
│ （dat 打印 "MAXIMUM SIZE OF A SUBSTRUCTURE 4096" 即 UPPER_CUTOFF）[R]  │
└─ 6. 子树 ─────────────────────────────────────────────────────────────┘
 pme_AmsSymbolicFactor::ComputeSubTreesNew → pme_TrimSubstructureTree
   → pme_SmpReducedEigenSolver（缩聚求解用修剪后的子树）
```

### 1.2.1 网格节点图收缩的实现（`tree_builder.cpp`）

**证据链**：Abaqus cant_20_fca.stf（真因子副本，二进制格式）的精确嵌套块
分布 3:9143/6:48/9:2/12:4（9197 块、27783=9261×3 DOF，合并率 ~0.6%）全部为
3 的倍数 → NodeND 的顶点是网格节点（每节点 3 DOF）而非单个 DOF；基础超节点
以节点为原子。

> 注意: 早期文档引用的 stf 文本块分布（cant_20: 3:6739/6:1033/9:70/12:48/
> 18:5/24:3）来自二进制 stf 解析脚本的"行度阶梯"启发式，实测是过度合并的
> 伪影（行度斜坡巧合），并非 Abaqus 的真实基础超节点；以本节 fca 精确嵌套
> 数字为准。

**实现**（`opt.dofsPerNode`，0=自动/1=逐 dof/3=强制）：
1. 自动检测：`n % 3 == 0` 且组内耦合边覆盖率 ≥99%（Abaqus 导出 mtx 会丢弃
   数值极小的跨向耦合，cant_20 中 1495/8820 节点缺 (0,1)/(0,2) 对、7 组全缺；
   缺失耦合由消元填充补回，不影响 3-DOF 超节点形成）→ dp=3，否则逐 dof；
2. METIS_NodeND 跑在节点图上（节点邻接 = 任意 dof 边的节点级去重）；
3. 节点层符号分析：`A_n = Rᵀ A R`（dp×dp dof 块求和；A SPD ⇒ A_n SPD，
   可安全数值 factorize），exact 判据在节点列上判 → 基础超节点节点对齐；
4. 节点块 [b,e) → dof 块 [dp·b, dp·e)（展开后连续），blockOfColumn 同步
   重建为 dof 层；下游 Collapse2/AggregateLeaves/4096 截断全在 dof 单位。

**实测（cant_20, 26460 dof）**：base 3:8788 + 6:16（8804 块，全部 3 的倍数；
对比 Abaqus fca 9197 块同形态）；聚合后 2601 块（3:1848/6:290/9:78/12:94/
…最大 120），K 全边祖先可比计数=0。cant_31（95232 dof）: base 3:31670/6:7/
9:20（31697 块 vs Abaqus fca 29488 块），聚合后 8859 块，可比计数=0。

### 1.2.2 基础超节点合并判据的标定（fca + 双模式）

Abaqus 真实合并判据的标定结论（2026-10-15，证据见 `tools/abaqus_stf_tree.py`
与 fca 二进制解析）：

1. **fca 精确嵌套块 = 真基础超节点**：对 `cant_10_fca.stf`/`cant_20_freqonly_fca.stf`
   逐行验证精确嵌套条件 row(r)==row(r+1)∪{r+1}，得 cant_10: 1308 块
   {3:1293, 6:11, 12:4}、cant_20: 9197 块 {3:9143, 6:48, 9:2, 12:4}——
   合并率仅 ~1%，无大块。
2. **判据是 K 图邻域嵌套而非因子填充嵌套**：把我们的结构规则
   （parent(j)=j+1 且 col(j)\{j,j+1}==col(j+1)\{j+1}）施加在 **Abaqus 自己的
   因子**上复现其真值（cant_10 得 1309 vs 1308，节点层）；而因子填充在规则
   网格区会产生长等结构链（cant_10 dof 层曾出现 363-DOF 巨链），与 Abaqus
   行为不符。故默认模式源为**排列后输入矩阵下三角（K 图，无填充）**。
3. **双模式开关 `AMLS_SUPERNODE_PATTERN`**：`k`（默认，Abaqus 式）、
   `factor`（L 填充嵌套，A/B 备选）。cant_20_K 实测: k 模式 base 8715 块
   {3:8614, 6:97, 9:4}（合并率 ~1.2%，与 Abaqus 同形态）；factor 模式
   4018 块 {3:2246, 6:201, 9:112, 12:1459}（合并率 ~44%）。两模式聚合后
   树相同（2601 块），12 个模态逐位一致。
4. **链长上限 `maxChainNodes=4`**（=12 DOF，恰为 Abaqus fca 的最大块）：
   防御性上限，K 模式下自然合并本就 ≤4 节点，factor 模式下截断填充长链。
5. **教训（1e36 罚对角）**：Abaqus 导出的 STIF2 全矩阵含 1e36 罚对角
   （约束 dof）。用它做求解输入会让缩聚产生 λ≈-1e18 的垃圾特征值——
   **与树判据无关**（HEAD/任何模式均复现）。求解验证必须用无罚项的
   导出矩阵（cant_20_K.mtx / cant_20_M.mtx，26460 dof）。

### 1.3 树构造一致性判据（验证标准）

1. **块 DOF 分布接近**：cant_20 应产 ~3/6/9/12 DOF 块，主峰 3 DOF、合并率
   ~1%（K 模式，与 Abaqus fca 实测一致）；
2. **无 bad_alloc**：块数不爆炸、单子链被折叠收敛；
3. **频率精度**：与 Abaqus 参考解（`cant_20_freqonly.dat`）在容差内一致；
4. **块内行度梯形**：每块内行非零数递减 1（stf 行度序列 80→79→78…）。

---

## 2. REA-AMLS 架构

### 2.1 总体结构（复用 PKU，新增聚合+蒸馏）

```
src/rea/                          # REA-AMLS 新增目录（不动 PKU 原有代码）
├── amls_rea.hpp/cpp              # 顶层求解入口（对 Pencil 求解，同 AmlsSolver 接口）
├── tree_builder.hpp/cpp          # ★ Abaqus 风格树构造（排序→etree→超节点→聚合）
├── aggregation.hpp/cpp           # ★ Collapse2 + AggregateLeaves + Cutoff 限制
├── distill_solver.hpp/cpp        # ★ 蒸馏+子空间迭代+Ritz（缩聚特征阶段）
└── common.hpp                    # 类型别名、从 PKU 复用的接口声明
```

### 2.2 模块复用清单（从 PKU 直接复用）

| PKU 模块 | 复用方式 |
|---|---|
| `partition/symbolic_analyzer.{hpp,cpp}` | ✅ 直接用：NodeND 排列 → etree + columnCount |
| `partition/supernode_builder.{hpp,cpp}` | ✅ 复用 `findExactRanges`/`splitContiguous` 作为基础分组，**新增聚合阶段** |
| `partition/metis_partitioner.cpp` | ✅ 复用 `supernodePartition` 的 NodeND 调用 |
| `partition/separator_tree.{hpp,cpp}` | ✅ 直接用（树数据结构 + 不变量校验） |
| `amls/solver.cpp` step2 N1–N5 高斯消去 | ✅ 复用 GaussEliminationModule/NodeTaskExecutor |
| `amls/*` 缩聚组装/恢复 | ✅ 复用 EigenPairSaver/SubspaceMassSaver/step4 |
| `linalg/colcomp_mat` / `dense_eigensolver` | ✅ 复用 |
| `io/matrix_market` / `core/types` | ✅ 复用 |

### 2.3 新增模块设计

#### (a) `aggregation.hpp` —— Abaqus 风格树聚合 ★核心

输入：基础超节点森林（exact 分组 + splitContiguous 拆分后的 `BlockForest`）。
输出：聚合后的森林（块数大幅减少、单子链收敛、块大小 ∈ [3, upperCutoff]）。

```cpp
struct AggregationParams {
    int64_t collapseMaxDofs = 120;   // Abaqus LEAF_SN_LIMIT: 单子链折叠上限
    int64_t leafAggrMaxDofs = 20;    // Abaqus LEAF_AGGR_LIMIT: 叶子聚合上限
    int64_t maxBlockDofs   = 4096;   // Abaqus UPPER_CUTOFF: 子结构上限(dat 打印值)
    int64_t minBlockDofs   = 80;     // Abaqus MIN_SN_SIZE
    int collapseMode       = 2;      // Abaqus SN_COLLAPSE
    bool  collapseL2R      = false;  // Abaqus COLLAPSE_L2R
    bool  aggregateLeaves  = true;   // Abaqus AggregateLeaves 开关
};

// 返回聚合后的森林（保持后序编号 + 祖先 id>后代 + DOF 覆盖不变式）
BlockForest aggregateAbaqusStyle(const BlockForest& base,
                                 const AggregationParams& p);
```

**算法**（自底向上沿后序，与 Abaqus `Collapse2`/`AggregateLeaves` 一致）：

1. **Collapse2 折叠**：对每个"恰好 1 个子且聚合总大小 < collapseMaxDofs"的非根节点，
   将其（连同子树）并入父节点（DOF 追加到父，父子关系重链）。
   重复至不动点（消除单子链，链长收敛到短链）。
   - Abaqus 逻辑 [R]：`if (子数==1 && 自身+子树大小 < LEAF_SN_LIMIT) → 折叠`。
2. **AggregateLeaves 叶子聚合**：对每个"所有子都是叶子（无孙）且聚合大小
   < leafAggrMaxDofs"的节点，把全部叶子子节点并入自身成为一个复合叶节点
   （DOF 合并），消除"单节点多小花盆"的碎片叶。
   - Abaqus 逻辑 [R]：`if (∀子 是叶 && Σ子大小+自身 < LEAF_AGGR_LIMIT) → 合并`。
3. **Cutoff 限制**：任何聚合后块大小不得超过 maxBlockDofs（超限则拒绝合并保持原状）；
   minBlockDofs 以下的小块尽量参与聚合。
4. **不变量维护**：聚合后重算 children/parent、后序重编号（祖先 id>后代）、
   校验 DOF 全覆盖/无重复/祖先可比（复用 supernode_builder 的校验逻辑）。

**设计的树形效果（预测）**：
- 3-DOF exact 块（单节点 3 平动 DOF）保留为主块型；
- 相邻 2–8 个叶片聚合 → 6/9/12/18/21/24 DOF 块（与 Abaqus 实测分布 3:26760, 6:2418,
  9:115, 12:179, 18:2, 21:13, 24:1 同构）；
- 单子链绝大多数被折叠（167 级→短链），树收敛，内存不再爆炸。

#### (b) `distill_solver.hpp` —— 蒸馏+子空间迭代+Ritz ★缩聚特征阶段

对应 Abaqus `pme_SmpReducedEigenSolver`（[R] 65 方法清单）：
subtree 分组 → 子树特征 → 蒸馏截断 → 一次子空间迭代 → Ritz 分析 → 恢复。

```cpp
struct DistillOptions {
    int64_t  maxSubtreeDofs = 5000;  // 子树规模上限（对应 pme_AmsDataLight::DistLimit）
    double   distillCutoff = 0.0;    // 蒸馏截断频率（= cutoff2/cutoff3 相关 [R]）
    int      subspaceIter = 1;       // 子空间迭代次数（Abaqus 默认 1 次 [R]）
    bool     useRitz = true;
};

// 输入: (K̃,M̃) 缩聚斜交基下的全局小问题 + 树（已修剪）
// 输出: 前 k 阶特征对（在缩聚坐标下）+ Ritz 向量
struct DistillResult { std::vector<double> evals; DenseMatrix eVecs; };
DistillResult distillSolve(const SparseMatrix& Kt, const SparseMatrix& Mt,
                           const TrimmedTree& tree, const DistillOptions& opt);
```

**算法**（对应 [Kim04] 蒸馏 + SIM + Ritz，[R] 映射）：
1. `GroupSubstrsToSubtree`：按树结构把相邻子结构聚成子树（规模 ≤ maxSubtreeDofs）；
2. `SolveSubtreeEigenProblem`：每子树解 (K̃_sub, M̃_sub) 特征问题（复用 dense dsygvx）；
3. `TruncateEigenSolution`：按 distillCutoff 截断（保留 λ ≤ distillCutoff²·(2πfmax)²+shift）；
4. **蒸馏基 V₁** = 各子树保留特征向量拼接（
   `AssembleDistilMassBlocks/DistillationTask`）；
5. `OneSubspaceIteration`：`X = K̃_B⁻¹ M̃ V₁`（静态缩聚/稀疏解，同时 `UpdateRemainingVectors`）；
6. `SmpRitzAnalysisOnSubstrModalSpace`：在 X 上投影 (K̃,M̃) → 小 Ritz 问题 → 特征对；
7. 输出 Ritz 向量 → 恢复杂凑坐标下的全局模态（复用 PKU step4 的 Ψ+Φ 回代）。

> 说明：本仓库 PKU 目前对缩聚 (K̃,M̃) 直接稠密 dsygvx（`solver.cpp` step3）。
> REA 版以 `distillSolve` 替换/作为 step3 的可选后端，对应 Abaqus 的
> Reduced Eigensolution Phase 与 cutoff2/3 语义（[R] ApplyClassicCutoffs 的
> `param_1[0x30/0x38]` = cutoff2/3 进蒸馏截断）。

#### (c) `tree_builder.hpp` —— 树构造管线编排

```cpp
struct AbaqusTreeOptions {
    int64_t maxBlockDofs = 4096;      // UPPER_CUTOFF
    AggregationParams aggr;           // 聚合参数（默认 Abaqus 值）
    bool keepForest = true;           // 保留多根（Abaqus 支持森林 [R] numberOfTrees）
    int metisSeed = 42;
};

// 完整管线: NodeND → etree(analyzeSymmetrized) → exact 分组 → 聚合 → SeparatorTree
// 返回与 PKU supernodePartition 相同的 Result{tree, perm}
IPartitioner<T>::Result buildAbaqusTree(const typename T::SparseMatrix& A,
                                        const AbaqusTreeOptions& opt);
```

管线明确对应 Abaqus 各阶段（见 §1.2 图）：
`GenerateOrder`↔METIS_NodeND；`BuildTreeAfterMetis`↔analyzeSymmetrized(etree)；
`CreateFundamentalSuperNodes`↔buildBlockForest(exact)；`Collapse2+AggregateLeaves`↔aggregation；
`ComputeSubTreesNew`↔（缩聚阶段调用 distillSolver 前的 TrimmedTree 构造）。

### 2.4 求解流水线（与 PKU solver.cpp 对齐，替换剖分层）

```
solve(pencil, opts):
  // Step1 树：与 Abaqus 一致
  km = {K, M};  A = |K|+|M|（结构并集）
  part = buildAbaqusTree(A, abaqusOpt(supernodePathOpts))
  reordered = permuteSymmetric(km, part.perm)
  tree = part.tree（已满足全部不变式: 连续段/后序/祖先id>后代）

  // Step2 约化（复用 PKU 高斯消去 + 局部特征 + μ）
  OriginalBlocks ob(reordered, tree)
  GaussEliminationModule gauss(ob)
  NodeTaskExecutor run(...)     // N1/N3/N4/N5 + solveLocalEigenPair
  saver / massSaver / temp       // 特征对 / S_{d,a} / μ
  localCut = (2π·γ·fmax)²        // cutoff1 语义 [R]

  // Step3 缩聚特征（★ 蒸馏后端，替代直接 dsygvx）
  assemble K̃ (diag 特征值), M̃ (对角 1 + S 块)
  result = distillSolve(K̃, M̃, trimmedTree, {cutoff2, cutoff3, ...})

  // Step4 恢复（复用 PKU 回代 + perm 反排列）
  recovered = backTransform(result.eVecs, gauss, saver)
```

---

## 3. 与 Abaqus 参数机制的对应

| Abaqus 参数 | REA 版对应 | 语义 |
|---|---|---|
| `*FREQUENCY` nev/fmin/fmax | `--nev/--min-freq/--max-freq` | 与 PKU 一致 |
| cutoff1 (5) | `--eta γ` 或新 `--cutoff1` | λ_cut=(2π·cutoff1·fmax)²，子结构局部截断 [R] |
| cutoff2 (1.7)/cutoff3 (1.1) | 新 `--cutoff2/--cutoff3` | 蒸馏截断/初始子空间门槛 [R] |
| UPPER_CUTOFF (4096) | `--max-block-dofs`（默认 4096） | 子结构 DOF 上限（dat 打印值） |
| LEAF_AGGR_LIMIT (20) | `--leaf-aggr-dofs` | 叶子聚合上限 [R] |
| LEAF_SN_LIMIT (120) | `--collapse-max-dofs` | 单子链折叠上限 [R] |
| ABA_PAMS_GLOBALSHIFT | `--global-shift` | 特征移位 [R] |
| ABA_PAMS_LAST_CUTOFF(×10) | 内部：根层放宽 | 根子结构用更高截断 [R] |
| EnhancedReduction (50×fmax) | `--enhanced-cutoff`（默认关） | level≤3 子结构增强截断 [R] |

---

## 4. 验证协议

1. **树形态对比**（对 cant_20/31）：
   - 块 DOF 分布直方图 vs Abaqus stf 实测（3:6739, 6:1033, 9:70, 12:48, 18:5, 24:3）；
   - 单子链数量、最大深度（Abaqus 167 级）、峰值块大小（Abaqus 24）；
   - 结论判据：主峰 3 DOF、含 6/9/12 块、无 bad_alloc、K 边祖先可比。
2. **频率精度**：cant_20（120Hz，12 阶）对照 `cant_20_freqonly.dat` 参考谱，容差 1e-4 相对；
   并对照 PKU `--partition supernode --max-block-dofs` 与 `--partition nd`。
3. **性能/内存**：cant_31（98k DOF）墙钟 + 峰值内存，与 PKU nd 路径对比。
4. **回归**：`tests/` 现有单元测试保持通过（新代码在 `src/rea/` 独立成库，不破坏原有路径）。

---

## 5. 实现步骤

1. **`aggregation.{hpp,cpp}`**：聚合适配器（本设计 §2.3a）。输入现有 `BlockForest`，
   输出聚合森林。含自检（覆盖率/祖先可比/后序）。
2. **`tree_builder.{hpp,cpp}`**：编排 NodeND→analyze→buildBlockForest→aggregation→SeparatorTree。
   CLI 加 `--partition abaqus`（新入口）与聚合参数。
3. **`amls_rea.{hpp,cpp}`**：solver 层入口，复用 PKU step2/step4，step3 先保留
   PKU dsygvx 后端（验证树效果），再接入蒸馏。
4. **`distill_solver.{hpp,cpp}`**：蒸馏+SIM+Ritz（对应 cutoff2/3）。
5. **验证**：cant_20/31 树形态 + 频率 + 内存对比，出新对比表。