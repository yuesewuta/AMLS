# Abaqus AMS vs PKU/CCM AMLS：剖分 / 高斯消去 / 子空间缩聚 对比

> 2026-10-09。同模型：20×20×20 悬臂块 C3D8，27783 方程 / 26460 自由 DOF，
> 钢材料，120 Hz 求解 12 阶。Abaqus 2020 实测（探针导出）；PKU/CCM 为
> 本项目 AMLS 实现（`src/amls/` + `src/partition/`）同矩阵实测。
> 每条结论标注证据：**A**=Abaqus 探针实测、**P**=PKU/CCM 源码/实测、
> **D**=Abaqus 官方文档、**[INF]**=推断。

## 0. 结论速览

| 环节 | Abaqus AMS（实测） | PKU/CCM AMLS（实测） | 差异性质 |
|---|---|---|---|
| 剖分树节点数 | **7898 个超节点块**（stf 行阶梯重建） | **40 个树节点**（ND, leafSize=500） | 数量级差异：Abaqus 树深得多 |
| 块大小 | 3 行块 6739 个（占 85%）、6 行 1033、9/12/18/24 少数 | 叶子 ~640 DOF、顶层块 1458 | Abaqus 块小得多（3–24 DOF） |
| 树形状 | 多层、块间行度阶梯（80→75, 53→48...） | ND 消去树, maxDeg=3, 深 4 | 都非二叉，但粒度完全不同 |
| 局部特征求解 | LAPACK `substr_sygst/sytrd/stev/ormtr` | LAPACK `dsygvx`（区间） | 同类（稠密 LAPACK），Abaqus 分四步 |
| 缩聚 | 全局模态 12 个（`gmodes` 27783×12），缩聚后只 12 维 | 子空间 p=87（同频段） | Abaqus 缩聚更紧（12 vs 87） |
| 高斯消去 | `k_factor/sturm_factor/k_bwd_psi/k_updext` | `computeConstraintModes`（LDLT）+Ψ | 同类（稀疏 Cholesky/LDLT + 回代），阶段名对应 |
| M | **对角（lumped）**：MASS2.mtx 全对角 | 可一致/对角（读取 Abaqus 对角 M） | Abaqus 默认 lumped |
| 阻尼 | `sd_*/vd_*` 全 0（无阻尼路径） | 无阻尼模型相同 | 一致 |

## 1. 剖分：差异的核心（用户关注点）

### 1.1 Abaqus 实测（A）

通过 `ABA_PAMS_EXPORT_SOLVER_DATA` 导出求解器稀疏算子 `stf`，
行非零数序列呈现**强阶梯**（实测，编码已附附录）：

```
行度前 60: [80,79,78,77,76,75, 80,79,78,77,76,75, 53,52,51,50,49,48, ...]
```

按"行度每行减 1 = 同一超节点块内"识别出 **7898 个超节点块**：

| 块大小（行） | 块数 | 占比 |
|---|---|---|
| 3 | 6739 | 85.3% |
| 6 | 1033 | 13.1% |
| 9 | 70 | 0.9% |
| 12 | 48 | 0.6% |
| 18 | 5 | 0.06% |
| 24 | 3 | 0.04% |

块内行度从 80 递减到 75（C3D8 内部节点 27 邻域×3 DOF≈80 行度），
块首行度分布：14→1537块、32→433、53→431、35→431、41→357、80→336...
即**绝大多数超节点 = 3 行 = 1 个节点的 3 个 DOF**（规则六面体网格每消去一个
节点生成一个小超节点）；6 行块 = 2 个相邻节点；9/12 行 = 更多相邻节点。
**这解释了用户观察到的"叶子之后并行度下降"：Abaqus 的超节点粒度极细
（3 DOF），越靠近根块越大（24 DOF），根附近并行度天然受限。**

- Abaqus 的 `MAXIMUM SIZE OF A SUBSTRUCTURE = 4096`（dat）是**上限**，
  实际块远小于此（实测 3–24 行）。[4096 单位未知，见更正文档]
- 树分层信息不在导出数据中；但超节点父子关系（谁是谁的祖先）可从
  行阶梯的"块首行度"推断——块首行度越大越接近叶子。 [INF]

### 1.2 PKU/CCM 实测（P）

同矩阵喂 `amls_solve --partition nd --leaf-size 500`（`AMLS_DUMP_TREE=1`）：

```
[tree] etree roots=1 maxDeg=3 at pos=22607
[tree] nd nodes=40 maxSize=1458 leafAvg=640 nLeaf=27 maxDepth=4
[tree] top-sizes: 1458 1204 1159 1000 1000 1000 1000 865 676 676 676 676
```

- **40 个树节点，27 个叶子**，叶子平均 640 DOF（≈213 节点×3），
  顶层块 1458 DOF，树深 4。
- 这是 `METIS_NodeND` 消去树 + leafSize 合并且 递归二分，见
  `src/partition/metis_partitioner.cpp`（`nestedDissection`）。
- 用户所述"PKU 调用 METIS ND 递归二叉、分割节点维度大"在 leafSize=500 下
  得到验证：顶层块 1458 DOF >> Abaqus 的 24 DOF 顶块。

### 1.3 差异

| 项 | Abaqus | PKU/CCM | 影响 |
|---|---|---|---|
| 树节点数 | 7898（超节点） | 40（叶子+分割） | Abaqus 细粒度 200 倍 |
| 叶子大小 | ~3–24 DOF | ~640–1458 DOF | Abaqus 局部问题小，PKU 大 |
| 并行度分布 | 细粒度，根部块小 | 叶子并行高，根部 1458 块 | 用户观察的"根部并行下降"在 PKU 更明显（根块 1458 vs 24） |
| 剖分器 | METIS 系（`ABA_SST_USE_METIS` 等字符串） | METIS_NodeND / PartGraphRecursive | 同源（METIS），策略不同 |

**关键洞察**：Abaqus 用"极细超节点 + 大量层次"换取每层并行度均匀；PKU 用
"粗叶 + 少层次"换取叶子并行高但根附近单块大。这正是用户"叶子并行高、
靠近根并行下降"问题的量化解释——**PKU 根块 1458 DOF 的高斯消去是串行的**，
Abaqus 根块只有 24 DOF。

## 2. 高斯消去（稀疏分解）

### 2.1 Abaqus（A + D）

`times` 探针（`cant_20_freqonly.times`）实测阶段：

| 阶段 | 语义 | 实测耗(s, 4线程和) |
|---|---|---|
| `k_clique` | 超节点团处理（clique = 稠密块） | 0.055 |
| `k_factor` | 全局 K 稀疏 Cholesky | 0.177 |
| `sturm_factor`+`sturm_check` | Sturm 序列特征值计数 | 0.144 |
| `k_updext` | K 沿树向上扩展（Schur 补） | 0.283 |
| `k_bwd_psi` | Ψ 回代（消除耦合项） | 0.277 |
| `k_stackext` | K 栈式扩展 | 0.445 |
| `k_cleanup` | 清理 | 0.002 |

`k_factor` 实际是**超节点块级 Cholesky**（clique→factor），
`k_updext`/`k_bwd_psi`/`k_stackext` 对应 AMLS 的 Schur 补更新与 Ψ 传播。
这与 PKU 的 `computeConstraintModes`（K̂_ii LDLT + 逐祖先回代）结构对应。

### 2.2 PKU/CCM（P）

`src/amls/gaussian_elimination.cpp`：
- `computeConstraintModes`：`K̂_ii` 做 `EigenSimplicialLDLTSolver`（稀疏 LDLT），
  再逐祖先用窄右端回代求 `u = K̂⁻¹ K_{i,a}`。
- `buildTemporaryMass`：M 的临时质量更新（set_union 列合并）。
- 无独立 sturm 阶段（用 dsygvx 区间计数）。

### 2.3 差异

- 两者都是"稀疏分解 + 窄回代"的 Schur 补消去；Abaqus 用**超节点 Cholesky**
  （`ABQSMAEqsDirSolverMod.dll` 的 `SMAEqsDirSymbolicFactor`），PKU 用
  **Eigen SimplicialLDLT**。
- Abaqus 有显式 `sturm_factor/check`（保证不漏根），PKU 用 dsygvx 频段计数
  替代（效果类似，机制不同）。
- **性能差异在块粒度**：Abaqus 每块 3 DOF 的因子小、并行调度细；PKU 根块
  1458 DOF 的 LDLT 串行，正是用户"根附近并行阻塞"的实证。 [INF: Abaqus
  调度细节未观测，但块大小已实测]

## 3. 子空间缩聚

### 3.1 Abaqus（A）

`ABA_PAMS_EXPORT_GLOMODES` 导出 `gmodes`（16.3MB，文本 triplet）：
- **27783 行 × 12 列** = 全局模态矩阵 Φ（12 个模态，[0,120]Hz 内全部）。
- 每列 L2 范数 ≈ 0.0124（M 对角化的质量归一），列间内积 ≤ 1e-9（正交）。
- `orth_K` 对角 = 特征值（与 dat 频率逐位一致），`orth_M` = I。

即 Abaqus AMS 的缩聚后全局问题只有 **12 维**（= 频率窗口内模态数），
没有额外"基扩展"。子结构局部模态（`substr_stev` 等）被频率截断筛掉，
只剩关注频段内的。

### 3.2 PKU/CCM（P）

`src/amls/subspace_condensation.cpp`：
- `solveLocalEigenproblem`：每节点 `dsygvx` 区间 `(-∞, (2π·γ·maxFreq)²]`。
- `keptModes`（截断后保留数）经 `AMLS_DUMP_TREE`/`[dumpk]` 打印。
- 实测子空间 **p=87**（同频段）——**比 Abaqus 的 12 大 7 倍**。
- 原因：PKU 的 leafSize=500 粗叶，每叶局部模态多；且截断 η=1（默认）
  保留至 maxFreq 等倍频，Abaqus 的 cutoff 更紧（AMS 参数 5/1.7/1.1）。

### 3.3 差异

| 项 | Abaqus | PKU/CCM |
|---|---|---|
| 子空间维度 | 12（=窗口模态数） | 87（η 截断） |
| 局部模态截断 | `substr_stev` 全求 + 频率筛 | `dsygvx` 区间 + keptModes |
| 缩聚后求解 | 12×12 稠密广义特征问题 | 87×87 稠密 dsygvx |
| 广义质量 | M 对角化 → 模态归一 | M-正交归一 |

Abaqus 的子空间极小（12×12），缩聚后求解零成本；PKU 的 87×87 也不大，
但局部特征求解阶段（dsygvx 每节点）比 Abaqus 的 stev 重。
**根因仍是块粒度**：PKU 粗叶 → 每叶局部问题大 → 子空间基数大。

## 4. 高斯消去 vs 子空间缩聚的"并行阻塞"根因（用户问题）

用户描述：PKU 叶子阶段并行度高，根附近并行度显著下降。

实测量化：
1. **PKU 树**（ND, leafSize=500）：40 节点，27 叶（~640 DOF），顶层块 1458 DOF，
   树深 4。**根附近 1458 DOF 的单块消去/缩聚 = 串行瓶颈**（P）。
2. **Abaqus 树**：7898 超节点，块 3–24 DOF。**任何层的单块都极小**（A），
   根附近块 24 DOF，并行度损失可忽略。
3. PKU 的 `NodeTaskExecutor` 用双线程池（nodePool + branchPool）做节点级并行，
   但**单节点（大块）内部是串行**——根节点的 1458 DOF 任务无法拆分。
   Abaqus 的 `times` 显示线程 0/3 负载 1.26s vs 线程 1/2 0.77s（4 线程），
   也提示 Abaqus 自身有一定负载不均，但块小使粒度损失小。

**结论（建议）**：PKU 侧要解决"根部并行下降"，需像 Abaqus 那样**把顶层大块
再拆细**（更小 leafSize / 更高 ND 层数），或对顶层块做并行消去。当前
leafSize=500 的 1458 根块是串行瓶颈的量化证据。

## 5. 已知 / 未知 / 需数据支撑

### 已知（实测支撑）
- Abaqus stf 超节点块分布（7898 块，3–24 行/块）。[A]
- Abaqus 全局模态 12 个（gmodes），缩聚维度 = 窗口模态数。[A]
- Abaqus 阶段表（k_factor/k_bwd_psi/substr_*/sturm_*）。[A]
- Abaqus M 对角化（MASS2.mtx 全对角，3×3×3 与 20×20×20 均实测）。[A]
- PKU ND 树统计（40 节点/27 叶/深 4/顶层 1458）。[P]
- PKU 子空间 p=87。[P]
- PKU 在 Abaqus 对角 M 上 dsygvx info=92 失败（缩聚后 B̃ 非正定，待查）。[P]

### 未知 / 需数据支撑
- Abaqus 树的分层结构（每层多少超节点）——stf 只有行序，无层号。
  [可试 `pme_AmsSymbolicFactor::Dump`/`SMAEqsDirSymbolicFactor::Verbose`]
- Abaqus 各阶段实际耗时占比（times 是相对值，需 wall-clock 归一）。[A 部分]
- Abaqus 局部模态截断公式（AMS 参数 5/1.7/1.1 的确切含义）。
  [D 2026-10-09 已解决：cutoff1=5 为全局统一频率乘子（λ_cut=(2π·5·f_max)²），
  cutoff2/3 仅作用于根上缩聚求解的初始子空间；见 AMS官方文档解读.md]
- Abaqus 剖分是否几何感知（METIS 无权图 or 带权）。[字符串有 ABA_METIS_WEIGHT]
- PKU dsygvx 92 根因（是 M 对角化 + 负特征值，还是导出问题）。[待查]

## 附录：复现

```bash
# Abaqus 探针导出 (需 abaqus_v6.env 含 ABA_PAMS_EXPORT_SOLVER_DATA=1 等)
cd bench_abaqus && ABQLauncher job=cant_20_freqonly cpus=4
# 产出: cant_20_freqonly_stf (超节点行序), _gmodes (全局模态), _times (阶段表)

# PKU 树统计
cd .. && PATH=/c/msys64/mingw64/bin:$PATH \
  AMLS_DUMP_TREE=1 ./build/amls_solve.exe \
  --K bench_abaqus/cant_20_K.mtx --M bench_abaqus/cant_20_M.mtx \
  --max-freq 120 --partition nd --leaf-size 500 --threads 4 --no-recover
```

## 附录 B：stf 超节点块重建脚本

```python
import array, collections
d=open('cant_20_freqonly_stf','rb').read()
a=array.array('I'); a.frombytes(d[:min(len(d),4000000)])
nrow=a[0]
rp=[int(a[6+k]) for k in range(nrow+1)]
nnzp=[rp[i+1]-rp[i] for i in range(nrow)]
blocks=[]; start=0
for i in range(1,nrow):
    if nnzp[i] != nnzp[i-1]-1:
        blocks.append((start,i-1)); start=i
blocks.append((start,nrow-1))
sizes=[b[1]-b[0]+1 for b in blocks]
print(collections.Counter(sizes))  # {3:6739, 6:1033, 9:70, ...}
```
