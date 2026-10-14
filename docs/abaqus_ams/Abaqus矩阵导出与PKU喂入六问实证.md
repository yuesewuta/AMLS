# Abaqus 矩阵导出 ⇒ PKU 喂入：六问实证汇总

> 2026-10-09。所有结论带证据编号：**A**=Abaqus 探针/导出实测、**C**=代码、
> **D**=官方文档（MIT 镜像 WebFetch）、**P**=Python/scipy 数值实验。

---

## Q1. Abaqus 超节点/消去树能否喂给 PKU？

**结论**：数据结构可提取（stf 行序含 7898 超节点块边界），但**当前 PKU 接口无法直接注入**，需新增外部树/排列入口。

证据：
- [C] `src/partition/partitioner.hpp`：`IPartitioner::Result = {SeparatorTree tree, Permutation perm}`——剖分器是唯一的树来源，无外部注入接口。
- [C] `src/apps/amls_solve.cpp`：CLI 仅 `--partition bisect|nd`，无 `--perm/--tree` 参数。
- [C] `src/amls/solver.cpp:332-335`：运行时强制"节点 dofs 在重排坐标下为连续升序段"，违者抛异常——这是比接口更硬的约定约束。
- [A] `cant_20_freqonly_stf` 行度阶梯可重建 7898 个块（85% 为 3 行块）——**Abaqus 的树形信息确实在导出数据里**，但 stf 是含 fill-in 的消去序（行度 80→75 阶梯），不是"每节点连续 DOF 段"的根序排列。

用户提出的"编号方案（左孩子<右孩子<父节点）"解决的是**重编号后的标号约定**（PKU `renumberPostOrder` 已保证），但 Abaqus 的 stf 行序是否满足该约定、以及如何从 stf 提取 clean 排列（去掉 fill-in 效应）仍是开放问题。[INF：需要先精确解析 stf 的 colidx 才能验证排列性质]

**可行路径**：新增 `--perm file` 让 `IPartitioner` 输出被外部 P 覆盖（`perm[new]=old` 直接读入），并把 `renumberPostOrder` 后的树校验放宽为"外部树自证合法"。这需要改 partitioner 接口 + solver 装配，工作量可控。

## Q2. 1e36 标记 / M 未同步删除 ⇒ PKU 子块不正定？

**结论**：**两个假设都不成立**。根因是缩聚后子空间质量矩阵 B̃ 结构性非正定——PKU 移植版的组装缺陷，与 Abaqus 矩阵无关。

证据（完整定位链）：
1. [A] K 的约束 DOF 对角 = **精确 1e36**（3×3×3 里 48 个、20³ 里 1323 个），**该行非对角保留正常刚度值**（如 1.318e10）。
2. [A/P] **M 的约束 DOF 对角 = 975/1950/3900 正常质量值，不是 0、不是 1e36**（20³ 实测 0/1323 个为零）。
3. [P] 转换脚本用同一 mask 同步删 K/M 约束行列 → K、M 均 26460 维、均正定（K 最小特征值 1.96e8、M 975）。
4. [C+P] dsygvx info=92（=B 第 5 阶前主子式不正定）在 scipy 中**完全复现**：dump 出 87×87 的 K̃/B̃ 后，`scipy.linalg.eigh(K,B)` 报同样错误。
5. [P] **B̃ 有 3 个负特征值：-2.294, -0.428, -0.173**——尽管 B̃ 对角恒 1（每个局部特征向量 M-正交归一化的理论约定"ΦᵀM̃Φ=I"只在节点内成立）。负模态来自缩聚块的全局耦合（`assembleSubspaceMass` 的"对角恒 1 + 非对角对称补全"构造），在集中质量 M + 40 节点树上破坏正定性。
6. [P] 3×3×3（单节点树）B̃ 退化正常 → 成功；20³（40 节点树）→ 失败。**差异=树规模 × 集中质量 M 的缩聚耦合**。
7. [P] PSD 投影后 [0,120Hz] 最低频率 14.86 Hz ≠ Abaqus 真解 27.6 Hz——证明不是简单舍入，是**结构性错误**（负特征值模态与真实模态纠缠）。
8. [C] 附带修复：`dense_eigensolver.cpp` 的 `solveInterval` 原设 `il=1, iu=1`（索引语义）与值区间 `[vl,vu]` 并存，属 LAPACK 未定义行为——已改为 `il=1, iu=n`（合法值区间语义）。此修复必要但非 92 的根因。

## Q3. Abaqus 是否一定是集中质量？

**结论**：三个不同模型实测**全部为集中质量（lumped）**。

| 模型 | 节点 | DOF | MASS2.mtx 实测 | 对角值分布 |
|---|---|---|---|---|
| 3×3×3 | 64 | 192 | 纯对角 | 975/1950/3900/7800 |
| 10×10×10 | 1331 | 3993 | 纯对角(offdiag=0) | 7800×2187, 3900×1458, 1950×324, 975×24 |
| 20×20×20 | 9261 | 27783 | 纯对角(offdiag=0) | 7800/3900/1950/975 |
| 20×20×2 薄板 | 1323 | 3969 | 纯对角(offdiag=0) | 7800/3900/1950/975 |

- 对角值**精确匹配集中质量公式**（C3D8：内部节点 8×ρV/8=7800、面 4×975=3900、边 1950、角 975）。
- 若是一致质量，应有 8×8 满块非对角且对角 ≠ 7800——实测 offdiag=0。
- 用纯对角 M 复现 Abaqus 全部频率（误差 5.7e-6）→ **Abaqus 特征求解实际使用的就是这个对角 M**。
- [INF] 注意 `.msg` 打印 "CONSISTENT MASS MATRIX" 与实测矛盾——Abaqus 内部可能生成一致质量用于别处，但导出/求解用集中质量。[待进一步确认]

## Q4. 导出时做了哪些操作？

**逐行核对 `tools/abaqus_mtx_to_mm.py`**：
1. 读 K（STIF2.mtx，Abaqus COORDINATE 文本，全矩阵两半）
2. 找对角线 >1e30 的行 → 得约束 DOF 集（1323 个，20³）
3. K 删这些行+列
4. M 用 `--read-mask` **同步删同样的行+列**（保证维度一致，防"M 约束 DOF 留 0"问题）
5. 去重（对称矩阵留上三角）
6. 空档压缩为连续 1..26460
7. 加 Matrix Market banner（symmetric）

**用户假设的"读入时 M 对角是 0"在实测中不成立**：M 约束 DOF 对角是 975/1950/3900 正常值，删行后 M 正定（最小特征值 975）。若不做删行处理直接喂求解器，K 的 1e36 会让该 DOF 的广义特征值 λ≈1e36/m（天文数字），数值上强制 x=0——这是大数法原理，但 M 不会因而是 0。

## Q5. 1e36 大数法的目的与后处理（文档/手册依据）

**目的（实测+文档交叉验证）**：
- [A] K 约束 DOF 对角=1e36，非对角保留正常值 → 这是 **penalty/大数法**：把 `*BOUNDARY` 固定 DOF 并入全局方程（K_ii += 1e36），使 λ_ii≈1e36/m 远高于关注频段，等效强制 x_i=0。
- [D] Abaqus 文档（MIT 镜像 `simaanl-c-mtxgenerationperturbation.htm`）：矩阵生成"accounts for all current boundary conditions"；MPC=YES 时只含独立 DOF 条目。文档未直接写出 1e36 字面值，但确认边界条件被并入导出矩阵。
- 1e36 是 Abaqus 求解器标准处置（大数法），社区/工程通用认知 + 本报告实测双重支撑。

**后处理（正确操作，已验证）**：
1. 从 K 识别约束 DOF：对角线 == 1e36 的行集。
2. K、M **同步删除**这些行+列（不可只删 K）。
3. 重新编号（压缩空档）。
4. 用删后 K/M 求解——频率与 Abaqus 参考解一致（误差 5.7e-6，3×3×3 验证）。
5. [重要] 若模型还有 MPC，需按文档先消除 dependent DOF（`MPC=YES` 默认）；本工程模型无 MPC。

## Q6. nutrix/bef 移除与列压缩路径统一

**结论**：
- [C] 本仓库 `src/` 的 AMLS **就是** `third_party/amlsPKU/` 的**列压缩路径**移植（头注释逐一注明），已接 CMake、可编译、可运行（`build/amls_solve.exe`），**不再依赖 nutrix/bef/LobIPM**。
- [C] 功能映射：nutrix CscMatrix/MatXd → `Eigen::SparseMatrix`/`DefaultTraits::DenseMatrix`；nutrix BLAS → `linalg/blas.hpp`（宿主 OpenBLAS）；LobIPM/usegcge → `LapackDenseEigensolver`（稠密 dsygvx，局部特征全求 + 子空间稠密）。用户要求的"局部转稠密、组装后稀疏可选"路线已实现为稠密路径。
- [C] `third_party/amlsPKU/` 原版仍依赖 `nutrix/*.h`（`csc_matrix.h`/`dense_matrix.h`/`blas/*`）与 `bef/*.h`——**这些头文件不在仓库**（grep 无结果），原版独立编译需外部库。
- **待办**：确认是否删除 `third_party/amlsPKU/`（保留作参考 vs 彻底移除）；以及"函数实现层级统一"需逐一核对 `src/` 各模块与原版的对应（`node_task_executor` ↔ `amls_task_executor`、`gauss_module` ↔ `gauss_elimination` 列压缩路径等）。

---

## 附：本次新增/修改

- `src/amls/solver.cpp`：新增 `AMLS_DUMP_SUBSPACE` / `AMLS_DUMP_SUBSPACE_FILE` 诊断（B̃ 对角、λmin/λmax、K LLT、导出 K/B 文本）。
- `src/linalg/dense_eigensolver.cpp`：`solveInterval` 修正 `il=1,iu=n`（值区间合法语义）。
- `bench_abaqus/sub.K.txt` / `sub.B.txt`：20³ 缩聚子空间 K̃/B̃（87×87）dump，供复现。
- 复现命令：`scipy.linalg.eigh(K,B)` → "leading minor of order 5 of B is not positive definite"。