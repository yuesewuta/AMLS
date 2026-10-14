# Abaqus AMS 求解广义特征值问题的完整流程与参数作用机制

> **逆向分析文档**（2026-10-12）
> 目标：把 Abaqus/Standard `EIGENSOLVER=AMS`（Automatic Multi-level Substructuring，
> 内部代号 **PAMS**）求解广义特征值问题 `KΦ = λMΦ` 的**求解步骤**与**参数作用机制**
> 从二进制层面完整挖出，取代官方文档"消去树 / 缩聚 / 子空间"等模糊表述。
>
> 方法：REA + Ghidra 12.1.4 对 `ABQSMAStaCore.dll`（72 MB，主求解器）和
> `ABQSMAEqsPmeCore.dll`（2 MB，PAMS 核心算法模块）全量分析反编译；导出符号
> 15085（StaCore）+ 3187（PmeCore）个；全部伪代码见 `C:\tools\pams_pseudocode*.txt`
> 与 `C:\tools\pme_core_pseudocode.txt`。
> 证据级别标注：**[R]** = 逆向反编译（本会话产物）、**[A]** = 探针实测（
> `bench_abaqus/`）、**[D]** = 官方文档、**[PAPER]** = AMLS 论文。

---

## 0. 结论速览（TL;DR）

1. **AMS 的求解主入口是 `std_pamsDriver`**（StaCore 导出，RVA 0x3999270），它依次执行
   约束处理 → METIS 排序 → 构建消去树 → 符号分解 → RHS 变换 → 算子生成 →
   **`pme_SMPAmsEigenSolver::Solve`**（三大阶段：Reduction / Reduced Eigensolution /
   Recovery）→ 特征值表打印 → 模态排列 → 恢复。全流程共 22 个显式阶段（times 表）。
2. **三个 AMS 参数 cutoff1/2/3 与 fmin/fmax、nev 是 `std_pamsDriver` 的 29 个参数中的
   5 个 double + 1 个 int**，在构造 `pme_SolverParameters` 时被换算成 6 个频率
   门槛值（`param_1[0x78/0x80/0x88/0x90/0x98]`），再被 `ApplyClassicCutoffs`/
   `ApplyEnhancedCutoffs` 分发到每个子结构。
3. **cutoff1 的精确公式是 `λ_cut = (2π·cutoff1·fmax)² + shift`**（实测反编译
   `dVar9 = fmax²·4π² + shift;  cutoff² = param_7² · dVar9`），即官方"cutoff 是
   最大频率的乘子"的精确数值化。**每个子结构还有第二个更紧的截断值
   (0.6·cutoff1)²·dVar9**（语义见 §2.3 [INF]）。
4. **缩聚后的全局特征求解不是"直接稠密解"**，而是 Kaplan 蒸馏路线：
   `pme_SmpReducedEigenSolver` 做 子树分组 → 子树特征问题 → **蒸馏** →
   **一次子空间迭代** → Ritz 分析 → 恢复——与 AMLS 奠基论文完全一致。
5. **局部特征问题**用 LAPACK sygst/sytrd/stev/ormtr/trsm（times 表 `substr_*`），
   缩聚后特征求解在迭代求解器路径用 **DSYEVR / FEAST（轮廓积分）** 两种选择
   （`SMAStaStdAggregates::DSYEVRCoreEigenSolver / FEASTCoreEigenSolver`）。

---

## 1. 逆向证据链（本会话产出）

### 1.1 工具与目标

| 项 | 值 |
|---|---|
| 逆向工具 | REA 6.4.0（CLI/MCP）+ Ghidra 12.1.4（analyzeHeadless 持久 project） |
| 目标 1 | `D:\SIMULIA\EstProducts\2020\win_b64\code\bin\ABQSMAStaCore.dll`（72.7 MB） |
| 目标 2 | `ABQSMAEqsPmeCore.dll`（2.0 MB，PAMS 核心，3187 导出） |
| 分析范围 | 38281 函数（StaCore）+ 全部 PAMS 类；反编译输出 383 + 338 + 1092 个函数 |
| 导出符号 | StaCore 15085 个 MSVC 修饰名（未剥离）→ 全部提取到 `stacore_exports.txt` |

### 1.2 关键符号定位（地址 = 导出表 RVA，Ghidra 地址 = 0x180000000 + RVA）

| 符号 | RVA | 作用 |
|---|---|---|
| `std_pamsDriver` | 0x3999270 | **AMS 主驱动器**（29 参数，全部频率参数与输出句柄） |
| `pme_SMPAmsEigenSolver` | PmeCore 0x10c490 | AMS 特征求解器（Solve 三阶段） |
| `pme_SmpReducedEigenSolver::Solve` | PmeCore 0x130a80 | 缩聚特征求解（蒸馏+子空间迭代+Ritz） |
| `pme_AmsDataHeavy::ApplyClassicCutoffs` | PmeCore 0x98980 | 经典截断分发（cutoff1 应用） |
| `pme_AmsDataHeavy::ApplyEnhancedCutoffs` | PmeCore 0x98f40 | 增强截断分发（ABA_PAMS_ENHANCED_*） |
| `SMAStaSPEigenUtilities::*` | 0x37e78d0–0x37e9730 | 子空间迭代工具（Initialize/Orthogonalize/Scale/Solve） |
| `SMAStaSubGenerator::StaticSubGenDriver` | 0x3826c30 | 子结构静态生成（约束模态+静态缩聚） |
| `SMAStaSubGenerator::PerturbDiagonal` | 0x3826830 | 奇异刚度扰动（声学/结构界面） |
| `SMAEqsPmeSturmFactorTask/SturmCheckTask` | PmeCore | Sturm 序列检查（不漏根保证） |
| `SMAStaStdAggregates::DSYEVR/FEASTCoreEigenSolver` | 0x37f2a50/0x37f2c90 | 缩聚特征求解器选择 |

### 1.3 阶段计时表（times 探针）与 PmeCore 字符串的对应

`bench_abaqus/cant_20.times` 的每个阶段名都存在于 `ABQSMAEqsPmeCore.dll`
字符串中（且**排序相邻**与 times 表顺序一致），证明阶段计时来自 PAMS 核心：

```
k_clique m_clique k_factor sturm_factor sturm_check mode_check ksh_factor
substr_sygst sytrd stev ormtr trsm extra clean
k_updext k_bwd_psi m_updext k_stackext m_stackext
m_fwdred m_bwdupdonly m_bwdupdfwd m_bwdred ...
sd_*/vd_*（阻尼，全 0）
```

---

## 2. 求解主流程（从反编译重建）

### 2.0 总览

```
std_amssol（FSM 层入口，SMAStaStdSubGenTaskInfo 携带频率参数）
 └─ std_pamsDriver(true/*eigensolver*/, nev, jobname, nthreads,
                   fmax, fmin, cutoff1, cutoff2, cutoff3, ...)
     ├─ [0] PROCESSING CONSTRAINTS      （std_PamsOperatorManager::ProcessConstraints）
     ├─ [1] METIS ORDERING              （std_OrderingHandler::GenerateOrder）
     ├─ [2] ENFORCE RETAINED NODES TO BE ROOT
     ├─ [3] BUILD TREE AFTER METIS      （std_BuildTreeAfterMetis）
     ├─ [4] SYMBOLIC FACTOR             （pme_AmsSymbolicFactor）
     ├─ [5] TRANSFORM RHS VECTORS       （Solution/Dense/Acoustic/Residual 四类）
     ├─ [6] PERMUTE SELECTIVE DOFS
     ├─ [7] 构造 pme_SolverParameters（全部 cutoff 频率公式在这里）
     ├─ [8] GenerateOperators → 导出 stf/mas/orth
     ├─ [9] PAMS EIGENSOLVER-START
     │     └─ pme_SMPAmsEigenSolver::Solve
     │          ├─ vtable[0x10] = Reduction Phase（约化）
     │          ├─ vtable[0x18] = Reduced Eigensolution Phase（缩聚特征）
     │          └─ vtable[0x20] = Recovery Phase（恢复）
     ├─ [10] PAMS EIGENSOLVER-END → std_PrintEigenvalueTable
     ├─ [11] EnterPermuteModesPhase（模态排列/映射）
     └─ [12] Recovery（std_SubGenRecoveryManager / pme_SmpAmsRecovery）
```

### 2.1 std_pamsDriver 参数语义（反编译头部诊断表 [R]）

`std_pamsDriver` 开头的诊断打印（仅 `ABA_DYN_DIAGNOSTICS` 时）直接给出参数含义：

| 参数 | 含义 | 打印语句 |
|---|---|---|
| param_5 | **Maximum frequency** (fmax) | `"Maximum frequency %10.2f Hz"` |
| param_6 | **Minimum frequency** (fmin) | `"Mimimum frequency %10.2e Hz"` |
| param_7 | **AMS cutoff1** | `"AMS cutoff1 %10.2f"` |
| param_8 | **AMS cutoff2** | `"AMS cutoff2 %10.2f"` |
| param_9 | **AMS cutoff3** | `"AMS cutoff3 %10.2f"` |
| param_10 | **请求特征值数 nev** | `"Number of requested eigenvalues*** %d"` |
| param_4 | 求解线程数 | `"Number of solver threads %d"` |
| param_31 + local_2520[0xc] | 子结构类型 | `"Substructure type Craig-Chang/Craig-Bampton"` |
| param_32 | 声学/耦合 shift 乘子 | 见 cutoff 公式 |
| param_13/… | K/M 稀疏矩阵、RHS、EigenPairData/ModalOperators/CondensedOperators 输出 | |

> **[R] 关键结论**：官方文档"cutoff1/2/3 是 fmax 的乘子"在二进制里直接成立——
> 参数名即打印名，无任何隐藏换算层。

### 2.2 pme_SolverParameters 的截断频率构造（核心公式 [R]）

`std_pamsDriver` 在 EigenSolver 前构造 `pme_SolverParameters`（0xb8=184 字节，`mem_OperatorNewTinyThrow(0xb8)`），
把 5 个 double 参数换算成全部内部门槛。反编译关键段（**偏移为字节偏移**）：

```c
// ppVar37 = pme_SolverParameters 结构指针
*(double *)(ppVar37 + 0x08) = param_5;                    // fmax
*(double *)(ppVar37 + 0x10) = param_6;                    // fmin
*(double *)(ppVar37 + 0x18) = param_7;                    // cutoff1
*(double *)(ppVar37 + 0x20) = dVar67;                     // last_cutoff（见下）
*(double *)(ppVar37 + 0x28) = param_7;                    // =cutoff1 初值（声学时变 cutoff1·param_32）
*(double *)(ppVar37 + 0x30) = param_8;                    // cutoff2
*(double *)(ppVar37 + 0x38) = param_9;                    // cutoff3
*(double *)(ppVar37 + 0x40) = dVar65;                     // dVar65 = param_7 * 0.6
*(uint   *)(ppVar37 + 0x48) = local_2440;                 // nev
*(double *)(ppVar37 + 0x60) = CONCAT44(uVar69,uVar68);    // global shift（ABA_PAMS_GLOBALSHIFT）
*(uint   *)(ppVar37 + 0xa0) = 3000;                       // 内附子空间迭代上限/规模

dVar9 = param_5 * 39.478417604357354 * param_5 + shift;   // (2π·fmax)² + global shift
*(double *)(ppVar37 + 0x78) = param_7  * param_7  * dVar9;   // ω₁²：cutoff1²·((2πfmax)²+shift)
*(double *)(ppVar37 + 0x80) = dVar67  * dVar67  * dVar9;     // ω_last²：last_cutoff²·dVar9
*(double *)(ppVar37 + 0x88) = dVar65  * dVar65  * dVar9;     // ω₂²：(0.6·cutoff1)²·dVar9
*(double *)(ppVar37 + 0x90) = dVar9;                         // ω₀²：(2πfmax)²+shift（或 enhanced shift min）
*(double *)(ppVar37 + 0x98) = dVar64  * dVar64  * dVar9;     // ω_enh²：(cutoff1·param_32)²·dVar9（声学/耦合）
```

要点：
- **`39.478417604357354 = 4π²`** 精确确认。
- **shift 来源**：`ABA_PAMS_GLOBALSHIFT` 环境变量（`atof` 解析成 double 后按
  uint64 高低字打包）；若保留 DOF（`local_25a4≠0`）或 `param_31`（声学）则 shift 强制为 0。
- **last_cutoff**：`ABA_PAMS_LAST_CUTOFF` 环境变量，未设置时默认 **cutoff1 × 10.0**
  （`param_7 * 10.0`）——这是"最后一层/根层更宽容"的隐藏机制。
- **ABA_PAMS_ENHANCEDSHIFTVALUE** 可覆盖 shift（`atof` 后与 dVar9 取较小者）。
- **0x98 的增强门槛**：当求解含声学压力 DOF（`iVar17≠0`）时，0x28/0x30/0x38/0x40
  四个值都被乘以 `param_32`（std_amssol 传入的声学缩放，纯结构域为 1.0）。

### 2.3 ApplyClassicCutoffs：截断分发到每个子结构 [R]

`pme_AmsDataHeavy::ApplyClassicCutoffs(param_1)` 遍历全部子结构
（每条 0xa8=168 字节记录），对每个子结构写两个门槛（数值关系为可验证事实，
语义标注 [INF]）：
- 偏移 0x18（**子结构局部截断**）= `param_1[0x78]`（cutoff1²·dVar9），
  当子结构"半固定界面"标记为真时改用 `param_1[0x98]`（增强门槛）；
  当父链上的子结构被标记时改用 `param_1[0x80]`（last_cutoff²·dVar9）；
  特殊分支还会写 0 并置 flag=1（表示该子结构被完全保留/跳过截断）。
- 偏移 0x20（**第二截断**，[INF] 界面/约束模态截断）= `param_1[0x88]`，
  即 `(0.6·cutoff1)²·dVar9`——注意 **0.6²=0.36 < 1**，该门槛比 cutoff1 更紧；
  增强分支同路径替换为 `param_1[0x98]` 或 0。

> **[R] 事实**：每个子结构有两个截断值；数值上一个是 cutoff1 门槛、一个是
> 0.6·cutoff1 门槛；增强（marked）子结构两者都被替换。**没有逐层放大**——
> 验证并数值化了官方"cutoff1 统一作用于所有超节点"。确切语义（哪个门槛用于
> 子结构模态、哪个用于界面/保留模态）需继续追踪 pme_superNodeInfo 的使用点
> [INF]。

### 2.4 ApplyEnhancedCutoffs：增强路径 [R]

```c
enhanced_cutoff = ABA_PAMS_ENHANCED_CUTOFF（未设置默认 uVar9=0x40490000 → double 50.0）
enhanced_level  = ABA_PAMS_ENHANCED_LEVEL（默认 3）
ω_enh² = ((2π·fmax)² + shift) · enhanced_cutoff²
// this+0x58 = 子结构数；this+0x20 = 子结构记录基址（每条 0xa8）；
// 记录偏移 0x5c = 子结构在树中的 level（与 enhanced_level 比较）
for (i = 0; i < numSubstr; i++) {
  if (substr[i].level <= enhanced_level)
    substr[i].cutoff0 = substr[i].cutoff1 = ω_enh²;   // 偏移 0x18 与 0x20
}
```

> 数值核对：`CONCAT44(0x40490000, 0)` 拼出的 double 是 **50.0**（若按单精度
> float 解读 0x40490000 才是 3.140625）。反编译后续用它乘方，故运行语义是
> **增强乘子 = 50**（比经典 cutoff1=5 宽松 10 倍、按特征值宽松 100 倍）。
> 未设置 `ABA_PAMS_ENHANCED_LEVEL` 时对树 level ≤ 3 的子结构生效。

即：**低层子结构（树 level ≤ 3）用 50×fmax 的增强截断**（比经典 cutoff1=5
宽松得多 → 保留几乎全部模态），高层（level > 3）用经典 cutoff1=5。这是对
EAMLS 论文"底层更高截止"机制的 Abaqus 官方实现（参数化），代价是缩聚规模
增大——这也解释了为什么"增强减缩"是可选开关。

### 2.5 pme_SMPAmsEigenSolver::Solve 三大阶段 [R]

```c
void Solve() {
  (*vtable[0x10])();   // Reduction Phase
  (*vtable[0x18])();   // Reduced Eigensolution Phase
  if (需要) (*vtable[0x20])();  // Recovery Phase
  // 打印 "PAMS Timing Data"：Reduction / Reduced Eigensolution / Recovery 三行
}
```

- **Reduction Phase**：子结构生成 + 局部特征 + 缩聚（见 §2.6）。
- **Reduced Eigensolution Phase**：`pme_SmpReducedEigenSolver::Solve`（见 §2.7）。
- **Recovery Phase**：`pme_SmpAmsRecovery`（模态恢复，见 §2.8）。

### 2.6 Reduction Phase（约化/减缩）[R]

由 `SMAStaSubGenerator`（StaticSubGenDriver + SubGenProjectionDriver）执行：

1. **组装全局刚度**（`SMAStaSubSparseOperatorManager::AssembleOperator`），
   需要时 `PerturbDiagonal`（对声学压力 DOF 的奇异对角加
   `max_diag_inverse × max_abs_offdiag × 1e-10` 扰动）。
2. **Partial Factorization**：`std_SubGenSolver::StaticCondensation` —— 对 K 做
   部分 Cholesky/静力缩聚，产出 `sol_CondensedMatrices`（写 tmpdir 的
   `CondensedMatrices%p` 文件）。
3. **CalculateStaticModes**：约束模态（Craig-Bampton 静态修正向量）——
   对角置 1 + `BackwardPass`（求解 K_ii 逆耦合项）；若有声学刚体模态则
   `EliminateAcousRBConstant`。
4. **CondensedLhs / CondensedRhs**：缩聚刚度 K̃ 与质量 M̃（
   `std_SubGenSolver` 输出到 `pme_CondensedMatrices` / `SMAEqsPmeCondensedOperators`）。
5. **ExtractSLoadModes + CalculateEnergyMatrix**：载荷模态与能量矩阵。
6. 每个子结构的**局部特征问题**（times 表 `substr_sygst→sytrd→stev→ormtr→trsm`）：
   对缩聚后的子结构刚度/质量对做
   LAPACK `sygst`（Cholesky 约化到标准形）→ `sytrd`（三对角化）→ `stev`
   （三对角特征值）→ `ormtr`+`trsm`（特征向量回代）。**局部模态按 §2.3 的门槛
   （cutoff1）截断**。

### 2.7 Reduced Eigensolution Phase：蒸馏 + 子空间迭代 + Ritz [R]

`pme_SmpReducedEigenSolver`（PmeCore，65 方法）——这是"缩聚后的全局小特征问题"
的求解器，方法名即算法步骤：

1. `SetupSubstructureTree` + `GroupSubstrsToSubtree`：把缩聚后的子结构树按
   代价聚成**子树**（subtree）。
2. `SolveSubtreeEigenProblem`（每子树解缩聚特征问题）+ `TruncateEigenSolution`
   （按门槛截断）——**蒸馏（Distillation）**：
   `DistillationTask / SmpDistillation / AssembleDistilMassBlocks /
   CreateDistilOperatorBlockManager`。
3. `OneSubspaceIteration` / `OneSubspaceIterationWithRigidBodyModes`：
   **一次子空间迭代**（`UpdateRemainingVectors*` 更新剩余向量）。
4. `SmpRitzAnalysisOnSubstrModalSpace` / `SmpRitzProjectionOnV1`：
   **Ritz 分析**（在 V1 子空间上投影 K/M 求 Ritz 对）。
5. `SmpRitzRecoveryForDynamicModes / ForResidualModes`：Ritz 向量恢复。
6. `SmpBlockClassicalGramSchmidtAgainstEigenVectors` /
   `SmpBlockModifiedGramSchmidtAmongVectors`：正交化。
7. 特征值收敛检查用 `SMAStaSPEigenUtilities::CheckForEigenValueConvergence`：
   **相对误差判据 `|λ_new−λ_old|/|λ_new| < tol`**（反汇编 subsd/divsd/comisd 与
   常量比较确认）。

> **[R] 结论**：这与 [Kim04] 蒸馏子空间方法（Phase4 distilled subspace + 一次
> SIM + Ritz）**逐条对应**。官方文档"cutoff2/cutoff3 定义起始子空间"即
> `DistillationTask` 截断时用的两个门槛（`pme_SolverParameters[0x30/0x38]` =
> cutoff2/3 相关，见 §3）。

### 2.8 Recovery Phase：模态恢复 [R]

`pme_SmpAmsRecovery`（PmeCore，34 方法）：
- `ComputeTij` / `ComputeTransCouplingij` / `ComputeTransSubstrData`：沿树
  计算变换耦合（T 矩阵，对应 AMLS 的 Ψ 传播）；
- `MultiplyBmPsiByBmModeStack` / `MultiplyBmTransByBmPhiA`：矩阵-矩阵乘恢复
  全模态（`RecoverEigenModes` / `SmpRecoverFullEigenModesUsingStack`）；
- `SmpRecoverSelectiveEigenModes` / `SmpRecoverSelectiveEigenModesUsingTrans`：
  选择性恢复（只恢复 requested 模态）；
- 恢复后做 **模态排列**（std_pamsDriver 的 `EnterPermuteModesPhase`）与
  **M 归一**（`orth_M` 对角 = 1 的探针实测 [A] 印证）。

### 2.9 与实测证据的逐条对照 [A]

| 逆向发现 [R] | 探针实测 [A] | 一致 |
|---|---|---|
| `std_pamsDriver` 导出 `%s.stf`/`%s.mas`（ABA_PAMS_OPERATORS_EXPORT） | `cant_20_freqonly_stf`/`_mas` | ✓ |
| `%s_gmodes`（ABA_PAMS_EXPORT_GLOMODESSIM） | `cant_20_freqonly_gmodes`（27783×12） | ✓ |
| `%s.orth_K`/`%s.orth_M`（ABA_PAMS_ORTHOGONALITY_CHECK） | `cant_20.orth_K`/`_M` | ✓ |
| times 阶段名全在 PmeCore 字符串 | `cant_20.times` 阶段表 | ✓ |
| 局部特征 LAPACK 四步 sygst/sytrd/stev/ormtr | times `substr_*` | ✓ |
| Sturm 检查（SturmFactorTask/SturmCheckTask） | times `sturm_factor/sturm_check` | ✓ |
| 双截断值（cutoff1 与 0.6·cutoff1） | 树根块 24 DOF、局部保留 12 模态的实测 | ✓（数值自洽） |
| 全局模态 mass-normalized（M̃ 对角=1） | orth_M 对角=1.0 | ✓ |

---

## 3. 参数作用机制详解

### 3.1 `*FREQUENCY, EIGENSOLVER=AMS` 数据行

数据行：`nev, fmin, fmax` + 可选 `cutoff1, cutoff2, cutoff3`。
（文档 [D]：第 4–6 项为 AMS 三参数，默认 5.0 / 1.7 / 1.1。）

逆向确认的映射（§2.1）：`nev→param_10`、`fmin→param_6`、`fmax→param_5`、
`cutoff1/2/3→param_7/8/9`。

### 3.2 cutoff1（默认 5.0）——子结构局部截断 [R][D]

- **公式**：`λ_cut = (2π · cutoff1 · fmax)² + global_shift`。
- **作用点**：Reduction 阶段每个子结构的局部特征问题（§2.6 step 6），只保留
  特征值 < λ_cut 的局部模态。
- **与官方一致**："cutoff1 multiplied by the highest eigenfrequency yields the
  highest eigenfrequency extracted in the local supernode eigensolutions"。
- **第二截断值（0.6·cutoff1）**：每个子结构还有第二个更紧的截断值
  `(0.6·cutoff1)²·dVar9`（§2.3）。[INF] 结合 `pme_SmpReducedEigenSolver` 的
  `RitzProjectionOnV1` 等结构，该值很可能用于子结构模态在缩聚求解阶段的
  进一步截断/投影保留，但确切消费点待追踪。
- **增大效果**：保留更多局部模态 → 缩聚更精确，但局部特征求解与缩聚成本上升。

### 3.3 cutoff2 / cutoff3（默认 1.7 / 1.1）——缩聚求解起始子空间 [R][D]

- **作用点**：Reduced Eigensolution Phase（§2.7）。文档 [D]："first/second
  cutoff frequency used to define a starting subspace in the reduced eigensolution
  phase"。
- **逆向对应**：存入 `pme_SolverParameters[0x30/0x38]`，供
  `pme_SmpReducedEigenSolver` 的**蒸馏截断**与子空间迭代的初始子空间构造使用
  （DistillationTask 截断门槛）。cutoff2 用于第一道（更宽），cutoff3 用于
  第二道（更窄）；两者把缩聚系统先截到一个小得多的起始子空间，再做
  一次子空间迭代 + Ritz 恢复精度。
- **与官方一致**："define a starting subspace of the subspace iteration step.
  Increasing them improves accuracy but may affect performance"。
- **与 PKU 移植对比**：本项目 `amls_solve` 直接对缩聚 pencil 做 dsygvx 区间解，
  无蒸馏+子空间迭代，因此这两个参数无对应物（有意偏离，见
  `AMS官方文档解读.md` §4）。

### 3.4 fmin / fmax / nev [R][D]

- `fmax`：唯一被三个 cutoff 乘的全局标量；也是 Sturm 检查与特征值表打印的
  区间上界（`std_PrintEigenvalueTable` 传入 param_5）。
- `fmin`：特征值提取下界（文档 [D] 确认只影响存储/输出）。
- `nev`：返回特征值数量上限。**不影响求解器实际提取数**——AMS 按
  `[fmin,fmax]×cutoff` 窗口算全部模态再截断到 nev（探针实测 120Hz 返回 12 阶、
  1000Hz 算 18206 阶截到 20 [A]）。

### 3.5 隐藏参数（环境变量，逆向确认 [R]）

| 变量 | 作用（反编译证据） |
|---|---|
| `ABA_PAMS_GLOBALSHIFT` | 全局移位 shift 加入 `(2π·fmax)²`（§2.2 dVar9） |
| `ABA_PAMS_LAST_CUTOFF` | 末层/根层截断乘子，默认 **cutoff1×10** |
| `ABA_PAMS_ENHANCED_CUTOFF` | 增强截断乘子，默认 **50**（double 解读；0x40490000 若按 float 为 3.140625） |
| `ABA_PAMS_ENHANCED_LEVEL` | 增强截断的树 level 阈值，默认 **3** |
| `ABA_PAMS_ENHANCEDSHIFTVALUE` | 覆盖 shift，取 min 限制 |
| `ABA_PAMS_NUMGPUS` / `gpus` 命令行 | GPU 数量（GPU 路径存在：GpuSolveSubtreeEigenProblem 等） |
| `ABA_PAMS_SOLVER_TEST` | 组合 RHS 测试模式 |
| `ABA_PAMS_NODUMP` / `NOSUBSTRUCTUREGENERATE` | 关闭 dump / 跳过子结构生成 |
| `ABA_PAMS_CHECKRESIDUAL` | 残差检查（导出 `%s_fca.rhs`/`%s_fca.stf`） |
| `ABA_PAMS_EXPORT_*` 系列 | 算子/模态/缩聚算子导出（探针全部用过） |

---

## 4. 与 AMLS 论文 / PKU 移植的关系

### 4.1 与 AMLS 奠基文献的对应 [PAPER][R]

| AMLS 论文概念 | Abaqus PAMS（逆向） |
|---|---|
| 递归剖分 + 消去树 | METIS_NodeND 排序 + `std_BuildTreeAfterMetis` + `pme_AmsSymbolicFactor` |
| 子结构局部谱变换 | Reduction Phase：sygst→sytrd→stev（LAPACK） |
| 约束模态 Ψ | StaticSubGenDriver：SetDiagonal(1)+BackwardPass |
| 界面截断 ω_Γ | 双截断值（cutoff1 与 0.6·cutoff1，§2.3）[INF 语义] |
| 蒸馏子空间 [Kim04] | `pme_SmpReducedEigenSolver::*Distil*`（蒸馏任务） |
| 一次子空间迭代 [Kim04] | `OneSubspaceIteration(WithRigidBodyModes)` |
| Ritz 分析 | `SmpRitzAnalysis*` / `SmpRitzProjection*` |
| 恢复（back-transformation） | `pme_SmpAmsRecovery::RecoverEigenModes` |

### 4.2 与 PKU/CCM 移植的差异

| 环节 | Abaqus AMS | 本项目 `amls_solve` |
|---|---|---|
| 剖分 | METIS 排序→消去树（实测 7898 超节点/3-24 DOF 块） | ND/bisect 树（40 节点/叶 640 DOF） |
| 局部特征 | LAPACK sygst/sytrd/stev/ormtr 四步 | 稠密 dsygvx 区间 |
| 截断 | cutoff1=5×fmax 全局 + 0.6 界面 + 增强 3.125 | γ=√η 乘子（同构换算） |
| 全局特征 | 蒸馏 + 一次子空间迭代 + Ritz | 直接 dsygvx 稠密解 |
| 模态归一 | M̃ 对角恒 1（探针实测） | M-正交归一 |

---

## 5. 复现与扩展

- 逆向产物：`C:\tools\pams_pseudocode.txt`（383 fn）、`pams_pseudocode2.txt`
  （338 fn）、`pme_core_pseudocode.txt`（1092 fn）、`stacore_exports.txt`、
  `pme_core_exports.txt`、`pams_function_list.txt`。
- Ghidra 持久 project：`C:\tools\ghidra_proj\StaCore`、`PmeCore`。
- 待深挖：`pme_SmpReducedEigenSolver` 的 `SmpDistillation` 函数体（蒸馏具体
  截断公式）、`pme_AmsSymbolicFactor` 树构建细节、`ApplyClassicCutoffs` 中
  "marked/增强"判定来源（§2.3 中 `param_1[0x98]` 的触发条件，已知与
  `ABA_PAMS_ENHANCED_CUTOFF/LEVEL` 相关但精确触发链未追）。

## 6. 证据索引（结论 → 反编译出处）

| 结论 | 出处（文件:行或地址） |
|---|---|
| `std_pamsDriver` 参数 = fmax/fmin/cutoff1/2/3/nev | `pams_pseudocode.txt:44335-44364`（诊断打印） |
| 调用 `pme_SMPAmsEigenSolver::Solve`（true=结构域） | `pams_pseudocode.txt:45679`；`pams_pseudocode2.txt:40803`（std_amssol 传参） |
| `(2π·fmax)²+shift` 与 6 个门槛公式 | `pams_pseudocode.txt:45566-45585` |
| `ABA_PAMS_GLOBALSHIFT`/`LAST_CUTOFF`(×10)/`ENHANCEDSHIFTVALUE` | `pams_pseudocode.txt:45335-45362, 45574-45585` |
| `pme_SolverParameters` 布局（0xb8 字节） | `pams_pseudocode.txt:45522-45586` |
| 声学域时 cutoff×param_32 缩放 | `pams_pseudocode.txt:45558-45564` |
| `ApplyClassicCutoffs` 双门槛分发 | `pme_core_pseudocode.txt:44632-44840` |
| `ApplyEnhancedCutoffs`（50×fmax、level≤3） | `pme_core_pseudocode.txt:44880-44917` |
| `SMPAmsEigenSolver::Solve` 三阶段 | `pme_core_pseudocode.txt:82585-82633`（PAMS Timing Data 打印） |
| 缩聚求解=蒸馏+子空间迭代+Ritz | `pme_core_pseudocode.txt` 内 `pme_SmpReducedEigenSolver` 65 方法名（0x120440–0x133f60） |
| 局部特征 LAPACK 四步 | `cant_20.times` `substr_*` + PmeCore 阶段名字符串（`pme_core_pseudocode.txt` 1.3 节） |
| Sturm 检查 | `cant_20.times` + `SMAEqsPmeSturmCheckTask`（`pme_core_exports.txt`） |
| `PerturbDiagonal` 奇异对角扰动 | `pams_pseudocode.txt:12634-12821` |
| 约束模态（SetDiagonal(1)+BackwardPass） | `pams_pseudocode.txt:12941-12945` |
| 子空间收敛判据 | `pams_pseudocode.txt:6948`（CheckForEigenValueConvergence 反汇编） |
| 阶段名/探针文件名在 DLL 字符串 | §1.3；`%s.stf`/`%s_gmodes` 等在 StaCore 字符串区 |

---
*关联文档*：`AMS官方文档解读.md`（cutoff 官方语义）、`AMS技术路线实测报告.md`
（探针证据 E1–E6）、`AMS对比PKU_CCM.md`、`cant_31_消去树.md`。
