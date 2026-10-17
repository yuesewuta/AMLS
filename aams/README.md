# AAMS —— Abaqus AMS（PAMS）逆向复原的 C++ 实现

对 Abaqus/Standard `EIGENSOLVER=AMS`（内部代号 **PAMS**）的完整逆向复原实现，
求解广义特征值问题 `KΦ = λMΦ`。算法语义、参数公式、阶段结构全部来自
REA + Ghidra 反编译证据（[R]），与探针实测（[A]）、官方文档（[D]）三方印证：

> 依据文档：`../docs/abaqus_ams/AMS求解流程与参数机制逆向分析.md`
> （伪代码：`C:\tools\pams_pseudocode*.txt`、`C:\tools\pme_core_pseudocode.txt`）

## 构建

```bash
cd aams
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# 运行（MSYS2 DLL 在 PATH 上）
./build/aams_solve.exe --K K.mtx --M M.mtx --fmax 120 --nev 20
```

依赖：MSYS2 Eigen3 / OpenBLAS / netlib LAPACK（静态 + BLAS3 自适应 wrap）/
spdlog；工程内预编译 METIS/GKlib（`../third_party`）。

## 与逆向符号的映射（本仓库代码 ↔ PAMS 二进制）

| Abaqus PAMS 符号 [R] | AAMS 代码 | 证据 |
|---|---|---|
| `std_pamsDriver`（主驱动，29 参数） | `src/ams/pams_driver.cpp` `AmsSolver::solve` | pams_pseudocode.txt:44335-44364 |
| `pme_SolverParameters`（184 字节，6 个频率门槛） | `src/ams/solver_parameters.hpp` `PamsParameters` | pams_pseudocode.txt:45522-45586 |
| `(2π·fmax)² + shift`（dVar9） | `PamsParameters::dVar9()` | 45566-45585 |
| `ω1² = cutoff1²·dVar9`（主截断） | `omega1Sq()` | ApplyClassicCutoffs 44632+ |
| `ωlast² = lastCutoff²·dVar9`（根层，默认 cutoff1×10） | `omegaLastSq()` | ABA_PAMS_LAST_CUTOFF [R] |
| `ωif² = (0.6·cutoff1)²·dVar9`（第二截断） | `omegaInterfaceSq()` | ApplyClassicCutoffs 偏移 0x20 [INF] |
| `ωenh² = 50²·dVar9`，叶起 level ≤ 3 生效 | `omegaEnhSq()` + 驱动逐节点分发 | ApplyEnhancedCutoffs 44880-44917 |
| 子结构 level 从叶计数（叶=0） | 驱动内 `levelFromLeaf = maxDepth − rootDepth` | 45683-45713（`max(子level)+1`，无子=0） |
| `METIS ORDERING` → `BUILD TREE AFTER METIS` → `SYMBOLIC FACTOR` | `src/ams/tree_builder.cpp` `buildAbaqusTree`（NodeND 于网格节点图 + etree + exact 超节点 + Collapse2/AggregateLeaves） | stf 块分布标定（cant_10/cant_20） |
| Reduction Phase（vtable[0x10]）：局部特征 `sygst→sytrd→stev→ormtr→trsm` | `NodeTaskExecutor` 局部 `dsygvx` 区间求解（稠密路径，区间空回退保 1 阶） | cant_20.times `substr_*` [A] |
| 约束模态 Ψ / 静力缩聚 | `gauss_module` / `temp_block_manager`（列压缩 Ψ/Schur/μ 传播） | StaticSubGenDriver SetDiagonal(1)+BackwardPass |
| ApplyClassicCutoffs / ApplyEnhancedCutoffs 逐子结构分发 | `lambdaHiPerNode`（逐节点 λ_hi 向量） | 44632-44840 / 44880-44917 |
| Reduced Eigensolution（vtable[0x18]）— DSYEVR/FEAST 直接路径 | `--reduced direct`（默认）：`LapackDenseEigensolver::solveInterval` 频段 [fmin,fmax] + nev 截断 | DSYEVRCoreEigenSolver 0x37f2a50 |
| Reduced Eigensolution — 蒸馏 + 一次子空间迭代 + Ritz | `--reduced distill`：`distill_solver`（cutoff2/cutoff3 定义起始子空间） | pme_SmpReducedEigenSolver 65 方法 |
| Recovery Phase（vtable[0x20]）：沿树 Ψ 传播恢复 + M 归一 | 分层并行回代 `X_i = Φ_i·z_i + Σ_a Ψ_{i,a}·X_a` + 反排列 | pme_SmpAmsRecovery ComputeTij/RecoverEigenModes |
| Sturm 序列检查（不漏根） | 区间求解的 LAPACK 计数语义（复数特征值计数） | SturmFactorTask/SturmCheckTask |
| 特征值表打印 | `aams_solve` `M O D E E X T R A C T I O N` 表（λ 与 f=√λ/2π） | std_PrintEigenvalueTable |
| PAMS Timing Data 三阶段 | `METIS_ORDERING_BUILD_TREE` / `REDUCTION` / `REDUCED_EIGENSOLUTION` / `RECOVERY` | pme_core_pseudocode.txt:82585-82633 |

## 参数语义（与 Abaqus `*FREQUENCY, EIGENSOLVER=AMS` 一致）

数据行 `nev, fmin, fmax, cutoff1, cutoff2, cutoff3`（默认 5.0 / 1.7 / 1.1）：

- **cutoff1**：子结构局部截断乘子，`λ_cut = (2π·cutoff1·fmax)² + shift` [R]。
- **cutoff2/cutoff3**：蒸馏路径的起始子空间定义（direct 路径不消费）。
- **nev**：不改变提取数量——按 `[fmin,fmax]` 窗口提取全部模态后输出截断到
  nev [R][A]（120Hz 实测 Abaqus 返回 12 阶；1000Hz 算 18206 阶截到 20）。
- **fmin**：提取下界；**fmax**：唯一被 cutoff 乘的全局标量、Sturm 上界。

隐藏参数（环境变量，与 Abaqus 同名同义）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `ABA_PAMS_GLOBALSHIFT` | 0 | 全局 shift 加入 (2π·fmax)² |
| `ABA_PAMS_LAST_CUTOFF` | cutoff1×10 | 根层截断乘子 |
| `ABA_PAMS_ENHANCED_CUTOFF` | （未启用） | 设置即启用增强截断（默认 50） |
| `ABA_PAMS_ENHANCED_LEVEL` | 3 | 叶起 level ≤ 该值生效 |

CLI 覆盖：`--shift/--last-cutoff/--enhanced/--enhanced-cutoff/--enhanced-level`
（显式给参时不再读环境变量）。

## cant_20 验证状态（2026-10-16）

Abaqus AMS 参考（`bench_abaqus/cant_20.dat`，120Hz）：12 阶，
λ1=30066（27.597Hz）。导出矩阵真值（scipy shift-invert）：λ1=30045，
≤120Hz 共 13 阶（第 13 阶 119.82Hz 在 Abaqus 表中未出现）。

| 配置 | λ1 偏差 | 高阶最大偏差 | 阶数 |
|---|---|---|---|
| AAMS（rea 树，cutoff1=5，direct） | 0.10%（vs .dat） | ~5%（λ12） | 12 ✓ |
| AAMS（rea 树，cutoff1=8，direct） | （扫描中） | | |
| amls_solve（nd 树，γ=5） | 0.03% | ~0.6% | 12 ✓ |

已知差距（rea 树高阶 4–5%）：Abaqus 在同 cutoff1 下更准，来自其
`k_clique/k_updext` 跨级直连传播与第二截断（0.6·cutoff1）的界面模态保留
——本实现的列压缩缩聚未复刻该机制（对应逆向文档 §5"待深挖"项）。
残差 ~0.3–0.6 为截断型子结构的固有属性（恢复向量缺高频成分，λ 精度
才是验收指标，见 `docs/` 验收协议 eigenvalueRelErr ≤ 4%）。

## 目录

```
aams/
├── apps/aams_solve.cpp        CLI（Abaqus AMS 参数语义）
├── src/
│   ├── ams/                   PAMS 核心（逆向复原层）
│   │   ├── pams_driver.*      std_pamsDriver + 三阶段 Solve
│   │   ├── solver_parameters.hpp  pme_SolverParameters 精确公式
│   │   ├── tree_builder.*     METIS→etree→超节点→Abaqus 聚合
│   │   ├── aggregation.*      Collapse2 / AggregateLeaves
│   │   ├── gauss_module.*     Ψ / Schur / 缩聚（列压缩）
│   │   ├── temp_block_manager.*   μ 创建/传播/右乘
│   │   ├── node_task_executor.*   节点 DAG 调度 + 局部特征
│   │   ├── distill_solver.*   蒸馏 + 一次子空间迭代 + Ritz
│   │   └── separator_tree.*   消去树容器
│   ├── linalg/                BLAS/LAPACK 封装、列压缩矩阵、稠密特征
│   ├── partition/             符号分析、超节点构建
│   ├── io/                    MatrixMarket 读写
│   ├── core/                  类型、日志、trace
│   └── reference/             误差指标
└── CMakeLists.txt             独立构建
```
