# AMLS 算法 C++ 实现与验证 —— 实施计划

> 记录日期：2026-10-04
> 工作目录：`Z:\codeSpace\AMLS`
> 状态：**Phase 0 已完成**（2026-10-04 第二轮确认）——环境齐备、范围已定、论文已下载，进入 Phase 1 编码
> 配套文档：[设计文档](设计文档.md) · [验证方案](验证方案.md) · [环境与构建说明](环境与构建说明.md)

## 0. 第二轮决策记录（2026-10-04 用户确认）

| 决策项 | 结论 |
|---|---|
| 底层求解器 | 采用 LAPACK 路线（OpenBLAS 提供） |
| METIS | 在工程内编译（`third_party/metis` + `third_party/GKlib`，已完成并冒烟验证） |
| 代码风格 | 模块化、组件可插拔、必要处使用模板（C++17） |
| 求解范围 | K、M 实对称，M 正定；只求特征对 |
| 精度标准 | 特征值相对误差 ≤ 4%（与用户原表述"频率误差 ≤ 2%"严格等价，系统内统一用前者），参考解来源任意（按规模分级，见验证方案） |
| 术语边界 | 系统设计完全代数化：仅使用叶节点/分割节点/根节点的树模型概念，无工程语义（设计文档 v0.2 §0） |
| 测试矩阵 | SuiteSparse .mtx，脚本化下载 |
| 构建规范 | C++17 + CMake + Ninja + Catch2 |
| 论文 | 用户已下载 6 篇至 `papers/`（详见第 3 节末尾清单） |

---

## 1. 目标定义

用 C++ 实现 AMLS（Automated Multi-Level Substructuring，自动多层子结构）算法，求解大规模稀疏对称广义特征值问题：

```
K x = λ M x
```

其中 K 实对称（半）正定、M 实对称正定，二者均为大规模稀疏矩阵（**纯代数假设，不预设任何物理来源**）。系统内部只使用树模型概念：叶节点、分割节点、根节点（术语映射见《设计文档》§0）。AMLS 是一种"一次性"（one-shot）投影方法：通过图划分把矩阵递归分解为多层的树，对每个节点求解小规模局部特征值问题，按截断阈值截断谱分量，构造降维子空间并投影，最终求解一个小得多的投影特征值问题，一次获得数百至数千阶低端特征对。

验证目标：在标准测试算例上验证正确性（与精确解/参考解对比相对误差），并与经典方法（移位反演 Lanczos）对比计算效率。

---

## 2. 环境检查结果（已核实）

MSYS2 安装于 `D:\msys2`，含 mingw32 / mingw64 / ucrt64 / clang64 / clangarm64 五个 shell 环境，**主要使用 mingw64**。

### 2.1 已具备（mingw64）

| 工具 | 版本 | 状态 |
|---|---|---|
| gcc / g++ | 16.1.0（MSYS2 Rev5） | ✅ 可用 |
| binutils (ar, ld) | 2.46 | ✅ 可用 |
| cmake | 4.3.3 | ✅ 可用 |
| ninja | 1.13.2 | ✅ 可用 |
| gdb | 17.2 | ✅ 可用 |
| pkgconf | 2.5.1 | ✅ 可用 |

结论：**C++ 编译与构建工具链完整可用，可以编译现代 C++（C++17/20/23）。**

### 2.2 缺失项（不成熟前提，见第 6 节）

| 缺失项 | 影响 | 建议解决方式 |
|---|---|---|
| gfortran（`mingw-w64-x86_64-gcc-fortran`） | 若编译原版 ARPACK、MUMPS、参考 LAPACK 需要 Fortran | `pacman -S mingw-w64-x86_64-gcc-fortran`（视选型决定是否必要） |
| BLAS / LAPACK（OpenBLAS） | AMLS 底层稠密求解与最终投影问题求解的核心依赖 | `pacman -S mingw-w64-x86_64-openblas`（自带 LAPACK + CBLAS，无需 Fortran） |
| Eigen | 稀疏矩阵存储与基础线性代数 | `pacman -S mingw-w64-x86_64-eigen3` 或 vendored（header-only） |
| METIS | **AMLS 的关键依赖**：多层图划分/嵌套剖分生成子结构树 | `pacman -S mingw-w64-x86_64-metis` |
| 稀疏直接解（SuiteSparse/CHOLMOD 等） | 移位反演、块 LDLᵀ 块高斯消元需要 | `pacman -S mingw-w64-x86_64-suitesparse`，或先用 Eigen 自带 SimplicialLDLT |
| 迭代特征求解器（ARPACK/Spectra） | 参考解生成 + AMLS 内部大规模子问题候选求解器 | Spectra 为 header-only C++（推荐）；原版 ARPACK 需 gfortran |

---

## 3. AMLS 论文清单（待下载）

按优先级分三层。标注 ✅ 的条目检索到了可公开获取的 PDF 来源；其余需通过学校数据库（知网 / Web of Science / SIAM / Elsevier / ACM）下载。文献细节（卷期页码）在下载时以出版方页面为准。

### 3.1 第一层：必读核心（算法本身与权威实现）

| # | 文献 | 类型 | 公开 PDF |
|---|---|---|---|
| 1 | Bennighof, J.K., Lehoucq, R.B., "An automated multilevel substructuring method for eigenspace computation in linear elastodynamics", *SIAM J. Sci. Comput.* 25(6): 2084–2106, 2004 | 期刊，AMLS 奠基论文（变分 + 代数两套表述） | 需 SIAM |
| 2 | Gao, W., Li, X.S., Yang, C., Bai, Z., "An implementation and evaluation of the AMLS method for sparse eigenvalue problems", *ACM TOMS* 34(4): 20, 2008 | 期刊，**实现细节最完整**（含算法伪代码、内存/时间分析、与 SIL 对比） | ✅ `https://portal.nersc.gov/project/sparse/xiaoye-web/amls_toms.pdf` |
| 3 | Kaplan, M.F., "Implementation of automated multilevel substructuring for frequency response analysis of structures", PhD thesis, UT Austin, 2001 | 学位论文，工程实现细节 | 需 UT Austin 仓库 |
| 4 | Kim, M., "An efficient eigensolution method and its implementation for large structural systems", PhD thesis, UT Austin, 2004 | 学位论文，实现与大规模算例 | 需 UT Austin 仓库 |
| 5 | Bennighof, J.K., Kaplan, M.F., Muller, M.B., Kim, M., "Meeting the NVH computational challenge: Automated multi-level substructuring", *Proc. 18th IMAC*, San Antonio, 2000 | 会议，工业应用背景 | 需检索 |

### 3.2 第二层：数学分析与理论理解

| # | 文献 | 类型 | 公开 PDF |
|---|---|---|---|
| 6 | Elssel, K., Voss, H., "An a priori error bound for automated multilevel substructuring", *SIAM J. Matrix Anal. Appl.*, 2006 | 期刊，AMLS 先验误差界 | 需 SIAM |
| 7 | Bekas, C., Saad, Y., "Computation of smallest eigenvalues using spectral Schur complements", *SIAM J. Sci. Comput.*, 2005 | 期刊，把 AMLS 解释为有理特征问题的一阶近似，给出二阶修正 | 需 SIAM |
| 8 | Yang, C., Gao, W., Bai, Z., Li, X.S., Lee, L.Q., Husbands, P., Ng, E., "An algebraic substructuring method for large-scale eigenvalue calculation", *SIAM J. Sci. Comput.* 27: 873–892, 2005 | 期刊，代数子结构（AS）版本 | 需 SIAM |
| 9 | Yi, J., Voss, H., Chen, P., "Improving eigenpairs of automated multilevel substructuring with subspace iterations" | 预印本/期刊，AMLS + 子空间迭代后处理提升精度 | ✅ `https://www.mat.tuhh.de/forschung/rep/rep158.pdf` |
| 10 | Hurty, W.C., "Dynamic analysis of structural systems using component modes", *AIAA Journal* 3(4), 1965 | CMS 源头 | 需 AIAA |
| 11 | Craig, R.R., Bampton, M.C.C., "Coupling of substructures for dynamic analysis", *AIAA Journal* 6(7): 1313–1319, 1968 | CMS（Craig–Bampton 固定界面法）源头 | 需 AIAA |
| 12 | de Klerk, D., Rixen, D.J., Voormeeren, S.N., "General framework for dynamic substructuring: history, review and classification of techniques", *AIAA Journal* 46(8): 1169–1181, 2008 | 子结构方法综述与分类 | 需 AIAA |
| 13 | Bourquin, F., "Component mode synthesis and eigenvalues of second order operators: discretization and algorithm", *M2AN*, 1992（及 Bourquin & d'Hennezel 后续） | CMS 的数学分析 | 需检索 |

### 3.3 第三层：扩展、改进与并行（后续迭代参考）

| # | 文献 | 类型 | 公开 PDF |
|---|---|---|---|
| 14 | Kim, J.-G., Boo, S.-H., Lee, P.-S., "An enhanced AMLS method and its performance", *Comput. Methods Appl. Mech. Engrg.* 287: 90–111, 2015 | EAMLS：残余模态补偿 + IRS 近似 | ✅ `http://cmss.kaist.ac.kr/cmss/papers/2015 An enhanced AMLS method and its performance.pdf` |
| 15 | Boo, S.-H., Kim, J.-G., Lee, P.-S., "Error estimation for the automated multi-level substructuring method", *Int. J. Numer. Methods Eng.*, 2016 | 期刊，AMLS 误差估计 | 需 Wiley |
| 16 | Hyun, C., Boo, S.-H., Lee, P.-S., "Improving the computational efficiency of the enhanced AMLS method", *Computers & Structures* 228: 106158, 2020 | 期刊，EAMLS 计算效率改进 | 需 Elsevier |
| 17 | Hyun, C., Lee, P.-S., "A load balancing algorithm for the parallel automated multilevel substructuring method", *Computers & Structures* 257: 106649, 2021 | 期刊，并行 AMLS 负载均衡 | 需 Elsevier |
| 18 | Stammberger, M., Voss, H., "Automated multi-level sub-structuring for a fluid-solid vibration problem", *Numer. Linear Algebra Appl.*, 2011 | 期刊，流固耦合（非对称）推广 | ✅ `https://www.mat.tuhh.de/forschung/rep/rep117.pdf` |
| 19 | Elssel, K., "Automated multilevel substructuring for nonlinear eigenvalue problems", PhD thesis, TU Hamburg-Harburg, 2006 | 学位论文，非线性特征问题推广（含 AMLS 流程的清晰描述） | ✅ `https://tore.tuhh.de/bitstream/11420/61/1/rep86.pdf` |
| 20 | Gerds, P., Grasedyck, L., "Solving an elliptic PDE eigenvalue problem via automated multi-level substructuring and hierarchical matrices" | 预印本，AMLS + H-矩阵 | ✅ `https://www.igpm.rwth-aachen.de/Download/reports/gerds/H-AMLS_preprint_homepage.pdf` |
| 21 | Liao, B.-S., Bai, Z., Gao, W., "The important modes of subsystems: a moment-matching approach", *Int. J. Numer. Methods Eng.*, 2008 | 期刊，子结构模态选取策略 | 需 Wiley |
| 22 | Ko, J., Bai, Z., "High-frequency response analysis via algebraic substructuring", *Int. J. Numer. Methods Eng.*, 2008 | 期刊，频响分析推广（ASFRA） | 需 Wiley |
| 23 | Kropp, A., Heiserer, D., "Efficient broadband vibro-acoustic analysis of passenger car bodies using an FE-based component mode synthesis approach", *J. Comput. Acoustics*, 2003 | 期刊，AMLS 工业标杆算例（被广泛引用为 AMLS 优于 SIL 的证据） | 需 World Scientific |
| 24 | Kalantzis, V., 等, "Enhanced algebraic substructuring for symmetric generalized eigenproblems", *Numer. Linear Algebra Appl.* | 期刊，现代代数子结构改进 | ✅ `https://vkalantzi.github.io/papers/nlaa2.pdf` |

> 下载建议：第一优先级下载 #1、#2、#3、#6、#7、#14，它们足以覆盖"算法定义 → 实现细节 → 误差理论 → 精度改进"的完整链条。#2（Gao et al. TOMS）应作为代码实现的主蓝本。

### 3.4 实际已下载清单（2026-10-04，`papers/`，文本已提取至 `papers/_txt/`）

| 文件 | 对应上表 | 备注 |
|---|---|---|
| An Automated Multilevel Substructuring Method for Eigenspace Computation in Linear Elastodynamics.pdf | #1 Bennighof & Lehoucq 2004 | 23 页，完整 |
| An Implementation and Evaluation of the AMLS Method for Sparse Eigenvalue Problems.pdf | #2 Gao et al. 2008 TOMS | 28 页，完整，**实现主蓝本** |
| Implementation of automated multilevel substructuring for frequency response analysis of structures.pdf | #3 Kaplan 2001 | 24 页，UT Austin **PREVIEW 版**（仅前置部分） |
| An efficient eigensolution method and its implementation for large structural systems.pdf | #4 Kim 2004 | 24 页，UT Austin **PREVIEW 版**；含知识产权声明（AMLS 商业化技术，自用/学术复现无碍，商用需授权 UT Austin） |
| Improving the computational efficiency of the enhanced AMLS method.pdf | #16 Hyun et al. 2020 | 14 页，完整，EAMLS 效率改进 |
| Fine-grained parallelization of automated multilevel substructuring method for modal frequency response analysis.pdf | （用户补充，清单外） | Wang, Wang, Hu, Cui et al.（湖南大学），13 页：基于 DAG 任务模型的细粒度并行 AMLS + 模态频响分析，后续并行化迭代的直接参考 |
| A hybrid OpenMPMPI parallel automated multilevel substructuring.pdf | （用户补充，清单外） | Wang et al., *Computational Mechanics* 78: 611–628, 2026，18 页：OpenMP/MPI 混合并行 AMLS，子树→MPI 进程静态映射 + 进程内 DAG 任务 + 非阻塞通信，96 核 23× 加速比，与 SLEPc 对拍验证；Phase 6 并行化的主参考 |

待补（如需要）：#6 Elssel–Voss 误差界、#7 Bekas–Saad、#9 Yi–Voss–Chen（公开 PDF 可代下）、#14/#15 EAMLS 原始与误差估计。当前 6 篇已覆盖 v1 实现所需全部信息。

---

## 4. 底层求解算法调研（AMLS 的"求解器层"）

AMLS 本身是计算流程框架，其内部至少有三处需要调用底层特征值/线性方程求解器：

1. **叶子子结构的局部广义特征问题**（规模小，通常几十～几千）；
2. **界面/separator 的局部广义特征问题**（随层级升高规模增大，顶层可达数千）；
3. **最终投影后的广义特征问题**（稠密，规模为截断后总模态数，通常数百～数千）；
4. （辅助）**稀疏线性方程组求解**：块 LDLᵀ 块高斯消元构造约束模态（`-K_ll⁻¹ K_li`），以及移位反演类方法中的线性求解。

按"稠密/稀疏 × 小规模/大规模 × 前 N 阶/区间"六个维度整理候选算法：

### 4.1 稠密 + 小规模

| 需求 | 算法 | 实现 | 备注 |
|---|---|---|---|
| 全部特征对 | QR 算法（`DSYGV`） | LAPACK / OpenBLAS | n < 几百时首选 |
| 全部特征对（更快） | 分而治之（`DSYGVD`） | LAPACK | Gao et al. 实测投影问题 n_proj < 5000 时 DSYGVD 显著快于 SIL |
| 前 N 阶 / 区间 | 二分 + 反迭代（`DSYGVX`）或 MRRR（`DSYEVR`+Cholesky 变换） | LAPACK | 只要低端一部分时用，省时间 |

适用场景：叶子子结构（小）、最终投影问题。这是 AMLS 实现中**最确定**的一层，直接用 LAPACK。

### 4.2 稀疏 + 小规模（叶子子结构偏大时）

| 需求 | 算法 | 实现 |
|---|---|---|
| 前 N 阶（低端） | Lanczos（直接法，针对 K 或 M⁻¹K 形式） | 自实现 / Spectra |
| 前 N 阶 | 稠密化后走 4.1（子结构 < ~2000 时通常直接稠密化更省事） | LAPACK |

工程惯例：叶子足够小就稠密化走 LAPACK，避免维护两套代码。

### 4.3 稀疏 + 大规模 + 前 N 阶（低端谱）

| 算法 | 特点 | 可用实现 |
|---|---|---|
| 移位反演 Lanczos（SIL） | 结构工程事实标准（NASTRAN 等）；收敛快、精度高；需对 (K−σM) 做稀疏分解 | ARPACK（Fortran）/ Spectra（header-only C++）/ 自实现 + 稀疏直接解 |
| 块 Lanczos（Grimes–Lewis–Simon） | SIL 的块版本，适合大量重根/密集谱 | 需自实现或 SLEPc |
| 子空间迭代（SSI） | 稳健、可块化、易并行；收敛慢于 Lanczos；Kaplan 在 AMLS 中用过其特制 SSI | 自实现（简单）/ SLEPc |
| LOBPCG | 仅需矩阵向量乘 + 预条件，无需精确分解；弱预条件下可能失败 | Spectra（实验性）/ BLOPEX / SLEPc / hypre |
| Krylov–Schur | 改进重启动的 Lanczos/Arnoldi，稳健性好 | SLEPc |
| Jacobi–Davidson | 适合内部特征值/目标点附近；实现复杂、稳定性敏感 | SLEPc / PRIMME |

适用场景：顶层 separator 子问题偏大时；以及生成**参考解**与 AMLS 对比。

### 4.4 稀疏 + 大规模 + 区间（spectrum slicing）

| 算法 | 特点 | 可用实现 |
|---|---|---|
| Sylvester 惯性 + 二分 + 移位反演 | 经典方案：用 LDLᵀ 惯性数定位区间特征值计数 | 自实现 + 稀疏直接解 |
| FEAST（围道积分） | 区间内全部特征对；天然并行；需估计区间内特征值个数，需多个复移位线性求解 | FEAST 官方库（Fortran/C 接口） |
| Sakurai–Sugiura / CISS | 与 FEAST 同族的围道积分谱投影 | SLEPc（CISS） |
| 谱切片（SISLICE 等） | 移位反演子空间迭代按谱切片并行 | 研究代码 |
| 多项式滤波（Chebyshev filter） | 仅需 SpMV，适合无法分解的超大规模 | EVSL 等 |

适用场景：若后续要按频率区间提取模态（频响分析需要），这层是 AMLS 的补充而非替代——AMLS 用 cutoff frequency 做"软"区间截断，正好对应这个需求。

### 4.5 稀疏线性方程求解（支撑层）

| 类型 | 候选 | 备注 |
|---|---|---|
| 稀疏直接 LDLᵀ / Cholesky | CHOLMOD（SuiteSparse）、Eigen SimplicialLDLT、MUMPS（需 Fortran）、PARDISO（MKL，商业） | 块高斯消元构造约束模态、SIL 的移位分解都需要；对称不定（K−σM）需 LDLᵀ（Bunch–Kaufman），CHOLMOD 只处理正定 |
| 迭代法（CG + 预条件） | Eigen / 自实现 | 超大规模时的后备 |

### 4.6 选型结论（初版实现）

- **叶子子结构 + 投影问题**：LAPACK（OpenBLAS 提供），`DSYGV` / `DSYGVD` / `DSYGVX` —— 必须项；
- **图划分**：METIS（递归二分生成多层子结构树）—— 必须项；
- **块高斯消元 / 约束模态**：Eigen `SimplicialLDLT` 起步，后续可换 CHOLMOD —— 必须项；
- **参考解/对比基准**：Spectra（header-only，内置 SymGEigsShiftSolver 即移位反演）—— 必须项；
- **大规模 separator 子问题、区间求解器**：初版不做，列为后续迭代的接口预留（第 5 节架构中把求解器抽象为接口）。

---

## 5. 拟定软件架构与实施路线

### 5.1 架构分层

```
┌─────────────────────────────────────────────┐
│ 应用层：测试算例驱动、误差评估、性能对比        │
├─────────────────────────────────────────────┤
│ AMLS 核心：子结构树管理 / 递归变换 / 模态截断   │
│           / 投影组装 / 回代                    │
├──────────────┬──────────────┬───────────────┤
│ 划分接口      │ 特征求解接口   │ 稀疏直接解接口  │
│ (METIS)      │ (LAPACK/     │ (SimplicialLDLT│
│              │  Spectra/…)  │  /CHOLMOD/…)   │
├──────────────┴──────────────┴───────────────┤
│ 矩阵层：CSR/CSC 稀疏存储（Eigen::SparseMatrix） │
└─────────────────────────────────────────────┘
```

关键设计点：三个底层能力（划分、特征求解、稀疏直接解）全部接口化，便于第 4 节中的候选算法替换对比。

### 5.2 阶段路线

| 阶段 | 内容 | 产出 | 依赖 |
|---|---|---|---|
| Phase 0 | 环境补齐（装库）、论文下载、确定测试矩阵来源 | 可编译的空工程 + 文献库 | **用户下载论文**；装库命令见第 6 节 |
| Phase 1 | 基础设施：CSR 矩阵 I/O（MatrixMarket/Harwell-Boeing）、LAPACK 封装、METIS 封装、闭式谱算例生成器（链式图/网格图 Laplacian） | 基础库 + 单元测试 | Phase 0 |
| Phase 2 | 单层 CMS（Craig–Bampton 固定界面法）实现与验证 | 正确性基线（AMLS 退化为 1 层时 = CMS） | Phase 1 |
| Phase 3 | 多层 AMLS 核心：递归剖分 → 块 LDLᵀ → 同余变换 → 子结构模态截断 → 投影 → 回代 | AMLS 求解器 v1 | Phase 2，主蓝本 Gao et al. TOMS Algorithm 1 |
| Phase 4 | 求解器接口化：底层求解算法按第 4 节做替换实验 | 性能/精度对比数据 | Phase 3 |
| Phase 5 | 标准算例验证：与解析解、Spectra 移位反演参考解对比；绘制误差-截止频率、耗时-规模曲线 | 验证报告 | Phase 3/4 |
| Phase 6（可选） | 增强 AMLS（EAMLS 残余模态补偿）、误差估计、OpenMP 并行 | 扩展版本 | Phase 5 |

### 5.3 测试算例来源（候选）

1. **解析可验**：1D 均匀杆、2D 矩形膜/Laplacian（有限差分或简单 FEM 生成），特征值有闭式解——用于 Phase 2/3 正确性回归；
2. **SuiteSparse Matrix Collection**（`https://sparse.tamu.edu`）：BCSSTK/BCSSTM 系列（Harwell-Boeing 经典结构矩阵对）、ND 系列（ND3K/ND6K/ND12K/ND24K，嵌套剖分测试问题）——用于中等规模验证；
3. **对比参考解**：Spectra 移位反演（稠密小规模可用 LAPACK 全谱作金标准）。

---

## 6. 前提条件清单 —— 全部闭环（2026-10-04 第二轮）

| # | 前提条件 | 状态 | 结果 |
|---|---|---|---|
| P1 | 论文下载 | ✅ 完成 | 用户已下载 6 篇（见 3.4）；文本已提取至 `papers/_txt/` 供实现期查阅 |
| P2 | 数值库安装 | ✅ 完成 | OpenBLAS、Eigen3、SuiteSparse、Catch2 经 pacman 装入 mingw64；**METIS 按用户要求改为工程内编译**（`third_party/`，IDXTYPEWIDTH=64，冒烟通过） |
| P3 | gfortran | ✅ 完成 | 已安装 16.1.0（用户明确要装；当前选型暂不依赖，留作后续 ARPACK/MUMPS 选项） |
| P4 | 求解范围 | ✅ 已界定 | K、M 实对称、M 正定；只求特征对；FRF 与并行列入 Phase 6 可选迭代 |
| P5 | 验证基准与精度标准 | ✅ 已界定 | 频率相对误差 ≤ 2%；参考解按规模分级（LAPACK 全谱 / Spectra SIL / 闭式解），详见《验证方案》 |
| P6 | 测试矩阵获取 | ✅ 已确定 | SuiteSparse .mtx 脚本化下载（Phase 1 提供 `tools/fetch_matrices.sh`），矩阵清单见《验证方案》§2.2 |
| P7 | 构建/代码规范 | ✅ 已确定 | C++17 + CMake + Ninja + Catch2 v3；模块化可插拔、必要处模板（详见《设计文档》） |

---

## 7. 风险与备注

1. **AMLS 精度特性**：AMLS 是一次性投影方法，截断后无法复用已算信息增量加密；精度由 cutoff frequency 控制。误差理论（Elssel–Voss 先验界）和精度改进（EAMLS、子空间迭代后处理）已在文献清单覆盖，实现时建议预留 cutoff 参数与后处理挂点。
2. **Gao et al. TOMS 论文是实现主蓝本**：给出了完整算法伪代码（划分 → 块分解 → 同余变换 → 子问题求解 → 投影 → 回代 → 反排列），并明确指出最终投影问题在 n_proj < 5000 时用稠密 DSYGVD 最优。
3. **商业背景提示**：AMLS 由 UT Austin 开发、CDH AG 商业化（已集成进 OptiStruct、NASTRAN 等），存在专利（如 US10255392B2，子结构生成相关）。学术复现与自用无碍，若后续有商业用途需注意。
4. **Windows/mingw64 注意事项**：大规模算例下注意 64 位整型（ILP64 vs LP64）一致性；OpenBLAS 默认 LP64，百万自由度内稀疏索引用 `int64_t` 自行控制即可。

---

*本文档为计划记录。Phase 0 已闭环（2026-10-04），详细设计见《设计文档》，验收标准见《验证方案》，环境操作细节见《环境与构建说明》。下一步：Phase 1 基础设施编码。*
