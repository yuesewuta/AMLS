# AMLS — Automated Multilevel Substructuring 特征值求解器

基于 Bennett (2005)、Yin/Voss/Chen (2013) 等论文实现的 AMLS（自动多级子结构）方法，用于求解大规模稀疏对称广义特征值问题 **K x = λ M x** 的低阶模态。

## 方法概述

AMLS 通过图划分将自由度组织为树状层级，逐级对子块做静态凝聚（subspace condensation），将原问题约减为规模小得多的界面问题，再用稠密/Lanczos 求解器解约减特征值问题，最后可做子空间迭代/谱截断修正提高精度。

主要模块（`src/`）：

| 目录 | 内容 |
|---|---|
| `src/core/` | DAG 调度、类型、日志、trace |
| `src/amls/` | 子块消除、separator tree、condensation、shift-invert Lanczos、谱截断、子空间迭代 |
| `src/linalg/` | BLAS/LAPACK 封装、稠密特征值求解、稀疏直接解法器 |
| `src/partition/` | METIS 图划分封装 |
| `src/io/` | Matrix Market 读取、算例生成 |
| `src/reference/` | 参考解（Spectra）与误差度量 |
| `src/apps/` | 可执行入口：`amls_solve`、`amls_verify` |

## 构建依赖

- CMake ≥ 3.20，C++17
- [Eigen3](https://eigen.tuxfamily.org)（header-only）
- OpenBLAS + netlib LAPACK（MSYS2 mingw64 包；并发 LAPACK 用 netlib 静态库）
- [Catch2](https://github.com/catchorg/Catch2) v3（单元/回归测试）
- [spdlog](https://github.com/gabime/spdlog)
- 内置 `third_party/`：METIS + GKlib（预编译静态库）、[Spectra](https://github.com/yixuan/spectra)（header-only，参考解）

以 MSYS2/mingw64 环境为例：

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

更多细节见 `docs/环境与构建说明.md`。

## 测试数据（Matrix Market）

测试矩阵来自 **[SuiteSparse Matrix Collection](https://sparse.tamu.edu)**（即 UF sparse matrix collection），以 Matrix Market（`.mtx`）格式下载，**不入库**，用脚本按需获取：

```bash
bash tools/fetch_matrices.sh [目标目录]   # 默认 tests/matrices/
```

下载的矩阵组（`tools/fetch_matrices.sh`）：

| 组 | 矩阵 | 来源 |
|---|---|---|
| HB | bcsstk01/08/09/13（刚度）及对应 bcsstm0x（质量） | https://sparse.tamu.edu/HB/ |
| ND | nd3k, nd6k, nd12k, nd24k（NASA 有限元结构） | https://sparse.tamu.edu/ND/ |

其中 ND 系列是 AMLS 文献（Yin et al. 等）常用的标准基准。仓库中不含任何 `.mtx` 数据；`tests/matrices/` 仅保留一个占位说明。

## 运行测试

```bash
./build/tests/unit/test_amls          # 单元测试
./build/tests/regression/test_regression   # 回归测试（需先下载矩阵）
```

## 文档

- `docs/设计文档.md` — 总体设计与模块划分
- `docs/AMLS_实施计划.md` — 实施计划
- `docs/验证方案.md` — 验证策略与基准矩阵选择
- `docs/子块结构性质与加速机会分析.md` — 性能分析

## 参考

1. Bennighof & Lehoucq, *An Automated Multilevel Substructuring Method for Eigenspace Computation in Linear Elastodynamics*, SIAM J. Sci. Comput., 2004.
2. Yin, Voss, Chen, *An Implementation and Evaluation of the AMLS Method for Sparse Eigenvalue Problems*, 2013.
3. Yang et al., *Improving eigenpairs of AMLS with subspace iterations*.
