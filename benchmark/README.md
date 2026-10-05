# 基准场景（Benchmark）

作为 AMLS 求解器的对比基准：7 个 SuiteSparse 刚度矩阵 + 单位质量阵 **M = I**，
一次性求出**前 200 阶**参考解，并由此得出各矩阵的输入频率 `f_user`。

## 基准矩阵

| 目标规模 | 矩阵 | 阶数 n | 来源 | 备注 |
|---|---|---|---|---|
| ≈2千 | HB/bcsstk13 | 2 003 | https://sparse.tamu.edu/HB/bcsstk13 | Boeing 飞机结构刚度（实测最小特征值 μ₁≈284>0，正定） |
| ≈5千 | Cylshell/s1rmq4m1 | 5 489 | https://sparse.tamu.edu/Cylshell/s1rmq4m1 | 圆柱壳 FEM 刚度，正定 |
| ≈1万 | HB/bcsstk18 | 11 948 | https://sparse.tamu.edu/HB/bcsstk18 | 飞机结构刚度 |
| ≈5万 | TKK/smt | 25 710 | https://sparse.tamu.edu/TKK/smt | 晶体管热应力刚度 |
| ≈10万 | Botonakis/thermomech_TK | 102 158 | https://sparse.tamu.edu/Botonakis/thermomech_TK | 热固耦合刚度 |
| ≈50万 | Botonakis/thermomech_dK | 204 316 | https://sparse.tamu.edu/Botonakis/thermomech_dK | 热固耦合刚度（20 万级；SuiteSparse 无接近 50 万的独立结构 K；注意组名是 Botonakis） |
| ≈100万 | Oberwolfach/boneS10 | 914 898 | https://sparse.tamu.edu/Oberwolfach/boneS10 | 取页面上的 A 即刚度 K |

质量矩阵一律取单位阵（`amls_solve` 不传 `--M` 即自动 M = I）。

## 一次性流程（生成参考解）

```bash
bash tools/fetch_benchmark.sh          # 1. 下载 7 个 mtx 到 benchmark/matrices/
python tools/make_reference.py         # 2. 求前 200 阶参考解 + mtx→bin + f_user (只跑一次)
```

- 脚本**可断点续跑**：已存在 `benchmark/reference/<name>_ref200.npz` 的矩阵自动跳过；
  删除对应 npz 可强制重算单个矩阵（或用 `--only 名字`）。
- 运行环境：conda base（`pip install numpy scipy pyamg`；pyamg 仅 20 万阶以上矩阵的参考解需要）。

### 求解策略

- n ≤ 130 000：`scipy.sparse.linalg.splu` 精确分解 + ARPACK 移位反演（小负移位
  σ=−1e-8·‖K‖，即使矩阵半正定含刚体模态也能稳定分解），可靠求最小 200 阶；
  bcsstk13 已与稠密分解交叉验证（前 200 阶最大相对误差 5.4e-10）。
- n > 130 000（thermomech_dK、boneS10）：3D 刚度矩阵的直接 LU fill-in 过大
  （本机仅 16 GB 内存），纯 ARPACK `which='SA'` 对内特征值收敛极慢；改用
  **LOBPCG + PyAMG 代数多重网格预条件**（smoothed aggregation，多取 16 阶共 216 阶
  保证第 200 阶质量），低内存且收敛快；该路线已在 thermomech_TK 上与移位反演结果
  交叉验证。大矩阵总耗时（AMG 建层 + LOBPCG）约十几分钟到一小时，属一次性成本。

### 病态谱的处理（质量阵缩放）

若某矩阵的特征谱量纲异常（如 1e-6 或 1e13），脚本**不重新求解**，而是选择
`alpha = 10^k` 并取 **M = alpha·I**：此时物理特征值 `λ = μ/alpha`，频率
`f = sqrt(λ)/(2π)` 随之落到正常区间。`alpha` 与是否触发缩放都记录在
`f_user.csv` 与 npz 中。

### f_user 规则

第 200 阶参考频率

```
f200 = sqrt(λ_200)/(2π)   [Hz]
```

写成科学计数法后**向上取整到最高有效位**作为 AMLS 的频率输入：

```
f200 = 7821 Hz  →  7.8×10³  →  f_user = 8000 Hz
```

汇总在 `benchmark/f_user.csv`（matrix, n, nnz, mass_scale_alpha, lambda_200, f200_Hz, f_user_Hz）。

## mtx → bin 缓存格式

`benchmark/bin/<name>.bin`（小端），避免每次测试重新解析 mtx：

```
偏移  类型        内容
0     char[8]    magic "AMLSBIN1"
8     u32        version = 1
12    i64        n        (方阵阶数)
20    i64        nnz
28    i64[n+1]   indptr   (CSR 行偏移)
      i32[nnz]   indices  (列索引, 0-based)
      f64[nnz]   data     (非零值)
```

矩阵已对称化并去除显式零元。Python 侧读取示例见 `tools/make_reference.py:write_bin`。
（`amls_solve` 目前仍读 mtx；bin 供后续加 `--K-bin` 快速加载或 Python 对比脚本使用。）

## 运行基准对比

```bash
bash tools/run_benchmark.sh              # 全部 7 个矩阵,结果写 benchmark/results/
bash tools/run_benchmark.sh bcsstk13     # 单个矩阵
```

每个矩阵用 `amls_solve --K <mtx> --nev 200`（M = I）求解，输出写到
`benchmark/results/<name>_amls.txt`，与 `benchmark/reference/<name>_ref200.npz`
中的前 200 阶参考特征值对比：

```bash
python tools/compare_reference.py              # 汇总所有矩阵的逐阶误差
python tools/compare_reference.py bcsstk13     # 单个矩阵
```

对比基于标准特征值 μ（M = I，α 缩放只影响物理频率，不影响该误差表）；
频率相对误差 ≈ 特征值相对误差的一半（f ∝ √λ）。

## 目录结构

```
benchmark/
├── matrices/<名>/<名>.mtx    下载的原始矩阵（不入库；tar 包顶层即矩阵名目录）
├── bin/<名>.bin              mtx 的二进制缓存（不入库）
├── reference/<名>_ref200.npz 参考特征值 mu/lam/f、alpha、f_user（不入库）
├── results/                  AMLS 求解输出
└── f_user.csv                各矩阵 f200 与 f_user 汇总
```

## 已生成的参考解（实测）

| 矩阵 | n | μ₁ | μ₂₀₀ | f₂₀₀ (Hz) | f_user (Hz) | α |
|---|---|---|---|---|---|---|
| bcsstk13 | 2 003 | 2.84e2 | 2.90e5 | 85.71 | 90 | 1 |
| s1rmq4m1 | 5 489 | 3.80e−1 | 1.18e1 | 0.5458 | 0.6 | 1 |
| bcsstk18 | 11 948 | 1.24e−1 | 2.19e0 | 0.2354 | 0.3 | 1 |
| smt | 25 710 | 3.34e−3 | 3.92e2 | 3.152 | 4 | 1 |
| thermomech_TK | 102 158 | ≈0（刚体模态） | 6.95e−1 | 0.1327 | 0.2 | 1 |
| thermomech_dK | 204 316 | 计算中 | | | | |
| boneS10 | 914 898 | 计算中 | | | | |

说明：
- 目前 5 个矩阵的谱量纲都正常，**没有触发 α 缩放**（α=1）；缩放机制保留供 dK/boneS10 使用。
- thermomech_dK 的 mtx 是 `general` 全存储（上下三角各一份，相对偏差 1.8e-16），
  脚本按 `(A+Aᵀ)/2` 平均对称化；若误用 `A+Aᵀ−diag` 会使谱整体放大 2 倍。
- bcsstk13 参考解已与稠密 `eigvalsh` 交叉验证（前 200 阶最大相对误差 5.4e-10）。
