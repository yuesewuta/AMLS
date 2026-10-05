# 基准场景（Benchmark）

作为 AMLS 求解器的对比基准：7 个 SuiteSparse 刚度矩阵 + 单位质量阵 **M = I**，
一次性求出**前 200 阶**参考解，并由此得出各矩阵的输入频率 `f_user`。

## 基准矩阵

| 目标规模 | 矩阵 | 阶数 n | 来源 | 备注 |
|---|---|---|---|---|
| ≈2千 | HB/bcsstk13 | 2 003 | https://sparse.tamu.edu/HB/bcsstk13 | Boeing 飞机结构刚度，**半正定（含刚体模态）** |
| ≈5千 | Cylshell/s1rmq4m1 | 5 489 | https://sparse.tamu.edu/Cylshell/s1rmq4m1 | 圆柱壳 FEM 刚度，正定 |
| ≈1万 | HB/bcsstk18 | 11 948 | https://sparse.tamu.edu/HB/bcsstk18 | 飞机结构刚度 |
| ≈5万 | TKK/smt | 25 710 | https://sparse.tamu.edu/TKK/smt | 晶体管热应力刚度 |
| ≈10万 | Botonakis/thermomech_TK | 102 158 | https://sparse.tamu.edu/Botonakis/thermomech_TK | 热固耦合刚度 |
| ≈50万 | Botanakis/thermomech_dK | 204 316 | https://sparse.tamu.edu/Botanakis/thermomech_dK | 热固耦合刚度（20 万级；SuiteSparse 无接近 50 万的独立结构 K） |
| ≈100万 | Oberwolfach/boneS10 | 914 898 | https://sparse.tamu.edu/Oberwolfach/boneS10 | 取页面上的 A 即刚度 K |

质量矩阵一律取单位阵（`amls_solve` 不传 `--M` 即自动 M = I）。

## 一次性流程（生成参考解）

```bash
bash tools/fetch_benchmark.sh          # 1. 下载 7 个 mtx 到 benchmark/matrices/
python tools/make_reference.py         # 2. 求前 200 阶参考解 + mtx→bin + f_user (只跑一次)
```

- 脚本**可断点续跑**：已存在 `benchmark/reference/<name>_ref200.npz` 的矩阵自动跳过；
  删除对应 npz 可强制重算单个矩阵（或用 `--only 名字`）。
- 运行环境：conda base（需 numpy + scipy，`pip install numpy scipy`）。

### 求解策略

- n ≤ 130 000：`scipy.sparse.linalg.splu` 分解 + ARPACK 移位反演（sigma=0），可靠求最小 200 阶；
  bcsstk13 的刚体模态（μ=0）也会被正确求出。
- n > 130 000（thermomech_dK、boneS10）：移位反演分解代价过高，直接 ARPACK `which='SA'`
  求**代数上**最小的 200 阶（含刚体模态时最低几阶为 0，结果一致）。
  大矩阵这一步可能需要数小时，属一次性成本。

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
├── matrices/<组>/<名>.mtx    下载的原始矩阵（不入库）
├── bin/<名>.bin              mtx 的二进制缓存（不入库）
├── reference/<名>_ref200.npz 参考特征值 mu/lam/f、alpha、f_user（不入库）
├── results/                  AMLS 求解输出
└── f_user.csv                各矩阵 f200 与 f_user 汇总
```
