# AAMS 性能深挖：cant_20 对标 Abaqus AMS（2026-10-17）

目标（/goal）：AAMS 在算例 cant_20 上的总耗时与 Abaqus AMS 差距 ≤20%。
本文记录深度分析、每一步优化的问题定位 → 证据 → 改动 → 实测效果，以及
剩余差距的量化路线。所有数字来自本机实测（16 逻辑核 Windows，MinGW 构建，
OpenBLAS 0.3.33 + netlib LAPACK；机器有后台负载，取多次运行稳定值/最小值，
负载可使结果膨胀 1.3–2×，正文注明）。

## 1. 基线：Abaqus AMS 在 cant_20 上的实测

探针 `bench_abaqus/cant_20.times`（ABA_PAMS_* 导出，4 线程，默认
cutoff1=5、频段 0–120 Hz，求 12 阶）：

| 任务（每线程累计） | 秒 |
|---|---|
| k_clique | ~0.9 |
| k_factor | — |
| k_updext | 0.066 |
| substr_sygst/sytrd/stev（局部特征四步） | 合计 ~0.29 |
| 每线程任务总和 | 1.258 |

加空闲与同步，Abaqus wall ≈ **1.4–1.65 s**。20% 目标 ⇒ AAMS 总耗时 ≤
**~1.7–2.0 s**。

## 2. 优化前基线（AAMS）

首轮稳定状态（16 线程）：

| 阶段 | 秒 |
|---|---|
| METIS 排序+建树 | 0.4 |
| REDUCTION（2601 子结构消元） | 8.2 |
| DISTILL（蒸馏+子空间迭代+Ritz） | 87 → 本轮先修至此 4.2 |
| RECOVERY | 0.2 |
| **TOTAL** | **~65（起始）/ 14–15.5（DISTILL 修复后）** |

## 3. 本轮逐项优化

### 3.1 DISTILL 去 Eigen 化（87 s → 4.2 s）

根因：Eigen 的 LDLT/SelfAdjointEigenSolver/mOrthoNormalize 无 BLAS 后端。
改为 LAPACK：dpotrf/dsytrf（σ 增广回退）+ dpotrs/dsytrs + dsyevd/dsygvd，
dgemm 显式 BLAS，且该阶段放开 OpenBLAS 多线程（Reduction 池已销毁，
BlasThreadsGuard 管理）。单点即 20×。

### 3.2 REDUCTION 插桩

- `NodeTaskExecutor`：11 类 DAG 任务计时/计数（AAMS_PROFILE=1 输出）。
- `col_ops`：opsColAcc gemv/gemm 两路的调用数、耗时、flops、临时分配、
  按 flops 五桶直方图（含每桶耗时）。

首轮剖析数据（cutoff1=5，16 线程）：

```
colAcc gemv: calls=4,580,532 flops=5.458e+10 t=36.9–54 core-s
hist (calls):  <100:82k  <1k:344k  <10k:403k  <100k:545k  ≥100k:58k
任务和: CompressUpdate 39–42 核秒 | MAccum 13.6 | KAccum 7.7 | SolveLocalEigenPair 3.9
```

关键事实：**4.58M 次调用、55 GFLOP，却花 37–54 核秒**；其中 315 万次是
B/C 无有效列的空转调用；实调用平均每次 ~26–31 µs，与块大小无关——纯
固定开销。

### 3.3 OpenBLAS 微基准（排除 BLAS 本身）

单线程小块基准（/tmp/cab3.cpp）：

| m×k×n | dgemm | n 次 dgemv | 手写内核 |
|---|---|---|---|
| 32×12×50 | 0.93 µs | 2.40 µs | 3.38 µs |
| 200×50×20 | 9.2 µs | 11.6 µs | 36 µs |

结论：BLAS 本身对典型形状只要 2–4 µs/调用。真实运行的 26–31 µs 是
**并行环境膨胀**（校验扫描、堆分配、统计原子、锁、16 线程争用叠加）。

### 3.4 校验 O(1) 化（最大单项固定开销）

`colCompShapeOk`/`coverageOk` 每次调用对 B、C、A 做 O(validCols) 全列
扫描（目标块 validCols 上千 ⇒ 每调用 2–4 万次迭代）。改为：
- O(1) 快检（尺寸/计数/缓冲一致性）每次必做；
- 全列深检（列号升序、激活计数一致）只在 `AAMS_COLCOMP_VALIDATE=1` 时做；
- 覆盖性兜底下沉到散加点（`accDstCol`，storageCol<0 即报错）。

效果：gemv 64.8 → 42.7 核秒（REDUCTION 16.7 → 9.4 s）。

### 3.5 空块短路 + 缓冲复用 + 统计门控

- 空块判断（B/C activeCols==0）提到校验前——69% 调用直接返回。
- gemv 路径 t/xs/y 改 thread_local 复用；统计原子全部门控到
  AAMS_PROFILE（生产路径零开销）。

### 3.6 分块 gemm（大 B 场景）

acC≥8 且 rows·validCols≥4096 时：w=32 列一块，gatherActiveCols 拼右端，
单次 dgemm（thread_local 缓冲），再逐列散加。B 流量除以 32。
（试验过仅按 acC 分流：OpenBLAS 小 dgemm 驱动开销 5–10 µs，小块反而
亏 2.4×，故保留尺寸门槛。）

≥100k flops 桶：202 µs/调用 → 79 µs/调用（545k 中等块桶由 3.4 节解决）。

### 3.7 CompressUpdate 批量化：4.58M 次乘法 → 3.51M → 75k

结构：`μ(d,a) += μ(d,node)·Ψ(node,a)`，逐 (descendant, ancestor) 对调用
opsColAcc，调用数 = Σ|desc|×|anc| ≈ 4.58M。

新增 `opsColAccBatch`（col_ops）：固定 (node,d)，把全部祖先 Ψ 的有效列
拼接成**一次 dgemm**（L 的有效列 gather 成连续左端，右端按 L 有效存储列
行序重建），乘积按列分组后逐目标**持锁散加**。锁持有时间同步大减。

效果：调用 4.58M → 3.51M，gemv 43 → 18.9 核秒，REDUCTION 9.2 → 5.66 s。

### 3.8 KAccum/MAccum 转置批量化（3.51M → 75k 次）

结构（固定 nodeI, j，对全部 k≥j）：
- K̃(j,k) += Ψ_jᵀ·K̃(nodeI,k) —— B=Ψ_j 在 k 上共享；
- M̃(j,k) += Ψ_jᵀ·M̃(nodeI,k) + M_temp_jᵀ·Ψ(nodeI,k) —— 两个共享 B。

新增 `opsColAccBatchTrans`：B 有效行预提取，各 C 有效列拼右端，一次
转置 dgemm，逐目标持锁按 B 有效行散射（占位行乘积丢弃，与逐对 gemv
语义一致）。

效果：**colAcc 调用 3.51M → 75,463（46×），耗时 18.9 → 1.34 核秒**；
REDUCTION 5.66 → 5.0 s。精度逐位一致（λ1=30096.77893，12 阶全同）。

### 3.9 线程饱和发现（REDUCTION wall 与线程数无关）

| 线程 | REDUCTION wall | gemv 核秒（批量化前） |
|---|---|---|
| 4 | 8.99 s | 17.8 |
| 8 | 8.80 s | 24.9 |
| 16 | 9.12 s | 42.0 |

加线程只烧核秒不加墙钟——瓶颈是**树依赖链关键路径**，不是并行度。
批量化后同样成立（4/8/16 线程 wall 均 ~4.9 s）。这意味着继续压缩
REDUCTION 必须砍关键路径本身（见 §5），堆线程无效。

## 4. 当前状态（2026-10-17）

3 次连跑（16 线程，AAMS_PROFILE=1）：

| 阶段 | 优化前 | 现在 |
|---|---|---|
| METIS | 0.4 | 0.39–0.42 |
| REDUCTION | 8.2（任务和 69 核秒） | **5.0（任务和 ~27 核秒）** |
| DISTILL | 87 → 4.2 | 4.8–5.0 |
| RECOVERY | 0.2 | 0.17–0.19 |
| **TOTAL** | **~65** | **10.8–12.4（最好 10.76）** |

精度：12 阶特征值与 scipy 参考 0.06–0.2% 内，残差 first=0.546/last=0.599，
与优化前逐位一致。

距目标（≤~2 s）还差 **~5×**。

## 5. 剩余差距量化路线

REDUCTION wall 5.0 s 但任务和只 ~27 核秒（16 线程利用率 ~34%）——
关键路径约束。分解：

1. **CompressUpdate 9.2 核秒**：批量后剩余成本是拼接 gather
   （O(actB×nAll)）+ 散射 + dgemm 的内存流。可做：Ψ 列与 L 有效列
   顺序一致时免 gather 直乘；散射融合进 dgemm 输出（beta=1 直写目标，
   需目标列布局连续）。
2. **SolveLocalEigenPair 4.9 核秒**：叶/中层 dsygvx 局部特征。Abaqus
   用 sygst/sytrd/stev 四步链。对深层大块可试 dsygvd（分治）或区间
   dsyevr。
3. **关键路径串行链**：树深处大节点的 9 段 DAG 串行。Abaqus k_updext
   仅 0.066 s/线程 —— 其 Reduction 是全稠密块单次 dgemm 链。结构上
   可探索：层间流水（上层节点 Gauß 与下层 Compress 重叠）、大节点
   CompressUpdate 的 dgemm 多线程化（当前 opsTlsThreads 预算制）。
4. **DISTILL ritz 3.2 s**：dsyevd(3000) 的三对角化 O(n³) 占主导。
   候选：p′ 降维（3000→2000 需验证精度）、dsyevr 只取 12 阶（省分治
   不省三对角化）、或把正交归一并入 dsygvd 广义型（免 ortho_y 1.1 s）。
5. **测量的注意事项**：机器后台负载（ZCode/ToDesk/ChatGPT 等）可致
   1.3–2× 膨胀；对拍时取多次最小值。`AAMS_COLCOMP_VALIDATE=1` 可恢复
   全量校验；`AAMS_NO_FUSED=1` 强制逐列路径（A/B 用）。

## 6. 复现命令

```bash
export PATH="/d/msys2/mingw64/bin:/d/msys2/usr/bin:$PATH"
cd aams && cmake --build build -j 8
AAMS_PROFILE=1 ./build/aams_solve.exe   --K ../bench_abaqus/cant_20_K.mtx --M ../bench_abaqus/cant_20_M.mtx   --fmin 0 --fmax 120 --cutoff1 5 --reduced distill --threads 16
```
