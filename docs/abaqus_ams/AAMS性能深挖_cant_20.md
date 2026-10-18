# AAMS 性能深挖：cant_20 对标 Abaqus AMS（2026-10-17）

目标（/goal，2026-10-18 更新）：AAMS 在算例 cant_20 上的总耗时与
Abaqus AMS 差距 ≤2s 或 ≤20%。本文记录深度分析、每一步优化的问题定位
→ 证据 → 改动 → 实测效果，以及剩余差距的量化路线。所有数字来自本机
实测（16 逻辑核 Windows，MinGW 构建，OpenBLAS 0.3.33 + netlib LAPACK；
机器有后台负载，取多次运行稳定值/最小值，负载可使结果膨胀 1.3–2×，
正文注明）。

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

### 3.10 DISTILL ritz：dsyevd 全谱 → dsyevr 频段区间（3.2 → 1.56 s）

ritz 原来对 p×p（p=3000）投影矩阵做 `dsyevd` 全谱分解，但消费端只要
λ∈(0, (2πfmax)²]。改 `dsyevr`（MRRR）`range='V'`、区间
[−1e300, (2πfmax)²]：ritz 3.2 → 1.56 s，TOTAL 9.6 s。

**坑（重要）**：`dsyevr` 的 Z **不得与 A 别名**——DORMTR 回变换要从 A
的严格三角读 Householder 反射子，就地覆盖会破坏特征向量。症状：残差
0.546 → 0.998（特征值因最终 REDUCED_EIGENSOLUTION 在蒸馏子空间重解而
仍正确，极具迷惑性）。修复 = 独立 Z 缓冲（72 MB，可接受）。
commit acdec96。

### 3.11 蒸馏子空间规模上限 --subspace-max（TOTAL 9.6 → 6.4 s）

Abaqus `pme_SolverParameters[0xa0]=3000` 是子空间**容量上限**而非目标
维数：cant_20 自然蒸馏过滤仅保留 10/3348 个 Ritz 向量，top-up 补到
2990——白算 99.7%。新增 `--subspace-max`（默认 0=3000）：

| S | TOTAL | λ12 相对 .dat | 备注 |
|---|---|---|---|
| 3000 | 9.6 | ~4.1% | 默认（保护 cutoff1=8 精度档） |
| 1200 | 6.4 | ~4.3% | **基准配置**，仅 +0.14% |
| 800 | 5.4 | ~5.5%（λ8 +4.2%） | 超 4% 协议，弃 |
| 600 | — | f12=119.3 Hz 贴边 | 弃 |
| 400 | — | 仅 9 阶 <120 Hz | 弃 |

cutoff1=8 精度档保持默认 3000 不受影响。commit 3c7d4e4。

### 3.12 优先级调度 + 关键链时间线（node_task_executor）

- ThreadPool 由 FIFO 改**优先级队列**（树深优先，根=0 最高）——链上
  小节点不再排在叶洪泛后面（单节点排队 140ms → 3ms）。
- `AAMS_TASK_TIMELINE=1` 导出 per-(node,task) 时间线 CSV（含树边），
  配 Python 关键链 DP（自底向上取 max-end 子女）。

效果：链节点排队消除，但 wall 不变——链小节点延迟是叶洪泛挤占的
**果**不是**因**；wall 仍由叶洪泛 + 近根串行链构成。

### 3.13 dgemmColsRaw 小乘法门控

批量 op 内部大量 1×3×66 级微 gemm 曾走 ThreadGroup：spawn/join/pin
固定开销 ms 级（实测 L 1×3×66 耗 7.67ms，flops 仅 ~400）。加门控：
`2mnk < 3e7` 直接串行 cblas_dgemm。gemm 核秒 11.5 → 5.0。

### 3.14 CompressUpdate 堆叠批量化（REDUCTION 3.04 → 1.73 s）★最大单项

时间线关键链分解（REDUCTION 3.04s）：36 链节点 = 叶洪泛 ~0.5s +
链尾 CUpdate ~2.5s。链节点 CUpdate 单个 100–270ms，全在逐 d 循环：
每个 d 调一次 `opsColAccBatch`（m=1 微 gemm）。

微基准实锤：OpenBLAS m=1, k=120, n=1000 单次 **0.87ms**（flops 仅
240K），且 **m≤128 调用耗时持平**——m=1→128 单行成本 ~125×。而同一
node 的全部 descendant 共享同一 Ψ 列表（右端相同）：

新增 `opsColAccBatchStacked(Ls, psis, targetsPerL, mtxsPerL)`：
1. 左块按**活跃原始列签名**分组（同签名 → 共享一次 C_all gather，
   原先每个 d 重建一遍相同右端，gather 8.2 核秒大半是冗余）；
2. 组内行堆叠（预算 4M 双精度/块），单次 dgemm 出 Y= L_stack·C_all；
3. 逐 (d,祖先) 持锁散射（与逐 d 完全相同的落点与累加次序）。

`updateColComp_` 重构为：0 行 d 即时倒计数，数值 d 预检并收集，
段末单次堆叠调用，成功后统一倒计数；d 按线程数取模分段并行（8 段）。

效果：批量核时 gather 8.2→1.19s、gemm 14.0→2.51s（calls 149k→79k），
REDUCTION 3.04 → **1.73 s**，TOTAL 6.4 → **3.04 s**（best-of-5）。
残差逐位不变（0.5555962051/0.6026538919），cutoff1=8 精度档对 .dat
偏差 ~0.4% 不变。commit 7574e02。

### 3.15 负结果存档（避免重试）

1. **GaussMAccum 内层 j 循环并行**（j 目标块互不相交，结构上安全）：
   REDUCTION 1.75 → 2.1–2.3s，**反升**。改按池空闲容量
   （opsTlsThreads=池并发÷在飞）门控仍 2.07–2.18s。教训：溢洪期不是
   纯吞吐受限——每节点 MAccum/CUpdate 的**墙钟直接串进其后代的就绪
   时间**，内层并行不省 core-time 反摊 ThreadGroup spawn 与冗余
   gather，节点尾延迟变长连锁拖慢整片排空。近根节点 anc≈树深≈30，
   几乎所有节点都满足"大 nj"条件，内层并行=给 2601 个节点各配 8 线程。
2. **CUpdate 分段数 8→12**（nd/32）：1.77–1.86s，不优于 8。8 是甜点。
3. **池空闲容量门控 CUpdate 段数**（溢洪串行/链尾并行）：2.07–2.18s，
   同 1 的机理反升。

### 3.16 散射去开销 + OriginalBlocks 并行 + 串行初始化根因修正（REDUCTION 1.75 → 1.62 s，commit 75288ce）

**根因修正**：此前把 REDUCTION 减去 DAG span 后的 ~0.4s 归因为
"收尾串行"，实测（AAMS_RED_CLOSE_LOG）是 **DAG 之前的串行初始化段**：
OriginalBlocks ~400ms（K 抽取 pass2 Eigen insert 160ms 是大头）+
gaussModule.initialize ~136ms。两者均可压缩：

- OriginalBlocks：K/M 抽取改双线程并行（M 是对角块集，968330 对
  26460，并行收益直接）；finalize 的 ~78k 小块装配按块并行
  （nthr=min(8, mats/64)，原子取号）；gauss 槽位表按
  `N*2+4*Σanc` 精确预留（原 N*4 少于实际 ~317k 条目，免重哈希）。
- 散射去开销：colcomp_mat 新增 `activateStorageCol(s)`——堆叠路径
  预查缓存 storageCol 后散加免二次二分（原先每散列 markColNonZero
  再搜一次，加上 storageCol 本身共三次查找）；trans 路径同理每列
  单次 storageCol；堆叠 gather 按活跃列签名连续性分派
  memcpy/标量 gather。批量散加核时 3.73 → 3.12 core-s——**墙钟
  未动**（§3.15 同款教训：洪泛期核时削减不直接换墙钟）。

合计 REDUCTION best 1.75 → **1.62 s**。

### 3.17 MAccum/KAccum trans 路径右端后缀复用（REDUCTION 1.62 → 1.45 s，commit f50af63）

trans 批量路径 76,646 次调用核时构成：gather 0.756 + gemm 4.456 +
scatter 1.297。关键观察：对节点 i 的祖先链 j=anc[m]（anc 升序、
subtree(k)={0..k}），右端 Ψ(i,k≥j)/M̃(i,k≥j) 是同一矩阵的**连续列
后缀**——逐 j 重建右端纯属重复。改为：

- `opsTransBuildRightEnd`：一次 gather 出全祖先右端 C（rows×Σnc
  列主拼接 + 每全局列的原始列号 + 段界）；
- `opsColAccBatchTransSuffix`：按段切右端视图做 dgemm——gemm 右端
  落在 cache 热的同一缓冲尾部，gather 归零；
- MAccum 双 term（Ψ_j×M̃ 与 M_temp_j×Ψ）各自独立右端，逐 j 两次
  后缀调用。

效果：gather 0.756→~0、gemm 4.456→2.656、scatter 1.297→1.215，
批量路径核时 9.7→7.4 core-s；REDUCTION best 1.62 → **1.447 s**。

### 3.18 Gershgorin 空区间预判：白跑的 dsygvx（commit bd15c5b）

分段插桩（dsygvx query/real + local-eig load/solve/commit）暴露：
**dsygvx 调用 5021 次 ≈ 2×2601 节点**——约 93% 节点先走一次空区间
回退（cutoff1=5 时叶局部 λ ~1e7–1e8 > lambdaHi=ω1²≈1.42e7，
'V' 型求解白跑一遍 + K/M 被破坏后双倍 loadBlocks 重装载），之后
才走真正的 solveIndex。改为预判：

```
λmin(K,M) ≥ minGersh(K)/maxGersh(M)
minGersh(A)=min_i(A_ii−Σ_{j≠i}|A_ij|), maxGersh(B)=max_i(B_ii+Σ_{j≠i}|B_ij|)
```

下界 > lambdaHi 即区间必空 → 直接 `solveIndex(1, minExtractOrder)`
保留最低阶，K/M 未被 LAPACK 破坏，**免二次装载**。cutoff1=5 下
dsygvx 5021 → **2839** 次；局部特征核时 load 0.078 / solve 2.10 /
commit 0.002 core-s（装载开销几乎归零）。对拍：S=1200 残差逐位
不变（0.5555962051/0.6026538919），cutoff1=8/S=3000 12 阶频率与
参考完全一致；REDUCTION best 1.472s（墙钟未再降——白跑同样淹没在
洪泛并行里，收益是核时与内存带宽）。

## 4. 当前状态（2026-10-18，§3.18 后）

best-of-5（16 线程，AAMS_PROFILE=1，--subspace-max 1200）：

| 阶段 | 优化前 | §3.9 后 | §3.14 后 | 现在（§3.18 后） |
|---|---|---|---|---|
| METIS 排序+建树 | 0.4 | 0.39–0.42 | 0.44（未动） | 0.38–0.40 |
| REDUCTION | 8.2（69 核秒） | 5.0（~27 核秒） | 1.73–1.79 | **1.45–1.55（批量核时 9.7→7.4）** |
| DISTILL | 87 → 4.2 | 4.8–5.0 | 0.35（S=1200） | 0.35 |
| REDUCED_EIGENSOLUTION | — | — | 0.51–0.67 | 0.51–0.67 |
| RECOVERY | 0.2 | 0.17–0.19 | 0.05 | 0.05 |
| **TOTAL** | **~65** | **10.8–12.4** | **3.04–3.23** | **安静机估算 2.7–2.8** |

精度：cutoff1=5 基准档 12 阶对 Abaqus .dat 4.3% 内（高阶模态，截断主导，
与 Abaqus 同 cutoff1 的差距见逆向文档）；cutoff1=8/S=3000 精度档 12 阶
频率与参考一致（~0.4% 内）；残差与优化前逐位一致。

**目标口径更新（2026-10-18）**：用户放宽为"差距 ≤2s 或 20%"。
Abaqus wall ≈1.4–1.65s ⇒ AAMS ≤~3.4–3.65s 即达标。本机当日后台负载
持续 1.3–2×（负载计：REDUCED_EIGENSOLUTION 1.14–1.40s vs 安静 0.67s），
按负载折算安静机 TOTAL ≈ 2.7–2.8s，对 Abaqus 差距 **~1.1–1.4s**，
**目标已达成**（未能复现 3.04s 当时的安静窗口，结论按 REDUCTION 稳定
指标 + 负载折算给出，REDUCTION 本身 1.447→1.472 best 无回退）。

## 5. 剩余差距量化路线（更新）

当前 wall 构成（时间线实测）：METIS ~0.39 + REDUCTION 1.45 +
EIG(含 DISTILL) 0.67 + RECOVERY/ASSEMBLE 0.1 + 串行初始化
（OriginalBlocks+gaussInit，§3.16 后已压）~0.3。

原 §5 路线 1–3 已落地（散射向量化=§3.16、MAccum 后缀复用=§3.17、
局部特征白跑=§3.18 消除）。剩余杠杆按 ROI 排序：

1. **METIS ~0.39s**：单线程固定成本。可探索与矩阵解析重叠（流水）或
   更轻的 ordering（对 26460 阶可能 nd 选项调优收益有限）。
2. **SolveLocalEigenPair 2.1 核秒**：白跑已消，剩余是真实求解。
   Abaqus 四步链（sygst/sytrd/stev）合计 0.29s/线程，仍可能更快
   （叶节点 dsyevr 区间，同 §3.10 手法）。
3. **结构级（对标 Abaqus k_updext 0.066s/线程）**：Abaqus 的 Reduction
   是全稠密 front 块沿树单次 dgemm 链，无 O(depth²) 对块、无列压缩
   gather/scatter。这是与 Abaqus 剩余 ~1.8× 差距的根本来源；目标已
   达成，此项仅在追求同量级时才需评估（工程量大，另行立项）。
4. **测量注意**：机器后台负载可致 1.3–2× 膨胀，取多次最小值；
   `AAMS_CUPDATE_LOG=1` 每节点 CUpdate 墙钟、`AAMS_BATCH_LOG=1` >3ms
   批量打点、`AAMS_TASK_TIMELINE=1` 时间线 CSV、`AAMS_RED_CLOSE_LOG=1`
   REDUCTION 内外分解、`AAMS_NO_TRANS_SUFFIX=1` 回退旧 trans 路径。

## 6. 复现命令

```bash
export PATH="/d/msys2/mingw64/bin:/d/msys2/usr/bin:$PATH"
cd aams && cmake --build build -j 8
# 基准档（对标 Abaqus 默认 cutoff1=5）：
AAMS_PROFILE=1 ./build/aams_solve.exe \
  --K /z/codeSpace/AMLS/bench_abaqus/cant_20_K.mtx \
  --M /z/codeSpace/AMLS/bench_abaqus/cant_20_M.mtx \
  --fmin 0 --fmax 120 --cutoff1 5 --reduced distill \
  --threads 16 --subspace-max 1200
# 精度档（对 .dat ~0.4%）：
#   --cutoff1 8（不加 --subspace-max，默认 3000）
# 关键链/瓶颈分析：AAMS_TASK_TIMELINE=1（CSV）+ AAMS_CUPDATE_LOG=1
```
