# AAMS 性能优化：设计思路与实现路径总结（2026-10-18）

本文是对 aams（按 Abaqus AMS 逆向成果用 C++ 复刻的实现）整体优化工作的
**设计思路与路径**总结。逐项优化的细节、证据与数据在
`AAMS性能深挖_cant_20.md`（下称"深挖文档"），逆向算法背景在
`abaqus_ams 逆向文档`。本文回答三个问题：优化遵循什么原则、按什么路径
推进、沉淀了哪些方法论。

## 1. 背景与目标状态

对标对象：Abaqus 2020/2025 的 AMS 特征值求解器（经 REA+Ghidra 逆向，
主流程 std_pamsDriver：METIS 超节点树 → 子结构局部特征（LAPACK 四步链）
→ 沿树传播缩聚 → 蒸馏+子空间迭代（Ritz）→ 三阶段 Solve 恢复）。

| 里程碑 | cant_20 (26,460 DOF) | cant_31 (95,232 DOF) |
|---|---|---|
| 优化前 TOTAL | ~65 s | — |
| 当前 TOTAL | 安静机估算 2.7–2.8 s（best 3.04） | ~31.7 s（16 线程 best-of-3） |
| Abaqus wall | ~1.4–1.65 s（4 线程） | ~19.6 s（4 线程） |
| 差距 | **~1.1–1.4 s，≤2s 口径达标** | ~12 s（~1.6×），依赖链瓶颈 |

精度纪律贯穿全程：基准档（cutoff1=5）残差与优化前**逐位一致**
（0.5555962051/0.6026538919），精度档（cutoff1=8/S=3000）12 阶频率与
Abaqus .dat 参考一致。任何优化先过对拍再谈性能。

## 2. 设计思路：五条原则

### 原则一：热路径零 Eigen、全 BLAS/LAPACK 后端

最初 87s 的 DISTILL 阶段根因是 Eigen 的 LDLT/SelfAdjointEigenSolver/
mOrthoNormalize 无 BLAS 后端。改为 LAPACK（dpotrf/dsytrf/dpotrs/
dsyevd/dsygvd + dgemm 显式调用）后单点 20×。此后成为硬性约束：
所有新代码（列压缩、批量累积、Ritz）只允许 BLAS/LAPACK 调用，
且注意 LAPACK 的就地破坏语义（DSYGVX 把 B 变 Cholesky 因子、
DSYEVR 的 Z 不得与 A 别名——都是踩过并写进记忆的坑）。

### 原则二：存储为散加/累积模式定制（列压缩 ColCompMat）

REDUCTION 的本质是沿树把 K̃/M̃/Ψ 按"祖先对"散加进累积矩阵。设计：

- ColCompMat 列压缩存储：结构态（标记列）与数值态分离，
  activateStorageCol 按 O(1) 激活已知存储列，散加免重复二分
  （原先每散列三次查找：storageCol + markColNonZero + 数值化）。
- dgemmColsRaw 包装：面向"多源块 × 同一累积矩阵"的列式 dgemm，
  小乘法门控（2mnk<3e7 走串行，避免 ThreadGroup spawn 的 ms 级
  固定开销吃掉收益）。
- 活跃列签名（活跃原始列序列）作为分组合并的判据——后继所有
  批量化都建立在"同签名 → 共享同一右端"这一观察上。

### 原则三：DAG 任务池 + 依赖驱动，不用全局屏障

REDUCTION 的 2601（cant_20）/8859（cant_31）个节点消元组织成 DAG
任务池：每节点 SolveLocalEigenPair → Gauss(M/K/MTemp)Accum →
Compress(Left/Update/Right) → Release 系列任务，就绪即发射。
调度加树深优先级（浅节点先做，避免深链尾饥饿）。

**关键约束（三次实验换来的）**：溢洪期不是纯吞吐受限——每个节点
任务的墙钟直接串进其后代的就绪时间。因此"只省 core-time、不省
节点墙钟"的改动（如内层 j 循环并行、散射向量化）不换墙钟甚至
反升（负结果存档见深挖文档 §3.15）。评估改动必须 A/B 墙钟，
core-time 只作为定位手段。

### 原则四：批量化三层递进（这是最大的收益来源）

对"每节点 × 每祖先对"的海量小乘法（初版 4.58M 次调用 55 GFLOP 花
37–54 核秒，OpenBLAS m=1 微 gemm 固定开销 0.87ms/次），按三层递进：

1. **去 O(1) 调用**：校验类操作缓存化、空块短路、缓冲复用、
   统计门控（3.4/3.5 节）——消灭与数据规模无关的固定开销。
2. **按签名分组 + 堆叠（stacked batching）**：同节点的全部
   descendant 共享 Ψ → 按活跃列签名分组共享一次 C_all gather，
   组内行堆叠成单次大 dgemm，再按原序散射。最大单项：
   REDUCTION 3.04 → 1.73s，calls 149k → 79k。
3. **右端后缀复用（suffix reuse）**：祖先链升序 ⇒ 对节点 i 的
   祖先 j=anc[m]，右端 Ψ(i,k≥j) 是同一矩阵的连续列后缀——
   一次 gather 出全祖先右端（opsTransBuildRightEnd），逐 j 只切
   视图做 dgemm（opsColAccBatchTransSuffix），右端落在 cache 热
   的同一缓冲。gather 0.76→0、gemm 4.46→2.66 核秒，
   REDUCTION 1.62 → 1.447s。

### 原则五：分析驱动的捷径（数学性质换计算）

不是所有收益都靠并行/批量化，有的靠"先算便宜的上/下界避免昂贵的
主计算"：

- **Gershgorin 空区间预判**：λmin(K,M) ≥ minGersh(K)/maxGersh(M) >
  λ上界 ⇒ 区间必空，直接 solveIndex 保留最低阶——消灭了 93% 节点
  白跑的空区间 dsygvx（5021→2839 次）+ K/M 破坏后的双倍重装载。
- **M 稀疏性利用**：M 是集中质量（对角块集），抽取/批量路径按
  此特化。
- **连续性分派**：gather/散加前检查活跃列是否连续，连续则
  memcpy/AXPY，否则才走索引循环（运行时分支代价远小于普遍标量）。

### 配套：插桩体系（可测性是设计的一部分）

环境变量族，全部生产关闭、按需打开：
`AAMS_PROFILE`（任务计时/计数 + 各 prof 分段）、`AAMS_TASK_TIMELINE`
（DAG 时间线 CSV，关键链分析）、`AAMS_RED_CLOSE_LOG`（REDUCTION
内外分解）、`AAMS_CUPDATE_LOG`/`AAMS_BATCH_LOG`（逐节点/大批量
打点）、`AAMS_NO_TRANS_SUFFIX`/`AAMS_NO_MT`（新旧路径 A/B 开关）。
没有这套东西，§2.3 的"墙钟 vs 核时"与 DAG span、串行初始化根因
等发现都不可能做出。

## 3. 实现路径：四个阶段

```
Phase 0 功能对标          → Phase 1 求解器去 Eigen → Phase 2 批量化
（aams/ 独立项目，         （DISTILL 87→4.2s，       （REDUCTION 8.2→1.73s，
 12 阶对 Abaqus ≤0.5%）     TOTAL ~65→10.8s）         TOTAL 10.8→3.04s）
                                          ↓
                          Phase 3 深挖收官（散射/并行抽取/后缀复用/Gershgorin，
                          REDUCTION 1.73→1.447s，TOTAL 安静机估算 2.7–2.8s，达标）
```

- **Phase 0（功能）**：按逆向文档复刻算法，验收是精度（蒸馏补足
  3000 子空间后 12 阶 vs Abaqus ≤0.5%；缩聚解 55s vs direct 702s）。
- **Phase 1（单点热点）**：profile 定位无 BLAS 后端的 Eigen 调用，
  全部替换。特征：改动局部、收益巨大、无算法改动。
- **Phase 2（算法级重构）**：把"每对一乘"改为"分组合批堆叠"，
  涉及 col_ops 接口族与 updateColComp_ 重构。特征：需要证明
  落点与累加次序不变（残差逐位一致是验收线）。
- **Phase 3（榨干剩余）**：插桩深挖 → 根因逐个消灭（散射三次查找、
  串行初始化误判修正、后缀复用、Gershgorin）。特征：每步收益变小、
  需要更细的测量与更严格的 A/B（含负结果快速止损）。

git 上每子任务一个 commit（7574e02 → 75288ce → f50af63 → bd15c5b），
feature-reaAMLS 分支，文档与代码同步提交。

## 4. 方法论沉淀

1. **对拍纪律**：任何性能改动前后跑双对拍——基准档残差逐位一致 +
   精度档参考频率一致。批量化重构的"落点与累加次序不变"靠它证明。
2. **墙钟 vs 核时**：并行洪泛期，core-time 削减不保证墙钟下降
   （散射 3.73→3.12 核秒墙钟不动）；依赖链期，core-time 削减可能
   反升（内层并行三次负结果）。唯一判据是控制变量的墙钟 A/B。
3. **测量学**：机器后台负载可膨胀 1.3–2×，取 best-of-N；REDUCTION
   是最稳的跨时段指标；用安静基准阶段（如 REDUCED_EIGENSOLUTION
   0.67s）做负载计折算。
4. **根因修正要有证据**：把 REDUCTION 的 ~0.4s 差额从"收尾串行"
   改判为"DAG 前的串行初始化"（OriginalBlocks 400ms + gaussInit
   136ms）靠 AAMS_RED_CLOSE_LOG 分解，修正后才可能对症（并行抽取 +
   精确 reserve）。
5. **负结果立即存档**（深挖文档 §3.15）：防止重试，也防止"结构上
   安全所以应该快"的直觉误导——安全≠更快。
6. **LAPACK 语义坑入库**：DSYGVX 就地破坏输入、DSYEVR Z 别名、
   空区间回退重用被破坏矩阵（cant_20 垃圾结果单根因）——同类
   LAPACK 包装代码一律先查这几条。

## 5. 当前状态与下一步

- cant_20：**达标**（≤2s 口径）。剩余路线（METIS 流水化、叶 dsyevr、
  结构级 frontal 重构）仅在追求与 Abaqus 同量级时立项。
- cant_31：**达标**（2026-10-18，详见 §7）。TOTAL best 20.66s vs
  Abaqus ≈19.6s，差距 ~1.0–1.5s ≤ 2s；REDUCTION 依赖链本身的
  算法级差距（AMLS 全 Ψ 累积 ~124 核秒 vs Abaqus 缩聚树 ~13.5
  核秒）仍在，本轮靠并行化串行初始化链（解析/树/抽取/槽位）与
  按需建槽补齐。若追求同量级，frontal 化仍是正道。
- 内存：16G 机器当前配置可测 ~12–13 万 DOF（cant_31 峰值 6.21GB，
  ∝DOF^1.4）；瘦配置实测省不了（大头在 REDUCTION 稠密块）。

## 6. 代码地图（对应上述设计）

| 模块 | 位置 | 设计原则 |
|---|---|---|
| 列压缩存储 | aams/src/linalg/colcomp_mat.hpp | 原则二 |
| 批量累积/散加 | aams/src/linalg/col_ops.cpp（BatchStacked/BatchTrans/BatchTransSuffix） | 原则四 2/3 |
| DAG 任务池 | aams/src/ams/node_task_executor.cpp（含 Gershgorin 预判） | 原则三/五 |
| 祖先累积 | aams/src/ams/gauss_module.cpp（suffix 复用装配） | 原则四 3 |
| 蒸馏/Ritz | aams/src/ams/…（LAPACK 全后端） | 原则一 |
| OriginalBlocks 抽取 | aams/src/ams/original_blocks.cpp（双线程+并行 finalize） | 原则五/Phase 3 |
| 驱动与计时 | aams/src/ams/pams_driver.cpp | 插桩体系 |

## 7. cant_31 专场（2026-10-18）：差距 ~12s → ~1.0–1.5s

目标（用户）：cant_31 上 AAMS TOTAL 与 Abaqus AMS 差距缩小到 2 秒内
（Abaqus 墙钟 ≈19.6s ⇒ 目标 TOTAL ≤ ~21.6s）。过程严格执行
**REA 定位差异点 → REA 分析商软技术细节 → 改 AAMS → 验证**的迭代，
不做无证据猜想。

### 7.1 REA 证据基线

- `bench_abaqus/cant_31.times`（ABA_PAMS_ 导出，4 线程）：每线程
  12.86/12.86/19.55/19.55 核秒；m_stackext 16.6 核秒完美均衡、
  局部特征（sygst/sytrd/stev/ormtr/trsm）~7.5、传播（k_updext 4.2 +
  k_bwd_psi 4.1 + m_updext 5.2）全部摊进固定小队 ⇒ 高利用率。
- AAMS REDUCTION 任务核时（8859 节点）：GaussMAccum 45.4 +
  CompressUpdate 29.0 + GaussKAccum 28.9 + …≈ **124 核秒**，传播段
  为 Abaqus 等效 ~13.5 核秒的 ~9×。这是 AMLS 全 Ψ 累积 vs Abaqus
  子结构缩聚树（Craig-Bampton 静凝聚 + 缩聚局部特征）的**算法级
  差距**，本轮明确不在会话内修（正面强攻 = frontal 化重构）。
- DAG 尾段（最后 ~6s，利用率 ≈2/16）：近根 ~23–60 个任务串行链，
  单任务 MAccum ~120ms（大量 <3ms 的 trans-suffix 批量 gemm）。

### 7.2 本轮改动（全部残差逐位同验收）

| 项 | 前 | 后 | 说明 |
|---|---|---|---|
| 矩阵解析（并行 parser） | K 4.5s | 0.37s | SMB 读 45s 需预热缓存后再计时 |
| 纯符号分析 | treebuild 4.5s | 1.59s | 只算 fill1 不做数值 |
| OriginalBlocks 抽取 | 1.47s | **0.50s** | pass1/pass2 按节点分线程 + 暂存合并（修耦合块双源写竞争）；**按需建槽**：count==0 的祖先对不建槽（K 块 555k→89k，消去模块走 findSub* nullptr→零初始化路径） |
| gaussInit | 892ms | **433ms** | 2.2M Entry 并行预分配 + 串行 map 搬移；planColumns_ 按层级并行（lvl(j)=1+max(lvl(孩子))，同层节点依赖已就绪、槽位互不相交） |
| TOTAL（cant_31） | 22.2s（best） | **20.66–21.06s** | 3 次 best-of-N，热缓存 |
| TOTAL（cant_20 A/B） | — | 2.26–2.53s | 残差 0.5555962051/0.6026538919 逐位同 |

### 7.3 负结果与教训（本轮新增）

1. **预算单持有锁未释放**（本轮最大事故）：改 ScopedReservation
   构造时把 `g_mtBudgetLock.clear()` 从析构删了 → 首个成功授予者
   永久持锁，全部后续 dgemm 借贷死掉（parallel=1），colAcc gemm
   50→151s、REDUCTION 16→33s。当时误判为"公平份额借贷灾难"，
   靠跨 run 对比 [gemmcols]/[redclose] 计数器（contended 2085 vs
   214）才定位。**改锁语义必须成对检查 acquire/release 路径。**
2. **近根 m-循环借贷无净收益**：parallelAncLoop 借线程并行化在
   cant_31 实测劣化（K/M 并发累积互饿 + helper 生成开销），公平
   份额版（acquireLendFair）干净数据点为中性（16.36 vs 16.0s），
   已回退为串行并删除死代码。Abaqus 的高利用率来自**操作内部**
   固定小队并行，不是任务间借线程——AMLS 的 per-call 借贷粒度
   太细，学不来。
3. **Eigen 暂存槽必须显式定尺寸**：`unordered_map[key]` 默认构造
   0×0 SparseMatrix 后 insert 越界 → 堆破坏 segfault（第一次并行
   化的崩溃根因，非"节点划分错误"残余）。
4. **串行初始化链是隐藏大头**：2.65s 的"解析+建树"里一半是
   OriginalBlocks/gaussInit；AAMS_RED_CLOSE_LOG 分解（§4.4）再次
   值回票价。ginit 拆分显示 slots 361ms / plan 531ms，对症下药。

### 7.4 复现命令

```bash
# cant_31（热缓存 + 三连取 best）
cat ../../bench_abaqus/cant_31_STIF2_free.mtx > /dev/null
AAMS_PROFILE=1 AAMS_RED_CLOSE_LOG=1 ./aams_solve.exe \
  --K ../../bench_abaqus/cant_31_STIF2_free.mtx \
  --M ../../bench_abaqus/cant_31_MASS2_free.mtx \
  --fmin 0 --fmax 120 --cutoff1 5 --reduced distill \
  --subspace-max 1200 --threads 16
# cant_20 对拍
./aams_solve.exe --K ../../bench_abaqus/cant_20_K.mtx \
  --M ../../bench_abaqus/cant_20_M.mtx --fmin 0 --fmax 120 \
  --cutoff1 5 --reduced distill --threads 16 --subspace-max 1200
```

注意：stderr 必须重定向到本地盘（Z: 是 SMB，逐行日志会把运行
拖成灾难）；机器后台负载可致 1.3–2× 膨胀，结论一律 best-of-N。
