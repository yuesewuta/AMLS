# REDUCTION 传播段 frontal 化设计（2026-10-18）

> 证据标注：[R] = Abaqus 逆向实测（Ghidra 反编译 / ABA_PAMS_ 导出）；[A] = AAMS 实测
> （AAMS_PROFILE 计时/计数）；[INF] = 推断。

## 0. 结论先行

cant_31（95,232 DOF，16 线程）上 AAMS REDUCTION 传播段与 Abaqus 的差距
**不是算得多，而是算得慢**：

| 指标 | AAMS [A] | Abaqus [R]/[A] |
|---|---|---|
| 堆搭建（stack building） | CompressUpdate 29.0 core-s | k_stackext 9.7 + m_stackext 16.6 = 26.3 core-s（持平） |
| 对传播（pair propagation） | GaussKAccum 28.9 + GaussMAccum 45.4 = 74.3 core-s | k_updext 4.2 + m_updext 5.2 + k_bwd_psi 4.1 = 13.5 core-s（**5.5×**） |

AAMS 传播总 flops ≈ 8.0e11 [A]；Abaqus 13.5 core-s 按稠密 tile dgemm
50–100 GF/s 折算 ≈ 0.7–1.4e12 [INF] —— **同一量级**。差距全部在实现效率：
skinny gemm（m 或 k ≈ 4）+ 逐列散射。

## 1. AAMS 传播段实测剖析 [A]（cant_31，AAMS_PROFILE，flop 计数）

colAcc 内核分解（本次新增 flops/散射体量计数后实测）：

| 路径 | gemm | GF/s | flops | scatter | 散射体量 | 散射速率 |
|---|---|---|---|---|---|---|
| trans（KAccum/MAccum 对传播） | 29.1 s | **17.0** | 4.95e11 | 22.7 s | 2.57e9 元（19.2 GB） | 0.85 GB/s |
| batch（μ 传播 + CompressUpdate） | 46.8 s | **6.6** | 3.08e11 | 58.5 s | 1.35e9 元（10.1 GB） | **0.17 GB/s** |

诊断：

1. **两条 gemm 都是 memory-bound skinny gemm**。trans 路径 m = n_j ≈ 4
   （单次调用一个 j，Ψ(I,j) 的列数），batch 路径 k = actB ≈ 4（节点约束
   DOF 数）。算术强度 ≈ 0.5 flop/byte，6.6–17 GF/s 已接近该形态的内存墙
   —— 效率提升只能靠**把 skinny 维度批量化**。
2. **散射是逐 touch 开销主导**（batch 0.17 GB/s）：每列锁 + storageCol
   二分 + activateStorageCol，连续段长度只有 rj ≈ 10 双精度量级，
   每 touch 约 2 µs。
3. Abaqus 对照 [R]（C:\tools\pams_exttasks.txt，150 函数反编译）：
   KUpdExtTask::PerformTask 在 **pme_DmpBlockManager 稠密 tile** 上工作
   （AccessBlock/ReleaseBlock），计算链 = pme_DmpBlockAssembleNoaux（一次性
   sparse→dense 组装）+ blac_dgemm + **tri_rectupdate/rect_triupdate**
   （三角矩形更新）+ pme_InvertDiagonals + CopyFactor —— β=1 直接累加，
   无散射、无逐块稀疏间接寻址。ComputeTaskFlops = block_size²×(stack_width+1)
   （稠密对称 rank 更新 flop 模型）。

## 2. 内存约束 [A]

- 全稠密化所有对块：9.43 GB（16 GB 机器、当前峰值 6.2 GB）—— 不可行。
  诊断常驻：`[ginit] ... mem dense=%.2fGB planned=%.2fGB ratio=%.2f`。
- 逐节点 arena（把节点 j 的全部对块按 validCols 连续拼接）：与现布局同量级
  （planned ≈ 1.36 GB）—— 可行。
- ColCompMat 拥有自己的 Eigen dense 缓冲（colcomp_mat.hpp:126），arena 视图
  需要改 sealed 类 —— Slice 2 再议；Slice 1 不动存储。

## 3. 改造方案（分片，每片对拍）

### Slice 1：trans 路径左端批量化（全积 gemm）——已实验，负结果

现状（gauss_module.cpp accumulateReducedStiffness/Mass）：对每个子孙 I，
对 j ∈ anc(I) 并行（parallelAncLoop），每个 (I,j) 调
opsColAccBatchTransSuffix：gemm m = n_j ≈ 4、n = 后缀宽、k = n_I。

改为：每 I 构建堆叠左端 Ψ_all(I)（n_I × mAll）一次 dgemm 全积
P = Ψ_allᵀ · RE（mAll × nAll），散射仅取 j ≤ k 上三角块。已实现
（opsColAccFullProductTrans，AAMS_TRANS_FULL=1），cant_31 实测：

- trans flops 4.95e11 → 6.48e11（1.31×，活跃列 gather 后浪费 <2×），
  **GF/s 17.0 → 15.4（不升反降）**，REDUCTION 13.65 → 15.84s。**回退默认**。
- 根因 [A]：DAG 洪泛期 `opsTlsThreads = cap / inFlight ≈ 1`
  （node_task_executor.cpp:429）——大 gemm 不涨线程也不涨 GF/s，只涨
  flops。**在 AAMS 的 DAG 线程预算体制下，"更大 gemm" 不是杠杆**；
  Abaqus 的 13.5 core-s 优势只能来自流（stream）消除，不是 flop 形态。

### Slice 1b：散射 actRun 连续 AXPY —— 中性

opsColAccBatchTransSuffix 补上与 BatchTrans 相同的连续 AXPY 快路径：
散射 22.7 → 22.5s（持平）。结论：散射瓶颈不在索引寻址，在**逐列
touch 开销**（连续段长仅 nb ≈ 4–12 双精度 → 2.57e9 元 ≈ 3 亿列 touch）。

### Slice 1c：直写快路径（tri_rectupdate 语义）—— 已实验，命中率为下一步杠杆

opsColAccBatchTransSuffix 增加直写路径（AAMS_TRANS_DIRECT=1）：B 存储
布局原始列连续 且 目标段存储列连续时，dgemm β=1 直写目标列块（免乘积
物化 + 免散射，即 Abaqus tri_rectupdate 的矩形更新）。cant_31 实测：

- 命中 29413 / 回退 293132（**命中率 9%**）；
- 散射体量 19.16 → 12.96 GB（-32%），散射 22.5 → 17.3s；
- 同窗口 best-of-2：REDUCTION 直写 14.28/15.07 vs 基线 14.57/15.15
  （机器负载漂移 ±10%，~0.2–0.3s 边际改善，与 9% 命中率一致）；
- 精度：cant_31 残差 bit 一致（0.7970324203/0.7304834395），
  cant_20 bit 一致（0.5555962051/0.6026538919）。

**命中率低的原因 [A/INF]**：目标块 validCols 是全体子孙源活跃列的并集
（列规划定理保证覆盖），单个子孙的段只是其子集且不连续 → 存储列连续
条件大多数失败。**Slice 2 的核心杠杆 = 布局规划对齐**：把目标块列
规划改为与传播源段对齐（Abaqus 的 tile 布局天然对齐——assemble 一次
承担 pattern→tile 映射，其后全部更新都是稠密 β=1）。

### Slice 2：布局对齐 + 直写为默认（下一步主体）

逐节点连续 arena + 单锁/节点；列布局对齐时 gemm β=1 直写（等价
tri_rectupdate 的矩形更新语义），不对齐时段内连续散射。存储仍 validCols
（内存不变）。可选：ColCompMat 增加外部缓冲视图构造。

### Slice 3：batch 路径（μ 传播）同治

Y 物化（10.1 GB）+ 逐列散射 → 堆叠 gemm 直写 arena（μ(d,·) 跨祖先连续），
消散射。

## 4. 验收标准

1. cant_31 REDUCTION 墙钟下降（基线 TOTAL 20.66–21.06s best-of-N 热缓存；
   目标传播段 74.3 → ≤40 core-s [A]）。
2. 精度：残差 bit-一致或末位 ulp 容差（累加次序变化允许；基线
   first=0.7970324203 last=0.7304834395）；前 10 阶频率 vs Abaqus ≤0.25%。
3. cant_20 回归：TOTAL 与残差不劣化。
4. 文档 + commit（仅 aams/）+ push。

## 5. 复现

```bash
cd aams/build
cat ../../bench_abaqus/cant_31_STIF2_free.mtx > /dev/null   # 热缓存
AAMS_PROFILE=1 ./aams_solve.exe \
  --K ../../bench_abaqus/cant_31_STIF2_free.mtx \
  --M ../../bench_abaqus/cant_31_MASS2_free.mtx \
  --fmin 0 --fmax 120 --cutoff1 5 --reduced distill \
  --subspace-max 1200 --threads 16 2> local.log
grep -E "colAcc (flops|trans|batch)" local.log
```

stderr 必须重定向到本地盘（Z: 是 SMB）；结论一律 best-of-N。
