# Abaqus AMS（Automatic Multi-level Substructuring）技术分析

> 目标：分析 Abaqus/Standard 中 `EIGENSOLVER=AMS` 特征值求解器的算法技术细节，
> 对照本项目自研 AMLS 实现，定位"为什么 Abaqus AMS 又快又省内存"，给出
> 可落地的优化方向。
>
> 方法：Abaqus 官方文档（在线需登录，离线未装）+ AMLS 奠基文献
> （Bennighof / Kaplan / Kim 等 UT Austin 团队论文，本项目 `papers/` 目录齐全）。
> 标注：[DOC] = Abaqus 官方文档明确；[PAPER] = AMLS 论文明确；[INF] = 推断。
> 论文缩写：[Kim04] = Kim 2004 博士论文；[Gao08] = An Implementation and
> Evaluation of the AMLS Method for Sparse Eigenvalue Problems (Gao 2008)；
> [BL04] = Bennighof/Lehoucq，SIAM 2004（线性弹性动力学 eigenspace）；[HM] =
> 混合 OpenMP/MPI（Wang 2026）；[FG] = 细粒度并行（Wang 2025）；[GPU] = GPU
> 加速（Wang 2024）；[SIM] = Yin/Voss/Chen 2013 子空间迭代；[EAMLS] = Hyun
> 2019 增强 AMLS；[OOC] = out-of-core（Wang）。

---

## 0. 一句话结论

**Abaqus AMS = AMLS 方法的商业级工程实现**（源自 UT Austin Bennighof 团队，
Abaqus 6.14+ 内置）。它把 FE 矩阵自动递归剖分成成千上万个小子结构，对每个
子结构做局部谱变换，只保留低于截止频率的子结构模态，把全局问题投影到一个小
得多的子结构模态子空间上求解——**把"求大特征问题"变成"求一堆极小局部特征
问题 + 一个小全局问题"**，从而比 block Lanczos 少几个数量级的浮点运算和内存。

---

## 1. AMLS 的软件阶段划分（[PAPER]）

UT Austin AMLS 软件明确分 5-6 个独立程序阶段（[Kim04] §1.2）：

| 阶段 | 名称 | 做什么 |
|---|---|---|
| Phase1 | FE 矩阵生成 | 由 FE 软件（MSC.Nastran 等）生成 K、M |
| Phase2 | 剖分 | 读矩阵稀疏结构，自动递归分成**数千个子结构、数十层** |
| Phase3 | 子结构谱变换 + 投影 | 计算**数万到数十万**个子结构模态（低于截止频率），把 K、M 投影到子结构模态子空间 |
| Phase4 | 全局特征解 | 解缩聚后的全局小特征问题（本文重点是它） |
| Phase5 | 向量恢复 | 把全局特征向量恢复到 FE 自由度空间 |
| Phase6 | 频响分析 | 模态频响（不属本讨论） |

**对 Abaqus AMS**：[INF] Abaqus 把 Phase2–Phase5 封装在求解器内部（对用户
不透明），但算法结构同源。

**耗时分布**（[HM][GPU]）：模型变换（Phase3，即子结构谱变换 + 投影）是**最重
阶段**，占总运行时间 **>60%**（[HM]）/ **>50%**（[GPU]）。这意味着优化 Phase3
是提速的首要目标——本项目当前也把最多功夫花在 `gaussian_elimination.cpp` 和
`subspace_condensation.cpp`（对应 Phase2b/3），方向正确。

---

## 2. 剖分策略（[PAPER] + [INF]）

- 剖分**基于矩阵稀疏结构**（邻接图），不基于几何：[SIM] "nested dissection/
  METIS 对非零图做 fill-reducing 排序"。
- 叶子子结构规模有上限；层级"dozens of levels"（[Kim04]）。
- [INF] Abaqus AMS 用**嵌套剖分**（与 METIS_NodeND 同类），因为它产生更少
  fill-in 且更平衡；本项目 `amls_solve --partition nd` 已实现 ND（见
  `src/partition/metis_partitioner.cpp`），方向一致。
- **剖分质量决定一切**：好的剖分 → 子结构内聚、界面稀疏 → 局部谱便宜 +
  缩聚后的 M̃ 更稀疏。

### 2.1 K/M 的分块结构（[SIM][EAMLS]）

重排序后 `PᵀKP` 呈**对角块 K_ss + 子结构与其祖先的耦合块**结构（
`K_{13}, K_{17}, K_{37}` 这类），M 同构。全局 `K = diag(K_ii) + 耦合块
K_ij（j∈A_i，非祖先耦合恒为零）`。这是 AMLS 一切稀疏性的根基：
- 变换矩阵 `T^(i) = diag(I_<i, Φ_i, Ψ_ij, I_>i)`（[EAMLS] 式3）
- 完全变换 `T = ∏T^(i) = Ψ̂Φ`（式7-8）

**基由两类向量张成**（[SIM] 式4-5）：
- **子结构法向模态** Φ_i：解 `K_ii Φ_i = M_ii Φ_i Λ_i`
- **约束模态** Ψ_ij = `−(K_ii)^(−1) K_ij`（叶子用 K_ss，非叶子用变换后 K̃_ss）

### 2.2 剖分性质（[Gao08] 实测）

- **3D 问题 separator 大**：PE12K 顶层 separator 达 4937 DOF；separator 数量
  随 nlevels **指数增长** → 投影问题维度爆炸（这是 nlevels 不能无限大的原因）。
- 叶子规模：EAMLS 各例剖 236（impeller）/ 1971（femur）/ 2046（airplane）个子
  结构；[BL04] 车体 **12,068 个子结构、24 层**。
- [Gao08] nlevels 为运行时参数（实验 2~6 层）。

---

## 3. 子结构谱变换（核心数学，[PAPER]）

### 3.1 单层分块与块 LDLᵀ（[Gao08] §2.1）

单层划分后，K、M 按 [内部₁; 内部₂; interface] 分块（interface 远小于内部）。
对 K 做**块 LDLᵀ**：

- L 含块 `Kᵢ₃ Kᵢᵢ⁻¹`，D = diag(K₁₁, K₂₂, K̂₃₃)，Schur 补
  `K̂₃₃ = K₃₃ − Σ Kᵢ₃ᵀ Kᵢᵢ⁻¹ Kᵢ₃`
- 对 M 做同余变换得 (K̂, M̂)——**Craig-Bampton 形式**，特征值不变、
  特征向量 `x = L⁻ᵀ x̂`

### 3.2 局部特征问题（[Gao08] §2.3 / [EAMLS] Eq.(4)）

- 子结构 i（固定界面模式）：`Kᵢᵢ^(i−1) Φᵢ = Mᵢᵢ^(i−1) Φᵢ Λᵢ`
- 耦合（界面/separator）模式：解 `(K̂₃₃, M̂₃₃)`（[BL04] 记为
  `K̃_Γ Z_Γ = M̃_Γ Z_Γ Λ_Γ`，Γ = separator/interface）
- 变换阵 U 含 `−K_Ωi⁻¹ K_Ωi,Γ`，得 `UᵀKU = diag(K_Ω1, K_Ω2, K̃_Γ)`

### 3.3 变换矩阵与截断（[EAMLS] Eq.(3),(8)）

- 变换矩阵 `T = Πᵢ T⁽ⁱ⁾ = Ψ̂Φ`，其中 Φ = block diag(Φᵢ)（子结构正则模态），
  Ψ̂ 为**多层约束模态矩阵**（非对角块 `Ψᵢⱼ = −(Kᵢᵢ^(i−1))⁻¹ Kᵢⱼ^(i−1)`），
  块结构（子结构内部块对角、各层耦合到祖先）。
- 按截断频率把 Φ 分为 dominant/residual：Φ = [Φ_d, Φ_r]，**只用 T_d**。

### 3.4 截断准则（[PAPER] 两套）

1. **截止频率**（[Kim04] + [EAMLS] 实测）：cutoff "远高于最高激励频率的平方"；
   [EAMLS] 实测系数：取最高激励频率 ω_c 时
   **ω_b = 11ω_c（bottom-level）、ω_Γ = 16.5ω_c（interface）**——
   界面用更高截止频率，保证界面模态代表性。
2. **ρ 因子**（[Gao08] Eq.(18)）：`ρ_λ(ω) = |λ/(ω−λ)|`——子结构模式特征值
   越靠近目标 λ 越重要，据此截断（替代硬 cutoff 的自适应准则）。

> **[D 2026-10-09] Abaqus 官方语义**：AMS cutoff1=5 是乘 f_max 的**全局统一**频率
> 截断（无逐层放大），cutoff2=1.7/cutoff3=1.1 只用于根上缩聚求解的初始子空间。
> 上面 [EAMLS] 的 11×/16.5× 是该文献作者自己的分层选择，不是 Abaqus 参数语义。
> 详见 AMS官方文档解读.md。

### 3.5 缩聚后的全局问题（[Gao08] §3.3 / [Kim04]）

- 投影后 **K̃ = SᵀK̂S 是对角阵**，**M̃ = SᵀM̂S 与 M 有相同分块稀疏结构、
  对角块为对角阵**（子结构间通过界面耦合）。缩聚问题维度
  m = Σ各子结构保留模式数。
- 量级：[Kim04] 10M DOF → ~100,000 DOF 子空间；[BL04] 2.9M DOF 车体保留
  **m = 40,336**；[EAMLS] femur 缩聚到 4264、airplane 5492。

---

## 4. Phase4：缩聚后全局问题的求解（[PAPER]，[Kim04] 核心贡献）

缩聚后的特征问题虽然小（~10 万），但对高模态密度模型仍不可解（要 >1 万特征对）。
Kim 的蒸馏子空间（distilled subspace）方法：

1. **定义子树（subtree）**：子结构树中把若干相邻节点聚成子树（受"最大子树
   尺寸"约束，[Kim04] Ch.6 表 6.2/6.3 有调参）。
2. **解子树特征问题**：每个子树的缩聚问题求特征对，**再截断**（蒸馏截止
   频率 ω_D；同时截断 subtree 和 branch substructure 的特征空间）。
3. 把全局问题**投影到蒸馏子空间 D**（利用 K/M 稀疏性；比子结构模态子空间
   更小，初始子空间表示为**稀疏矩阵**）。
4. **一次 subspace iteration（"inverse iteration"）**恢复精度——好的起始
   子空间使一次迭代就够。
5. **Rayleigh-Ritz 分析**：在精化子空间上解小特征问题，得全局近似特征值；
   特征向量用**矩阵-矩阵乘法**恢复。

### 等价方案：SSI（special subspace iteration，[Gao08] §4）

Kaplan 蒸馏的等价实现：把 K̃ 中"不想要"对角元对应的行/列去掉得 (K̃_B, M̃_B)，
用 SIL 解之得 n_k 个 Ritz 向量作起点，然后做**块逆迭代** `X⁽ʲ⁾ =
K̃_B⁻¹ M̃_B · Y^(j−1)`，每个向量与其 Ritz 向量配对成 2 维子空间求最优逼近。

**实践指引**（[Gao08] §3.3）：**n_proj ≤ 5000 且所需特征对数占比大时，用稠密
求解器 DSYEVD 优于 SIL**——这正是本项目当前的路径（稠密 dsygvd），在小缩聚
问题上并不差；当缩聚问题更大 / 模态密度高时才需要蒸馏。

**误差来源**（[SIM]）：截断的残差法向模态 Φ_i^r（按 ω_cutoff² 保留 Φ_i^d）；
界面模态代表性（界面保留数不足是精度损失主因）。

---

## 5. 为什么 AMS 比 block Lanczos 快/省（[PAPER] 原文理由）

[Kim04] §1.1 明确列出（理解"为什么商业 AMS 快"的钥匙）：

1. **浮点运算少**：所有计算在小子结构模态子空间上进行，不是在整个 FE 自由度
   空间；对 K、M 只做**一次数值遍历**，不像 Lanczos 需要多次分解 + 反演 + 乘。
2. **内存带宽要求低**：Lanczos 算术/访存比低，依赖高带宽；AMS 是稠密小块
   GEMM 为主，缓存友好。
3. **数据传递量小几个数量级**：Lanczos 需把数万个特征向量（全 FE 空间表达）
   反复写盘/读盘（论文说**数十 TB I/O**）；AMS 不在全 FE 空间产生中间特征
   向量，恢复是最后一步。
4. **天然可并行**：不同子结构相互独立，可同时处理；越靠近根，单个子结构
   并行度越大。
5. **内存峰值低**：局部子结构问题逐个处理，稀疏直接法局部 fill-in 受控；
   全局缩聚问题小。

**Lanczos 固有缺陷**（[Kim04]）：高模态密度时 I/O 爆炸、并行难（并行 Lanczos
每核复制全模型 → 内存×核数）、算术/访存比低。

---

## 6. 并行化技术细节（[HM][FG][GPU]）

这是 Abaqus AMS 商业实现提速的关键（也是本项目的核心差距）：

### 6.1 树状结构的并行机会与瓶颈
- **模型变换是自底向上遍历消除树**：同一层节点间无依赖、可乱序并行 [GPU]。
- 叶子节点约占总节点一半、规模小、后效计算轻 [GPU][HM]。
- **瓶颈在树根部**：低层节点多而轻、高层节点少而重；二叉消除树根附近并行度
  骤减：[HM] 96 核仅 23x、[FG] 24 线程仅 7.93x——归因于树高层少量大节点
  （>1000 DOF，集中于根部）矩阵分解/局部特征求解在多线程下约 **5 线程即饱和** [FG]。

### 6.2 任务分解（[HM][FG] 的核心洞察）
- **分层任务分解**：低代价节点聚合成**粗粒度子树任务**（一线程执行，减调度
  开销；cut_level 启发式 `level(i)+log2(numThreads)+2~4`），高代价节点再拆
  **细粒度任务**。
- **细粒度任务拆法**（[FG] Table 1）：单子结构变换拆成 6 个任务
  A=Kii 分解、B=约束模态、C=特征求解、D=祖先刚度更新、E=祖先质量更新、
  F=后代质量更新。**关键洞察：parent 的 A 只依赖子节点的 D（刚度更新），
  特征值解 C 不再是阻塞依赖** → 打破父子同步屏障，线程不等空转。
- **依赖计数 + DAG 任务池**：任务完成递减依赖计数，归零进 ready 队列，
  线程动态取任务，无全局同步 [HM][FG]。**本项目已实现**（`solver.cpp` 的
  `DagRunner`、`gaussian_elimination.cpp` 的 N1/N3/N4/N5 + `subspace_condensation`
  的 N7–N10 进同一 DAG），方向与 [FG] 一致。

### 6.3 OpenMP/MPI 分工（[HM]）
- OpenMP：共享内存内的粗/细粒度任务（同 6.2）。
- MPI：**静态节点映射**按 FLOPs 估价子树负载，循环分配（负载阈值 1.1）；
  分 local nodes（无进程间通信）和 global nodes（跨进程根，通信密集）；
  4 类消息（刚度/质量更新发往父进程，约束模态/特征向量分量发往子进程）；
  非阻塞 MPI_Isend 掩盖传输延迟；回代时共享节点**本地重算优于广播**。

### 6.4 GPU 加速（[GPU]）
- GPU 承担：刚度分解、约束矩阵求解 `Kii⁻¹Kij`、祖先矩阵更新。
- CPU 保留：区间特征方程、后代质量更新（低并行度/逻辑复杂）。
- 叶节点 Pardiso（主机侧），非叶/根级稠密；小规模特征求解用 GPU 上厚重启
  Lanczos（因 K̄ 对角，逆易求）。
- **分层策略**：cutoff level `Lc = min(Ls, Lt+5)`，按平均节点规模 ≥400 DOF /
  `Lt=⌈log2 Nt⌉` 判；小节点传输时间>计算时间，下层留 CPU。
- 每节点一个 CUDA stream，异步传输与 kernel 重叠；零拷贝；90% 显存为
  "上设备"判据。
- 加速比：1×RTX 4090 + 16 CPU 线程比纯 CPU **>3x**（典型 3–5x），比 ARPACK
  **~10x**（68.5s vs 676.0s）。

### 6.5 对本项目的直接启示
1. **BLAS 并行策略**：本项目当前 `blas::setNumThreads(1)`（`solver.cpp`），
   BLAS 永久单线程、并行只走节点级线程池——理由是"BLAS 内部线程与 std::thread
   互踩"。但 [FG] 明确用**多线程 BLAS 三级并行**（"列数>2×线程数"动态切换），
   且列级并行。**可以重新评估**：给 GEMM 提供受控的多线程路径（限制 BLAS 线程
   数 ≤ 工作线程数，或仅在大块 GEMM 上开）。
2. **子树聚合**：cut_level 启发式（低代价节点聚合减少调度开销）——本项目
   `leafSize`/ND 已有树，但任务调度可能缺"聚合"层级。
3. **依赖拆细**：[FG] 把父节点 A 只依赖子节点 D、C 不再阻塞——本项目 DAG
   已做类似（N1 哨兵/updatesDone），可对照确认 C（N7 局部谱）是否也被正确
   从阻塞依赖中摘出。
4. **只算耦合非零列 + 稀疏/稠密选路**（CSR+SPMV vs GEMM）[GPU][FG]：本项目
   `subspace_condensation` 的 N8/N9 是否已对耦合块稀疏性分路？

---

## 7. 精度增强技术（[SIM][EAMLS][OOC]）

### 7.1 子空间迭代（AMLS-SIM，[SIM]）
- 迭代式 `Q^(k+1) = K⁻¹ M Q^(k)`（用变换后分块对角 K̃ 免分解原 K，前后向
  按树遍历实现）。
- **AMLS-SIMa**：Q⁰=V（AMLS 特征向量），对原 K 一次 LDLᵀ，每步解 KQ=MQ，
  迭代后再 Rayleigh-Ritz。
- **AMLS-SIMb**：利用变换后分块对角 K̃=UᵀKU，免分解原 K（式28-32），比
  SIMa 省 29%~58% 时间。
- 收敛率因子 `λ_i/λ_(nq+1)`；**1~2 步即显著改善低端谱**，把模态误差从
  >10⁰ 压到 10⁻³ 容差内。
- 收敛判据 `|1−λ_(c,i)^(k)/(x_c,i^(k)ᵀ x_c,i^(k))|^(1/2) ≤ ε_tol`；`n_q =
  max(2n_p, n_p+8)`。

### 7.2 Enhanced AMLS（[EAMLS]）
- 传统 EAMLS 对所有子结构补偿残差模态效应，需显式构造与全局矩阵同维的 Ψ̂，
  缩减矩阵变 100% 稠密 → **>100 万 DOF 直接 OOM**（股骨例）。
- 新 EAMLS：加 **interface subspace reduction**（解 `K̄_Γ Ξ = M̄_Γ Ξ Θ`），
  只用**底层子结构残差柔度**修正：`F̃_i^rb = K_ii⁻¹ − Φ_i^d(Λ_i^d)⁻¹ Φ_i^dᵀ`。
- 只补质量矩阵（K̃_e = K̃_d）；利用质量正交性把缩减质量矩阵密度从 100% 降到
  44.71%（叶轮）、刚度保持 0.07%。
- 同精度下模型可缩小约 **4 倍**（23872→5492）。

### 7.3 残差柔度（[Kim04] App.A / [EAMLS]）
- 频响分析用残差柔度 `F̃_rs ≈ Φ̃_r K̃_r⁻¹ Φ̃_rᵀ`（忽略 O(ω²) 以上项）与
  block Gram-Schmidt 正交化。
- [Kim04]：AMLS 频响比 shifted block Lanczos **快至 6.4 倍**、内存更小、
  数据量少一个数量级。

### 7.4 Out-of-core（[OOC]）
- 写盘：子结构特征向量 Φ_i 与约束模态 Ψ_ij（当前节点算完后至 back
  transformation 才复用）、子结构对祖先的更新矩阵。每线程单独二进制文件 +
  记录文件名/偏移；更新矩阵每子结构单独文件顺序写读删。
- **节点三分**：叶子（全程稀疏，不取更新数据）、次底部（不从磁盘取）、其余
  （组装原矩阵+取盘+更新三步）。
- 结果：4 模型（2.33M~8.68M DOF、7871~29171 子结构）**峰值内存降 34%~53%**
  （bat_867：48.6→23.0 GB），时间仅增 ~25~40%。
- 本项目**已有雏形**：`gaussian_elimination.cpp` 的 `releaseNodeData`（diag/
  耦合列/Mtemp 释放，u 列按需保留）+ 层间流水线，方向同 [OOC]，但未做磁盘
  换出。

---

## 8. 论文实测基准数字（对比锚点）

把这些数字作为我们"合理期望"的标尺（Abaqus AMS 的商业实现应不弱于论文值）：

| 来源 | 模型 / n | AMLS 结果 | 基准对比 |
|---|---|---|---|
| [Gao08] | DDS6_FINE 65,730 DOF, nev=100 | 365s（phase1 288 + phase2 77） | SIL 407s |
| [Gao08] | PE3K 9,000 DOF, nev=500, 6 层 | 483s | SIL 956s（约快 2 倍） |
| [Gao08] | PE12K 36,000 DOF, nev=1000 | 15,511s（不全选分隔模态 >36,000s） | SIL 18,000s |
| [Gao08] | DDS6_FINE 内存 | 最优 263MB（semiimplicit L 省 50%） | SIL 308MB |
| [BL04] | 2.9M DOF 车体 | **12,068 子结构 / 24 层** / 截止 4000Hz（关心 400Hz）/ 保留 40,336 模态 | 耗时约为 Nastran block Lanczos（Cray T90）**一半**；频差 0–50Hz <3e-5，300–400Hz <3e-3 |
| [EAMLS] | femur 610,260 DOF / 1971 子结构 | AMLS 59.6s（递归变换占 81.8%）；新 EAMLS 61.6s | 原 EAMLS 374.9s |
| [EAMLS] | airplane 2,025,180 DOF / 2046 子结构 | AMLS 3317s；新 EAMLS 3604s | 原 EAMLS OOM（>32GB） |
| [EAMLS] | 精度/内存 | 同精度缩 4 倍模型（23872→5492）；缩聚 M̃ 密度 44.71%、K̃ 0.07% | 峰值内存 14.85GB |
| [HM] | BIW 830,987 DOF | 单核 328.5s → 96 核 14.4s（23x） | 96 核比 RAMLS 快 25% |
| [FG] | 车门 1.225M DOF / BIW 9.4M DOF | 24 线程 7.93x；1.225M → 53.1s | 弱扩展 1→16 线程保持 ~17–24s |
| [GPU] | 0.63M–6.96M DOF, 树深 12–16 | HAMLS>PAMLS 3x | 比 ARPACK 快 ~10x |

**关键含义**：
- [BL04] 车体例子（2.9M DOF / 12,068 子结构 / 24 层 / 保留 4 万模态）是我们
  1M DOF 悬臂块最直接的对标（挂柱：12k 子结构、24 层、截止频率=10×关心频率）。
- [EAMLS] femur 说明**递归变换占 81.8%**——和本项目 `condense` 占 phase2 大头
  的方向一致；优化变换阶段 = 优化大头。
- [Gao08] 当 **nev < 100 时 AMLS 无优势**（缩聚成本高于 SIL）——小模态数场景
  不算 AMLS 的舞台。

## 9. 本项目 AMLS 的差距定位与优化方向（[INF] + 本实现）

本项目 `src/amls/` 已实现：`solver.cpp` 四阶段流水线（剖分 → 高斯消去 →
子空间缩聚 → 降阶解 → 恢复）、`gaussian_elimination.cpp`（N1/N3/N4/N5 任务化 +
DAG）、`subspace_condensation.cpp`（N7 局部谱 / N8 种子 / N9 折叠 / N10 传播）、
`separator_tree.hpp`（分割树）、ND/bisect 剖分（`metis_partitioner.cpp`）。

### 8.1 直接差距（按影响排序）

1. **降阶后直接稠密全解，无蒸馏子空间（Phase4 distilled subspace）**
   我们 `solver.cpp` step3 对缩聚后的 (K̃, M̃) 直接 `dense.solveIndex/
   solveAll`（稠密 dsygvd）。AMS/[Kim04] 的 Phase4 用**蒸馏子空间 + 一次子空间
   迭代 + Ritz**，把全局小问题进一步缩小，且能处理高模态密度。当 `nev` 大 /
   f_user 高时我们可能在此成为瓶颈。→ 实现 distilled subspace（定义子树→子树
   特征问题→截断→投影→一次 SIM→Ritz）。

2. **BLAS 线程策略过保守**
   `blas::setNumThreads(1)` 全局单线程 + 节点级线程池。论文明确多线程 BLAS
   三级并行有效（[FG]）。→ 重新评估：大块 GEMM（N8/N9/backtransform）开
   受控多线程 BLAS，避免与线程池互踩（限制 BLAS 线程数 / 仅大块开）。

3. **模态截断准则差异**
   我们 `lambdaCut = η·λ̂`（Ritz 上界×因子，`solver.cpp`）；AMS/[EAMLS] 用
   **截止频率**（ω_cut，且界面用更高 ω_Γ）。→ 对物理问题（有 f_user），界面
   子结构的更高截断准则可提升精度/效率平衡；可对比 η 与 ω_cut 的等价性。

4. **并行粒度 vs [FG] 6 任务拆解**
   我们 DAG 已拆 N1/N3/N4/N5/N7–N10 + 哨兵，但 [FG] 明确"父节点 A（Kii 分解）
   只依赖子节点 D（刚度更新），特征值解 C 不阻塞"。→ 核对我们的 N7（局部谱）
   是否被放在阻塞依赖里；若是，照 [FG] 摘出，进一步减少空等。

5. **耦合块稀疏分路**
   [GPU][FG] 只算耦合矩阵非零列，CSR+SPMV vs GEMM 选路。→ 核对
   `subspace_condensation` 的 N8/N9 对 Mtemp 耦合列的零结构是否利用（`colcomp`）。

6. **semiimplicit L 表示（[Gao08] §3.2，省内存 50%）**
   [Gao08] 从不显式存储含大量 fill-in 的 M̂ 与 L：L 用 **semiimplicit 表示**
   （只存 separator 节点对应的非对角块，叶节点块用时重算），据此省 50% 内存
   （DDS6_FINE 最优 263MB vs SIL 308MB，代价仅 +10~15% 耗时）。
   → 核对本项目 `gaussian_elimination.cpp` 对 L（约束模态 Ψ）的存储：是否
   全部显式持有？若是，可试半隐式方案（重算 vs 存储的平衡）。

7. **界面双截断（[EAMLS]）**：界面子结构截断频率取 ω_Γ = 16.5ω_c ≫ 子结构
   ω_b = 11ω_c。→ 此处"界面"即我们的 separator 节点；对 separator 节点用更
   高保留率，改善界面模态代表性（[SIM] 说界面保留数不足是精度损失主因）。

8. **ρ 因子自适应截断（[Gao08] Eq.(18)）**：`ρ_λ(ω) = |λ/(ω−λ)|` 衡量子结构
   模式对目标特征值的贡献，比硬 cutoff 更优。→ 可作为 `lambdaCut` 的改进
   准则（现为 η·λ̂ 单阈值）。

### 8.2 远期（与 Abaqus 重装后实测对比验证）

9. **Out-of-core 换出**：>100 万 DOF 时特征向量/更新矩阵写盘（[OOC] 峰值内存
   −34~53%）。本项目 `releaseNodeData` 已有释放，可补磁盘换出。
10. **残差柔度增强**（频响场景）：补 `F̃_rs`，提升频响精度（[Kim04] App.A）。
11. **多核扩展性**：[HM][FG] 显示树根部是并行瓶颈（>1000 DOF 大节点 5 线程
    饱和）。→ 对树根部大节点内部做任务拆分（不止节点级），或接受 sublinear。

### 8.3 待实测验证（Abaqus 装好后）

- AMS 的剖分规模（多少子结构/多少层）——从 `.msg`/`.dat` 或行为反推。
- AMS 各阶段耗时占比——从 `.dat` timing / 墙钟分段。
- AMS 内存曲线——`.dat` memory estimate + 进程峰值。
- AMS 是否用蒸馏子空间 / SIM——黑盒难直接看，但可通过"增大 nev 时耗时/内存
  的增长率"反推（若为蒸馏+SIM，增长应比稠密全解平缓）。
- 对比协议见 `docs/abaqus_ams/运行期观测清单.md`。

---

## 9. 待补充章节

- [ ] Abaqus 官方文档细节：在线文档需 3DS 登录、本机 CAE-only 安装无离线文档，
      在线抓取子代理亦因 API 额度失败——**等 Abaqus 完整重装后从本地文档或
      实测补充 [DOC] 标注条目**。已从本机 `abaqus_v6.env` 实证一条：
      `freqEigenAMS: {'ams': ON}`（AMS 走独立 JLE 任务路径，见
      `运行期观测清单.md` §4b）。
- [ ] [Kim04] Ch.2/4/6 原文（当前提取文本止于第 24 页）：Trim-Body/Full-Vehicle
      /8.4M DOF 的精确耗时、subtree 上限与 ω_D 的具体取值表。

---
*关联文档*：`docs/abaqus_ams/运行期观测清单.md`（Abaqus 重装后的实测采集清单）、
`docs/AMLS_实施计划.md`（本项目实现）、`docs/设计文档.md`。
