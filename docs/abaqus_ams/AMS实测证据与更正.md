# Abaqus AMS 实测证据与更正

本记录以 Abaqus 2020 的本机输出为依据，区分日志实测、官方算法说明与尚未确认的实现细节。更新于 2026-10-09（运行日志时间）。

## 1. 撤回的结论

此前回答中以下说法没有来源或实测支持，应撤回：

- Abaqus 按几何、材料或“弱耦合面”划分。
- 实际叶子为 32–64 个节点。
- 每个分隔器只有 2–4 个节点。
- 3D 模型必然形成八叉树，cant_20 的树深为 7–9 层。
- 把论文中的 METIS 或二叉树实现直接等同于 Abaqus 2020。

有限元节点、自由度、树节点（子结构/超节点）是不同概念。树的分支数不能用来表示分隔器矩阵维度。

## 2. 本机确实观测到什么

`bench_abaqus/cant_20.msg`：

- 第 19 行：实际使用 AMS eigensolver。
- 第 25 行：最高关注频率 120 Hz。
- 第 29–31 行：三个 AMS 参数为 5.0、1.7、1.1。
- 第 32 行：`MAXIMUM SIZE OF A SUBSTRUCTURE 4096`。
- 第 66 行：`NUMBER OF EQUATIONS: 27783`。
- 第 94 行：矩阵生成阶段标为 `CONSISTENT MASS MATRIX`。
- 第 100–101 行：文本输出 `cant_20_STIF2.mtx` 和 `cant_20_MASS2.mtx`。
- 第 140–141 行：CPU 10.7 秒、墙钟 97 秒，包含频率分析与矩阵生成/文本输出两个分析步。

4096 出现在求解设置中，**不是叶子尺寸统计**。现有日志没有说明该数的单位是物理节点、自由度还是其他内部尺寸，也没有提供实际叶子块尺寸分布。不能宣称“叶子为 4096 DOF”。

线程的 ELEMENT LOAD BALANCE 是单元操作的线程负载，不能据此恢复 AMS 子结构树。`REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT` 也不足以确定剖分器。

本机 `.dat/.msg` 没有报告子结构数量、separator 尺寸、树分支数、树深、局部保留模态数或每个内部阶段的时间。内存估算也不能当作测量得到的峰值工作集。

## 3. 官方文档能支持的技术路线

来源：[Natural frequency extraction — Controlling the AMS eigensolver](https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-freqextraction.htm)。这是公开的 2017 文档镜像，其内容属于算法说明，不是当前 2020 算例的内部树输出。

Reduction phase 原文：

> The approach combines a sparse factorization based on a multi-level supernode elimination tree and a local eigensolution at each supernode.

> Starting from the lowest level supernodes, we use a Craig-Bampton substructure reduction technique to successively reduce the size of the system as we progress upward in the elimination tree.

> At each supernode a local eigensolution is obtained based on fixing the degrees of freedom connected to the next higher level supernode…

可据此描述：AMS 沿多层超节点消除树从底层向上，结合稀疏分解与各超节点的局部特征求解，采用 Craig–Bampton 子结构缩减；局部求解时固定连接到上层超节点的自由度。

这不是四叉/八叉空间网格树的证据。文档没有在这些段落中披露树如何生成、分支数、METIS/其他剖分器或实际块维度。

通用 AMLS 的代数图剖分有论文依据，但应保留实现边界：

- `papers/Improving_eigenpairs_of_automated_multilevel_substructuring_with_subspace_iterations.md` 第 45 行：substructuring 可以是 purely algebraic，并可将 CHACO/METIS 用于矩阵非零模式对应的无向图。
- 同文第 47 行：二叉划分是示例，公式也可推广到非二叉划分。
- `papers/A_hybrid_OpenMPMPI_parallel_automated_multilevel_substructuring.md` 第 113 行：separator 被作为独立子结构处理。
- GPU AMLS 论文采用 METIS 5.1.0 和二叉树；这些是论文作者实现，不能当作 Abaqus 实测。

## 4. 10 Hz 复测

为防止覆盖 120 Hz 成功结果，以独立作业名运行：

```text
cant_20_10hz
*FREQUENCY, EIGENSOLVER=AMS
20, 0.0, 10.0
```

网格为 20×20×20，9261 节点、8000 C3D8 单元，总 DOF 27783；固支面 441 节点，去除约束后为 26460 个自由 DOF。材料和几何保持不变。

`bench_abaqus/cant_20_10hz.msg`：

- 第 25 行：`HIGHEST FREQUENCY OF INTEREST 10.000`。
- 第 32 行：最大子结构尺寸参数仍为 4096。
- 第 66 行：方程数 27783。
- 第 70 行：`***ERROR: AMS EIGENSOLVER HAS FOUND NO EIGENVALUES.`
- 第 95–96 行：CPU 5.3 秒、内部墙钟 3 秒。

Launcher 记录的 Standard 阶段为 00:02:33 至 00:02:53（约 20 秒），整个提交/预处理/退出流程约 35 秒。内部摘要与外部阶段时间口径不同，不混用。

**结果是错误退出，而不是成功求出 20 阶。**没有进入后续矩阵生成步。

原因有成功求解结果支持：`cant_20.dat` 的 120 Hz 频率表最低频率为 27.597 Hz，表中实际返回 12 阶：

```text
27.597, 27.598, 37.458, 66.044, 72.992, 73.001,
89.961, 106.38, 111.02, 113.67, 113.71, 113.83 Hz
```

因此该模型在 0–10 Hz 内没有模态；降低频率上限不会改变模型本身的固有频率。此前“120 Hz 大约 300 个模态”的估算与输出不符，应撤回。1000 Hz 运行曾报告计算 18206 个模态后截掉 18186 个，只保留请求的 20 个；该现象不能推广成所有运行必定计算相同数量或证明各内部阶段耗时。

若后续测试要求在 10 Hz 内成功获得模态，需要明确改变几何尺度或材料参数，这会改变 K/M 与特征值问题。本次未作此改变。

## 5. 可复用输入和当前边界

生成工具已补全 `--fmax` 命令行参数，默认 10 Hz：

```text
python tools/gen_abaqus_cantilever.py --nx 20 --ny 20 --nz 20 --nev 20 --fmax 10 -o bench_abaqus/cant_20_10hz.inp
```

本机已跑通的矩阵生成写法为：

```text
*STEP, NAME=matgen, PERTURBATION
*MATRIX GENERATE, STIFFNESS, MASS
*MATRIX OUTPUT, STIFFNESS, MASS, FORMAT=COORDINATE
*END STEP
```

`NUMEIGEN`、`GLOBAL`、`FORMAT=COORDS` 等此前无效写法不能称为已证实的 2020/2025 版本差异。输出名数字对应分析步编号；这里矩阵生成是第 2 步。

现在能回答的树问题只有：官方描述为多层超节点消除树，本机设置打印最大子结构尺寸参数 4096。实际叶子大小、separator 大小、树形和树深仍未知，不能从现有日志还原。
