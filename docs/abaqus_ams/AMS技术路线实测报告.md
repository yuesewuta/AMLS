# Abaqus AMS 技术路线实测报告（120 Hz，探针探测）

> 2026-10-09。模型 20×20×20 悬臂块（不变）：9261 节点 / 8000 C3D8 /
> 27783 方程 / 26460 自由 DOF / 钢 E=210e9 ν=0.3 ρ=7800。
> 求解参数：`*FREQUENCY, EIGENSOLVER=AMS`，数据行 `20, 0.0, 120.0`，cpus=4。
> 本文每条结论都标注**证据来源**：日志字段、探针 dump 文件、DLL 静态字符串、官方文档或论文。带 [INF] 的才是推断。

---

## 1. 探测方法

### 1.1 静态探针（不运行求解器）

对 `ABQSMAStaCore.dll`（72 MB，Standard 核心求解器）做二进制字符串扫描，
提取出 320 个 `ABA_*` 内部环境变量和全部 AMS 类/函数名。这些是**编译器保留的
明文字符串**（Abaqus 求解器符号没有被完全剥离），等价于把黑盒开了一条缝。

关键发现（全部有字符串证据）：
- AMS 内部代号 **PAMS**（Parallel AMS）：`PAMS DRIVER FOR STRUCTURAL DOMAIN-START/END`、
  `PAMS EIGENSOLVER-START/END`、`total pams driver`。
- 符号分解类 `pme_AmsSymbolicFactor`，暴露消去树结构：
  `ChildrenArray`/`ChildrenMap`（孩子表）、`SuperNodeParent`（父指针）、
  `SuperNodeSizes`/`FrontNodes`（超节点大小/前端节点）、`NumberOfSuperNodes`、
  `LeafCount`/`LeafNode`（叶子）。
- 直接求解器符号分解 `SMAEqsDirSymbolicFactor`：`SuperNodeSize`、`SuperNodeParents`、
  `SuperNodeFrontSize`、`NodalCliqueSizes`、`NumberOfSolverNodes`，
  以及 verbose 提示 `Supernode %d cliqueSize %d frontSize %d parent %d topEquationId %d`。
- 控制开关 30 余个 `ABA_PAMS_*`：`EXPORT_CONDENSEDOPERATORS`（缩聚算子）、
  `EXPORT_SOLVER_DATA`（求解器稀疏算子）、`EXPORT_STATICMODES`（静态模态）、
  `EXPORT_RESIDUALVECTORS`（剩余模态）、`EXPORT_GLOMODES`（全局模态）、
  `CHECKRESIDUAL`、`ORTHOGONALITY_CHECK`、`FULL_RETAINED`、`GLOBALSHIFT`、`LAST_CUTOFF`、
  `NUMGPUS`、`NODUMP`、`NOSUBSTRUCTUREGENERATE`、`SOLVER_TEST`。
- 内部数据文件名：`%s_amsData.stiff.%d`、`%s_amsData.mass.%d`、`%s_amsData.symbFct`、
  `%s_amsData.mapRes`、`%s_amsData.mapSel`、`%s_amsData.soln.%d`。
- 迭代求解器（EQSITR，多层聚合 AMG）系列变量：`ABA_EQSITR_MAX_NUM_LEVELS`、
  `MAX_COARSE_SIZE`、`MAX_AGGREGATE_SIZE`、`CYCLE_TYPE`、`SMOOTHER_TYPE` 等
  —— 说明 AMS 内部子问题/局部问题与迭代求解器共享技术栈。
- METIS 相关：`ABA_SST_USE_METIS`、`ABA_FORCE_PARMETIS`、`ABA_DBG_PARMETIS`、
  `ABA_PARMETIS_*`（EXPORT_DATA/IMPORT/SEPS/SIZES/CHECK_GRAPH/DGRAPH）、
  `ABA_METIS_MEMORY`、`ABA_METIS_WEIGHT`。**证明剖分器是 METIS/ParMETIS 系**。
- 求解流程控制：`ABA_EXIT_AFTER_FIRST_SOLVER_PASS`（跑完第一遍退出）、
  `ABA_SKIP_PF_CALCULATION_FOR_AMS`、`ABA_PAMS_NOSUBSTRUCTUREGENERATE`。

证据文件：[ABQSMAStaCore.dll 字符串扫描]（本报告附录 1）。

### 1.2 动态探针（真实运行）

在本目录放 `abaqus_v6.env`（用户级，只影响本目录 job），注入：
`ABA_SYMBOLIC_FACTOR_VERBOSE=1`、`ABA_PAMS_CHECKRESIDUAL=1`、
`ABA_PAMS_ORTHOGONALITY_CHECK=1`、`ABA_PAMS_EXPORT_CONDENSEDOPERATORS=1`、
`ABA_PAMS_EXPORT_SOLVER_DATA=1`、`ABA_SST_USE_METIS=1`、
`ABA_DEBUG_DECOMPOSITION=1`、`ABA_ITERATIVE_SOLVER_VERBOSE=1`、
`ABA_EQSITR_SETUP_VERBOSE=1`。

运行 `ABQLauncher job=cant_20_freqonly cpus=4`。结果：作业 COMPLETED，
控制台出现 `Dumped out operators in solver order`（EXPORT_SOLVER_DATA 生效），
工作目录新增 6 个探针 dump 文件。**这些文件不是 Abaqus 正常产物，是探针强制导出的内部数据**。

## 2. 实测证据链（逐条）

### E1. 算子导出与规模（EXPORT_SOLVER_DATA）

| 文件 | 大小 | 内容 |
|---|---|---|
| `cant_20_freqonly_stf` | 13,499,960 B | 重排后的全局刚度算子（CSR 二进制） |
| `cant_20_freqonly_fca.stf` | 13,499,960 B | 因子副本（与 stf 同尺寸） |
| `cant_20_freqonly_mas` | 917,048 B | 全局质量算子（CSR 二进制） |
| `cant_20_freqonly.orth_K` / `.orth_M` | 3,589 B | 模态基下的 K/M（正交性检查） |
| `cant_20_freqonly.times` | 3,265 B | 全阶段计时表（4 线程） |

`stf` 头部解码（uint32）：`[0]=27783`（行数=方程数）、`[1]=1,007,523`（非零总数）、
`[3:]=` 行偏移数组。**实测：K 有 1,007,523 个非零，每行平均 36.3 个非零**。
对比：C3D8 六面体网格 27 节点邻域 × 3 DOF ≈ 36 非零/行，量级吻合。

### E2. 方程重排 = 消去树序（stf 行偏移）

行非零数分布呈现**强阶梯/梯形**：
- 22,599 行非零数递减，5,183 行递增，0 行持平。
- 3,756 个"非零数突变 >10"的边界（supernode 块边界）。
- 前 40 行非零数序列：`17, 80,79,78,77,76,75, 53,52,51,50,49,48, 71,70,69,68,67,66, ...`
  即每块内行非零数从大到小递减 1，块间跳变——正是**稀疏 Cholesky 消去树超节点
  （supernode）的梯形/阶梯结构**：块内行共享同一列模式，行间非零差 1。

> 证据：stf 行偏移数组的统计。结论：Abaqus AMS 先把方程重排为**消去树超节点序**，
> 这与官方文档"multi-level supernode elimination tree"一致。

### E3. 阶段划分（times 表）

`times` 文件是 AMS 求解器内部计时表（4 线程列）。核心阶段及其语义：

| 阶段名 | 含义（结合 AMS 算法） | 实测总耗时(4线程和, s) |
|---|---|---|
| `k_clique` / `m_clique` | K/M 超节点"团"处理（clique = 超节点内稠密块） | 0.055 / 0.028 |
| `k_factor` | 全局 K 稀疏 Cholesky 分解 | 0.177 |
| `sturm_factor` + `sturm_check` | **Sturm 序列检查**（每个子结构/超节点局部特征问题按移位计数特征值） | 0.143 + 0.001 |
| `substr_sygst` | 子结构广义特征问题 → 标准形（LAPACK sygst，Cholesky 约化） | 0.139 |
| `substr_sytrd` | 对称矩阵三对角化（LAPACK sytrd） | 0.178 |
| `substr_stev` | 三对角特征值求解（LAPACK stev） | 0.635 |
| `substr_ormtr` / `substr_trsm` | 特征向量回代（LAPACK ormtr）+ 三角求解 | 0.027 / 0.015 |
| `substr_extra` / `substr_clean` | 子结构提取/清理 | 0.002 |
| `k_updext` / `k_bwd_psi` | K 沿树向上扩展 / **Ψ 回代**（AMLS 核心传播） | 0.283 / 0.277 |
| `m_updext` / `m_stackext` | M 沿树向上扩展（栈式扩展） | 0.256 / 0.581 |
| `m_fwdred` / `m_bwdred` | M 前向缩减 / 后向缩减 | 0.014 / 0.006 |
| `m_bwdupdonly` / `m_bwdupdfwd` | M 后向更新（仅/含前向） | 0.027 |
| `rhs_*` / `destroy_aux` / `*_cleanup` | 右端项与清理 | <0.01 |
| `sd_*` / `vd_*`（结构/粘性阻尼） | **全 0** —— 无阻尼模型，阻尼路径未进入 | 0 |

> 证据：`times` 文件。这些阶段名与 AMLS 论文（Yin 2013 / Bennighof 2004）的
> 算法分解高度一致：**每个子结构做稠密特征问题（sygst→sytrd→stev→ormtr），
> 然后沿消去树做 Ψ 传播与 K/M 缩减/扩展**。这是 Abaqus AMS = 经典 AMLS 的直接证据。

### E4. 模态正交性与广义质量（ORTHOGONALITY_CHECK）

`orth_K`/`orth_M` 是**最终保留模态基下的 K/M 投影**（12×12，上三角）：
- `orth_K` 对角 = 特征值：3.0066e4, 3.0068e4, 5.5392e4, 1.7220e5, ...（换算频率
  27.597, 27.598, 37.458, 66.044, 72.992, 73.001, 89.961, 106.380, 111.018,
  113.670, 113.711, 113.827 Hz）—— 与 `.dat` 特征值输出**逐位一致**。
- `orth_K` 非对角最大 4.2e-9（≈0）：模态基下 K 完全对角化。
- `orth_M` 对角 = 1.000000（广义质量归一），非对角最大 8.3e-15（≈0）：
  模态基下 M = I。

> 证据：orth 文件与 dat 对照。结论：**AMS 返回的模态满足 K-正交、M-正交、广义质量
> 归一（mass-normalized eigenvectors）**——与 dat "THE EIGENVECTORS ARE SCALED
> SO THAT THE GENERALIZED MASS IN EACH VECTOR IS UNITY" 一致。

### E5. 频率截断 vs 数量上限（双重机制）

`*FREQUENCY, EIGENSOLVER=AMS` 数据行 `20, 0.0, 120.0`（模态数, 最低, 最高频率）：
- dat/msg 记录：`NUMBER OF EIGENVALUES 20`、`HIGHEST FREQUENCY OF INTEREST 120.00`、
  `LOWEST 0.0`。
- **实际返回 12 阶**（[0,120] Hz 内只有 12 个模态），AMS 自动截断到频率范围。
- 1000 Hz 时返回 20 阶但警告"TOTAL 18206 EIGENMODES ARE COMPUTED BELOW THE MAXIMUM
  FREQUENCY... 18186 TRUNCATED"：**AMS 在 [0,fmax] 内算全部模态，再按请求数量截断**。
- dat 警告：同时给数量+频率上限可能性能退化（"IF THE NUMBER OF EIGENVALUES BELOW
  THE MAXIMUM FREQUENCY IS MUCH LARGER THAN THE NUMBER TO BE CALCULATED"）。
- AMS 三参数：FIRST/SECOND/THIRD AMS PARAMETER = 5.0 / 1.7 / 1.1（cutoff 相关倍率）。
  相关字符串：`AMS cutoff1/2/3`、`cutoffdist`、`too short cutoff distance. Use
  environment variables...`、`ABA_PAMS_LAST_CUTOFF`。

> 证据：dat/msg 字段 + StaCore 字符串。结论：AMS 按**频率窗口**算模态（频率截断），
> 请求数量只作返回上限。

### E6. 求解器技术栈（DLL 静态）

- `standard.exe`（3.2MB）是**启动器**：加载 `ABQSMAStaCore.dll`（72MB 主求解器）、
  `ABQSMAMtrAbqStdModule.dll`（矩阵/标准模块）、`ABQSMAEqsDirSolverMod.dll`
  （直接求解器：超节点 Cholesky）、`ABQSMAEqsItrSolverMod.dll`（迭代求解器 AMG）、
  `ABQSMAEqsConstraintSolverMod.dll`（约束）。
- `ABQSMAStaCore.dll` 内含字符串：`Number of substructure dynamic modes = %d`、
  `Number of substructure eigen modes = %d`、`AMS-based Substr Generate`、
  `=== AMS-based Substr Generate (Structural Domain) ===`、`*Full Dynamic Modes
  Recovery`/`*Selective Dynamic Modes Recovery`、`NUMBER OF INTERFACE EQUATIONS
  (DDM SOLVER) %u`、`Number of AMS selective recovery dofs = %d`、`=== Number of
  residual modes: %8d ===`、`REAL EIGENVALUE PROBLEM WAS SOLVED USING THE
  SYMMETRIZED SUBSTRUCTURE STIFFNESS MATRIX`。
- 迭代求解器模块字符串：`Eigenvalues:`、`INDEFSPARSE`/`STRONGSPARSE`/`WEAKSPARSE`、
  `sparse.%d.%s` dump、AMG 层级（`SMAEqsItrAMGLevels`、`GetSmoother`）。

> 证据：DLL 导入表 + 字符串。结论：**AMS 主逻辑在 ABQSMAStaCore.dll，
> 与直接/迭代稀疏求解器、METIS/ParMETIS 剖分器、LAPACK 稠密特征求解器集成**。

## 3. 技术路线综合（可确认部分）

1. **方程重排**：K 被重排成超节点消去树序（E2），对应官方"multi-level supernode
   elimination tree"。
2. **剖分器**：METIS/ParMETIS 系（`ABA_SST_USE_METIS`、`ABA_FORCE_PARMETIS`、
   `ABA_PARMETIS_*` 字符串），且 `ABA_DEBUG_DECOMPOSITION`/`ABA_DBG_PARMETIS`
   存在调试入口。 [INF：具体调 METIS 的哪个 API、权重策略未观测]
3. **子结构缩减**：每个超节点做稠密特征问题（E3：sygst/sytrd/stev/ormtr），
   即 Craig–Bampton/Component Mode 缩减的局部模态提取；`substr_*` 阶段存在。
   [D 2026-10-09：官方文档确认局部截断 = AMS cutoff1×f_max 全局统一频率乘子
   （默认 5），见 AMS官方文档解读.md；此前"逐层 cutoff 推导"推断作废]
4. **沿树传播**：K/M 前向缩减 + 后向扩展 + Ψ 回代（E3：k_bwd_psi/m_fwdred/
   m_bwdred/m_updext/m_stackext），与 AMLS 论文一致。
5. **全局问题**：缩聚后的小规模广义特征问题求解（模态基），数量 = 频率窗口内
   全部模态（E5），由 Sturm 检查保证不漏根（E3：sturm_factor/sturm_check）。
6. **模态后处理**：K/M 正交化验证 + 广义质量归一（E4），选择性/全量恢复
   （`*Selective Dynamic Modes Recovery` 字符串 + `ABA_PAMS_EXPORT_STATICMODES`）。
7. **无阻尼路径**：sd_*/vd_* 阶段全 0（E3），阻尼不参与。

## 4. 仍未知/需更深入探测

- **实际超节点树形**：supernode 数、每 supernode 的 DOF 数、树深。已知
  `MAXIMUM SIZE OF A SUBSTRUCTURE 4096`（默认参数），`ABA_EQSITR_MAX_NUM_LEVELS`
  等只属迭代求解器。要拿树本身需触发 `pme_AmsSymbolicFactor::Dump` 或
  `SMAEqsDirSymbolicFactor::Verbose`（`ABA_SYMBOLIC_FACTOR_VERBOSE` 未在 dat/log
  可见，输出可能进 stderr 被吞）。
- **局部截断阈值**：每个超节点保留模态数的具体公式（`ABA_PAMS_LAST_CUTOFF`、
  `AMS cutoff1/2/3` 存在，未观测到数值）。
- **METIS 参数**：权重/平衡因子（`ABA_METIS_WEIGHT`、`ABA_PARMETIS_NO_WEIGHTS`）。
- **内存策略**：dat 明说 AMS 不做 I/O 最小化估算（"MEMORY TO MINIMIZE I/O ... NOT
  ESTIMATED DUE TO THE NATURE OF THE AMS REDUCTION TECHNOLOGY"）。

## 5. 复现清单

```bash
cd /z/codeSpace/AMLS/bench_abaqus
# 1. 生成模型 (freq-only, 120Hz)
python /z/codeSpace/AMLS/tools/gen_abaqus_cantilever.py --nx 20 --ny 20 --nz 20 \
    --nev 20 --fmax 120 -o cant_20_freqonly.inp
# 截去 matgen 步 (见 E 文件生成说明)
# 2. 放探针 env (已存在 abaqus_v6.env)
# 3. 运行
/d/SIMULIA/EstProducts/2020/win_b64/code/bin/ABQLauncher.exe job=cant_20_freqonly cpus=4
# 4. 收集
#    cant_20_freqonly_stf/_fca.stf/_mas  : 求解器算子 (CSR)
#    cant_20_freqonly.orth_K/_M          : 模态正交性
#    cant_20_freqonly.times              : 阶段计时
#    cant_20_freqonly.dat/.msg           : 参考解 + 警告
```

## 附录 1：ABQSMAStaCore.dll 关键字符串（原始证据）

见本报告正文；完整 320 个 ABA_* 变量清单可通过二进制扫描复现：

```python
import re
d = open(r'D:\SIMULIA\EstProducts\2020\win_b64\code\bin\ABQSMAStaCore.dll','rb').read()
s = [x.decode(errors='replace') for x in re.findall(rb'[\x20-\x7e]{6,}', d)]
vars_ = sorted(set(x for x in s if re.search(r'ABA_[A-Z0-9_]{6,}', x)))
print(len(vars_), '\n'.join(vars_))
```
