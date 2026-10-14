# AMS cutoff1/2/3 官方文档解读（Abaqus 2020 本地文档实测摘录）

> 证据级别：**[D] 官方文档**（本地安装介质全文档检索，非网络转载、非推断）。
> 检索位置：`D:\SIMULIA\DS SIMULIA Suite 2020 Win64\5\SIMULIA_Documentation\AllOS\1\CAFS\DOC\English\`
> - `SIMACAEDoc.zip` 内 3DX 内容寻址页（SHA 文件名，无标题，按正文全文检索定位）：
>   - Keywords Reference `*FREQUENCY` 页：`0fb201bf2961a723cbe95b60dde35adcecfc1e2fe7f6d46c1dc5e0a921449512`
>   - Analysis Guide "Natural frequency extraction" 页：`cb22659582ec50111cf727560c02002eb79bde0aec75c89b4d00c01632c1bab2`
>   - Scripting Reference `FrequencyStep` 对象页：`48b4ba99f91091b971ee9f472cdddb232aff6f81e977f6509647bf84bffa3dc6`
> - 检索要点：3DX 文档把 `AMS cutoff1` 排版为 `A ⁢ M ⁢ S cutoff 1`（字符间夹 MathJax
>   不可见乘号），**必须先剥 HTML 标签再搜 "multiplier of the maximum frequency" 等
>   短语**；直接搜 "AMS cutoff" 或 "AMS PARAMETER" 均为 0 命中。

## 1. 关键词参考原文（*FREQUENCY, EIGENSOLVER=AMS 数据行，逐字）

数据行第 4–6 项：

> **AMS cutoff1**, the first AMS parameter.
> AMS cutoff1 is a cutoff frequency for **substructure eigenproblems**, defined as a
> **multiplier of the maximum frequency of interest**. The default value is 5.
>
> **AMS cutoff2**, the second AMS parameter.
> AMS cutoff2 is **the first cutoff frequency used to define a starting subspace in
> the reduced eigensolution phase**, defined as a multiplier of the maximum frequency
> of interest. AMS cutoff2 < AMS cutoff1. The default value is 1.7.
>
> **AMS cutoff3**, the third AMS parameter.
> AMS cutoff3 is the **second** cutoff frequency used to define a starting subspace in
> the reduced eigensolution phase, defined as a multiplier of the maximum frequency of
> interest. 1.0 < AMS cutoff3 < AMS cutoff1. The default value is 1.1.

第一数据行前 3 项：要求模态数（可留空＝取区间内全部）、最低/最高关注频率。
文档另明确：**请求的模态数不影响求解器实际提取的模态数，只影响存储/输出哪些**。

## 2. Analysis Guide 原文（三阶段职责，逐字）

> The execution of the AMS eigensolver can be controlled by specifying three
> parameters: AMS cutoff1, AMS cutoff2, and AMS cutoff3. **These three parameters
> multiplied by the maximum frequency of interest define three cutoff frequencies.**
> AMS cutoff1 (default value of 5) controls the cutoff frequency for substructure
> eigenproblems **in the reduction phase**, while AMS cutoff2 and AMS cutoff3
> (default values of 1.7 and 1.1, respectively) control the cutoff frequencies used
> to define a starting subspace **in the reduced eigensolution phase**. Generally,
> increasing the value of AMS cutoff2 and AMS cutoff3 improves the accuracy of the
> results but may affect the performance of the analysis.

Reduction phase（约化阶段）：

> You can make trade-offs between cost and accuracy during the reduction phase
> through the AMS cutoff1 parameter. **This parameter multiplied by the highest
> eigenfrequency specified for the full model yields the highest eigenfrequency
> that is extracted in the local supernode eigensolutions.** Increasing the value of
> AMS cutoff1 increases the accuracy of the reduction since more local eigenmodes
> are retained.

Reduced eigensolution phase（缩聚求解阶段）：

> Although the reduced system typically is two orders of magnitude smaller in size
> than the original system, generally it still is too large to solve directly. Thus,
> the system is **further reduced mainly by truncating the retained eigenmodes and
> then solved using a single subspace iteration step**. The two AMS parameters,
> AMS cutoff2 and AMS cutoff3, define a starting subspace of the subspace iteration
> step. ... When a more accurate solution is needed, the recommended procedure is to
> **increase both parameters proportionally from their respective default values**.

Recovery phase（恢复阶段）：用缩聚问题的特征向量与局部子结构模态恢复原系统位移。

Scripting Reference 交叉印证（`FrequencyStep` 构造参数名）：
`substructureCutoffMultiplier`（默认 5.0，"cutoff frequency for substructure
eigenproblems"）、`firstCutoffMultiplier`（1.7）、`secondCutoffMultiplier`（1.1），
均定义为 "a multiplier of the maximum frequency of interest"。

## 3. 三条硬结论（替代此前全部 [INF] 推断）

1. **三个参数都是频率乘子，且都乘同一个全局标量 f_max（最高关注频率）——不存在
   逐层增长。** 此前"底层 5.0、每层 ×1.7、界面 ×1.1"的逐层模型是错的；
   γ(v)=5·1.7^h(v) 到根部爆炸的场景在 Abaqus 语义中**不存在**。
2. **cutoff1 = 5 统一作用于所有超节点（叶子与分隔符、每一层）的局部特征问题**：
   ω_cut,local = 5·f_max，即 λ_cut = (2π·5·f_max)²。"highest eigenfrequency that is
   extracted in the local supernode eigensolutions" 是单数全局规则，无层级项。
3. **cutoff2 = 1.7 与 cutoff3 = 1.1 只在缩聚求解阶段（根上）各用一次**，作用是给
   "对缩聚系统先截断保留模态、再做单步子空间迭代"提供**初始子空间的两个频率门槛**；
   与任何中间节点的截断无关。约束：cutoff2 < cutoff1，1.0 < cutoff3 < cutoff1。

## 4. 与 PKU 列压缩移植的映射

| Abaqus | PKU/本移植对应物 | 一致性 |
|---|---|---|
| cutoff1=5，ω_cut=(5·f_max) 全局统一 | `gainCoefficient`（γ），λ_hi=(2π·γ·f_user)²，逐节点同一 γ | **新截断语义与 Abaqus 同构**（频率乘子）；旧语义 λ_cut=η·(2πf)² 是特征值乘子，等价换算 γ=√η（η=4 ⇒ γ=2），已写入 docs/列压缩移植基准对比.md |
| cutoff2/cutoff3：根上初始子空间两道频率门槛（之后单步子空间迭代） | 本移植最终直接对缩聚 pencil 做 dsygvx 区间求解，**无子空间迭代 ⇒ 无对应物**（更精确、根上更贵）；PKU 原码同样无此二参数 | 有意偏离，属最终求解策略差异 |
| 缩聚后 M̃ 对角恒 1、非对角为超节点耦合块 | `SubspaceMassSaver::assembleSubspaceMass` 对角恒填 1 约定 | 逐字一致（文档原话 "the reduced mass matrix has unit diagonal values but contains off-diagonal blocks ... coupling between the supernodes"） |
| 请求模态数只影响存储不影响提取 | f_user 协议（minFreq/maxFreq 区间提取） | 一致 |

## 5. 对既有实测证据的重新解读

- **EAMLS 实测 ω_b=11ω_c、ω_Γ=16.5ω_c**（docs/AMS技术分析.md §3.4）：不再需要
  "界面用更高截止"的参数解释——那是该文献作者自己的实现选择（论文级证据），与
  Abaqus 的单一 cutoff1=5 规则并存不悖；Abaqus 下同一算例各层保留率差异来自
  **缩聚使上层谱自动变软**（常数 γ 下保留率自然上升），而非参数随层放大。
- **cant_20 p=12（缩聚规模）vs PKU p=87**：Abaqus 树细（7898 超节点 vs 40），
  每超节点本征维数小，5·f_max 窗口内自然只留少量模态；粗树（PKU）在同样 γ 下
  每节点保留更多。这解释了"参数更松、总量反而小"的反直觉现象，且无需任何
  逐层机制。
- `AMS cutoff1/2/3` 字符串与 `ABA_PAMS_LAST_CUTOFF`（求解器二进制 strings）与
  文档三参数一一对应；探针侧"局部截断公式 [INF]"现升级为 [D]。

## 6. 遗留未定（仍为推断/待观测）

- cutoff2/cutoff3 定义初始子空间的**具体构造式**（哪些模态进初始子空间、二者如何
  分工：文档只说 "define a starting subspace"，未给公式）——文档未公开，属实现细节。
- Abaqus 缩聚阶段的单步子空间迭代细节（迭代次数、收敛判据）——文档未给。
- 本地文档为 2020 版；用户另有 2025 介质，关键词页措辞如有变化需按同法复查。
