# Abaqus AMS 对比分析文档集

本项目对标 Abaqus/Standard 的 AMS（Automatic Multi-level Substructuring）
特征值求解器，分析其算法技术细节，为自研 AMLS 实现（`src/amls/`）提供
优化方向，并建立"同一矩阵、同一机器"的公平基准。

> 当前使用 Abaqus 2020，Standard/AMS 和小模型矩阵导出已运行成功。历史安装状态及旧输入示例已过时；关于实际剖分大小/树形的结论请先阅读 [AMS实测证据与更正.md](AMS实测证据与更正.md)。10 Hz 独立复测因模型基频为 27.597 Hz 而未找到模态，结果以错误退出保存。

## 文档

| 文件 | 内容 | 状态 |
|---|---|---|
| [AMS技术分析.md](AMS技术分析.md) | AMS/AMLS 算法技术细节：阶段划分、剖分、谱变换数学、蒸馏子空间、并行化、精度增强、论文实测基准、本项目差距与优化方向 | ✅ 已完成（论文 3 路笔记已整合） |
| [运行期观测清单.md](运行期观测清单.md) | Abaqus AMS 运行期的观测采集清单（.dat/.msg/内存估算/环境变量/矩阵导出） | ✅ 清单已就绪，待实测填充 |
| [AMS技术路线实测报告.md](AMS技术路线实测报告.md) | **2026-10-09 探针实测**：120Hz 下用 ABA_PAMS_* 等内部开关导出 stf/times/orth，解出阶段划分、超节点重排、METIS、LAPACK 子结构特征求解 | ✅ 实测（探针 dump 已在 bench_abaqus/） |
| [AMS实测证据与更正.md](AMS实测证据与更正.md) | 更正历史无证据结论；记录 10Hz/120Hz 实测边界 | ✅ |
| [AMS对比PKU_CCM.md](AMS对比PKU_CCM.md) | **2026-10-09 同模型对比**：Abaqus 7898 超节点(3–24行) vs PKU ND 40 节点(顶层1458)；子空间 12 vs 87；根因量化"根部并行阻塞" | ✅ 实测 |
| [Abaqus矩阵导出与PKU喂入六问实证.md](Abaqus矩阵导出与PKU喂入六问实证.md) | **2026-10-09**：超节点树喂 PKU 可行性、B̃ 不正定定位（3 负特征值）、M 集中质量实证、1e36 大数法后处理、nutrix/bef 移除 | ✅ 实证 |
| [外部树剖分器开发与验证.md](外部树剖分器开发与验证.md) | **2026-10-10**：`--partition external --tree-file` 外部树注入；ND 树导出回读闭环；单根/15节点树频率逐位一致；Abaqus 7898 块实测 | ✅ 开发+验证 |

## 状态

- **Abaqus 安装**：CAE-only，Standard 求解器缺失（`exec/` 空），
  `abaqus verify -all` 三个求解器全部 ERROR。待完整重装。
- **算法分析**：基于 AMLS 奠基文献完成（Kim 2004 / Gao 2008 / Bennighof
  SIAM 2004 / Wang 系列 / Yin 2013 / Hyun 2019），非 Abaqus 内部代码。
- **官方文档**：在线需 3DS 登录 + API 额度失败；重装后从本地文档/实测补
  [DOC] 条目。

## 待办（Abaqus 完整重装后）

1. 按 `运行期观测清单.md` 采集 AMS job 的观测数据（小模型先验证链路）。
2. 生成 100 万 DOF 悬臂块（`tools/gen_abaqus_cantilever.py` 已就绪），
   跑 `*FREQUENCY, EIGENSOLVER=AMS` 得参考解 + 性能/内存。
3. `*MATRIX GENERATE` 导出 K/M → Matrix Market → 喂 `amls_solve --K --M`，
   同矩阵对比频率、耗时、内存。
4. 按 `AMS技术分析.md` §8 优化方向逐项落地，用对比数据验证。
