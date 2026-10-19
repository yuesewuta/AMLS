# AMS 超节点块原子 = 网格节点 DOF 数（dp）——壳/平面/实体三案验证

日期：2026-10-19。探针：`ABA_PAMS_EXPORT_SOLVER_DATA=1`（bench_abaqus/abaqus_v6.env），
stf 二进制解析工具 `tools/stf_block_hist.py`（块识别判据同 abaqus_stf_tree.py：块内行度逐行减 1）。

## 结论

**AMS 的消元块原子不是"3"，而是"每个网格节点的自由度数 dp"。**
基础块尺寸随单元类型精确变化，全部块是 dp 的倍数（更大的块是聚合合并的产物）：

| 模型 | 单元 | dp | DOF | 基础块 | 块尺寸直方图 | 全部为 dp 倍数 |
|---|---|---|---|---|---|---|
| cant_10/20/31/plate | C3D8 实体 | 3 | 3993~98304 | 3 | {3:957, 6:157, 9:12, 12:6}（cant_10）等 | ✔ 全为 3 的倍数 |
| cant_shell | S4R 壳 | 6 | 1230 | 6 | {6:141, 12:23, 18:6} | ✔ 全为 6 的倍数 |
| cant_plane2d | CPS4 平面应力 | 2 | 410 | 2 | {2:137, 4:25, 6:6} | ✔ 全为 2 的倍数 |

- 壳模型 205 节点 × 6 DOF，基础块恰为 6；平面模型 205 节点 × 2 DOF，基础块恰为 2。
- 说明之前实测的"{3: 9143, ...}"中的 3 并非 AMS 内部常量，而是实体单元 dp=3 的投影。
  逆向文档中"节点级图跑 NodeND"的表述应推广为：**AMS 在网格节点图上做一切拓扑决策，
  节点宽度随单元类型而定**。
- 对我们的实现（tree_builder.cpp）：`extractNodeGraph/detectGroupedLayout` 本身已参数化 dp，
  但 `buildAbaqusTree` 自动检测只尝试 dp=3（`aams/src/ams/tree_builder.cpp:152`）；
  对壳/平面模型需扩展检测序列（2,3,6）或经 `opt.dofsPerNode` 显式指定。

## 实验工程记录（教训）

- 新 inp 必须用 **CRLF** 行尾（Git Bash 生成 LF 文件 Abaqus 前处理器仍能读，但需注意）。
- `*SHELL SECTION` 语法（与 CalculiX 相反！）：数据行是 `厚度, 积分点数`（如 `1.0, 5`），
  材料名走关键字参数 `MATERIAL=steel`。写成 `1.0, steel` 会报
  `OdbError: Material name is missing for this section`。
- `ELSET=EALL` 须在 `*ELEMENT` 行上定义（`*ELEMENT, TYPE=S4R, ELSET=EALL`），
  Abaqus 不会自动创建 EALL。
- 生成脚本：`tools/gen_cant_dp_test.py`；CAE 语法对照件：`bench_abaqus/gen_cae_shell.py`
  → `cae_shell_kw.inp`（ground-truth 关键字）。
