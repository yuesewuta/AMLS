# 测试矩阵数据（不入库）

本目录的 Matrix Market 矩阵来自 SuiteSparse Matrix Collection（https://sparse.tamu.edu），
体积较大（约 800MB），不纳入 git 仓库。

获取方式：

    bash tools/fetch_matrices.sh

下载 HB/bcsstk01,08,09,13（刚度）与对应 bcsstm0x（质量）矩阵，以及 ND/nd3k,6k,12k,24k。
