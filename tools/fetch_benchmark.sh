#!/usr/bin/env bash
# 下载基准刚度矩阵（SuiteSparse Matrix Collection，Matrix Market 格式）。
# 用法: bash tools/fetch_benchmark.sh [目标目录]   # 默认 benchmark/matrices/
#
# 基准矩阵一览（质量矩阵一律取单位阵 M = I）:
#   目标规模   矩阵                      阶数     说明
#   ≈2千      HB/bcsstk13               2003     Boeing 飞机结构刚度,半正定(含刚体模态)
#   ≈5千      Cylshell/s1rmq4m1         5489     圆柱壳 FEM 刚度,正定
#   ≈1万      HB/bcsstk18              11948     飞机结构刚度
#   ≈5万      TKK/smt                  25710     晶体管热应力刚度
#   ≈10万     Botonakis/thermomech_TK 102158     热固耦合刚度
#   ≈50万     Botanakis/thermomech_dK 204316     热固耦合刚度(20万级,-collection 无 50 万档独立结构 K)
#   ≈100万    Oberwolfach/boneS10      914898    取 A 即刚度 K
set -euo pipefail
DIR="${1:-benchmark/matrices}"
mkdir -p "$DIR"
cd "$DIR"

fetch() {  # fetch <组> <名>
    local group="$1" name="$2"
    if [ -f "$name/$name.mtx" ]; then
        echo "已有 $name,跳过"
        return
    fi
    echo "下载 $group/$name ..."
    curl -sL --retry 3 -o "$name.tar.gz" "https://sparse.tamu.edu/MM/$group/$name.tar.gz"
    tar xzf "$name.tar.gz"
    rm "$name.tar.gz"
}

fetch HB bcsstk13
fetch Cylshell s1rmq4m1
fetch HB bcsstk18
fetch TKK smt
fetch Botonakis thermomech_TK
fetch Botanakis thermomech_dK
fetch Oberwolfach boneS10
echo "完成: $(du -sh . | cut -f1),共 $(ls */*.mtx | wc -l) 个矩阵"
