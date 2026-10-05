#!/usr/bin/env bash
# 下载 SuiteSparse 测试矩阵（验证方案 §2.2）。用法: bash tools/fetch_matrices.sh [目标目录]
set -euo pipefail
DIR="${1:-tests/matrices}"
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

fetch HB bcsstk01; fetch HB bcsstm01
fetch HB bcsstk08; fetch HB bcsstm08
fetch HB bcsstk09; fetch HB bcsstm09
fetch HB bcsstk13; fetch HB bcsstm13
fetch ND nd3k; fetch ND nd6k; fetch ND nd12k; fetch ND nd24k
echo "完成: $(ls */*.mtx | wc -l) 个矩阵"
