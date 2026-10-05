#!/usr/bin/env bash
# 基准对比运行脚本:对每个基准矩阵跑 AMLS(M = I,前 200 阶)。
# 用法: bash tools/run_benchmark.sh [矩阵名...]   # 缺省全部;结果在 benchmark/results/
# 前置: 1) bash tools/fetch_benchmark.sh   2) python tools/make_reference.py
#       3) cmake --build build   (amls_solve.exe)
set -euo pipefail
cd "$(dirname "$0")/.."
# mingw64 运行时 DLL(spdlog/OpenBLAS 等)
for d in /d/msys2/mingw64/bin /c/msys64/mingw64/bin /mingw64/bin; do
    [ -d "$d" ] && PATH="$d:$PATH"
done
SOLVER=build/amls_solve.exe
[ -x "$SOLVER" ] || SOLVER=./build/amls_solve.exe
mkdir -p benchmark/results

MATRIX_DIR=benchmark/matrices
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then
    NAMES=(bcsstk13 s1rmq4m1 bcsstk18 smt thermomech_TK thermomech_dK boneS10)
fi

for name in "${NAMES[@]}"; do
    mtx=$(find "$MATRIX_DIR" -name "$name.mtx" | head -1)
    echo "=== $name ($mtx) ==="
    # M = I:不传 --M 即可,amls_solve 自动使用单位阵
    "$SOLVER" --K "$mtx" --nev 200 --out "benchmark/results/${name}_amls.txt" --quiet
done
echo "完成,结果见 benchmark/results/"
