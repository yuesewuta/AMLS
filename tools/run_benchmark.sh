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
    NAMES=(bcsstk13 s1rmq4m1 bcsstk18 smt thermomech_TK thermomech_dK
           af_shell3 boneS10)
fi

# f_user 协议:从 f_user.csv 查各矩阵的 alpha(质量缩放)与 f_user(Hz)。
# AMLS 以 M = alpha*I 求解,--fuser 触发绝对阈值截断(λ_cut=η·(2πf_user)²,
# η=thresholdFactor:分量模态截到 η 倍),输出全部降阶模态,与参考解 lam
# 列对比。η 按规模分档:η=4(rel@200≈1%)小中矩阵;≥50万 用 η=2 控内存
# (rel@200≈2-3%);bcsstk13 为半正定自由-自由结构,粗树空间贫瘠,特调
# 细剖分 + 大 η。
CSV=benchmark/f_user.csv
eta_for() {
    case "$1" in
        bcsstk13) echo 8 ;;
        af_shell3|boneS10) echo 2 ;;
        *) echo 4 ;;
    esac
}
for name in "${NAMES[@]}"; do
    mtx=$(find "$MATRIX_DIR" -name "$name.mtx" | head -1)
    echo "=== $name ($mtx) ==="
    row=$(awk -F, -v n="$name" '$1==n {print $4" "$7}' "$CSV")
    if [ -z "$row" ]; then
        echo "!! f_user.csv 中无 $name,先运行 make_reference.py" >&2
        continue
    fi
    alpha=${row%% *}; fuser=${row##* }
    eta=$(eta_for "$name")
    extra=""
    [ "$name" = bcsstk13 ] && extra="--leaf-size 100"
    echo "    alpha=$alpha f_user=$fuser eta=$eta $extra"
    "$SOLVER" --K "$mtx" --mass-scale "$alpha" --fuser "$fuser" --eta "$eta" \
        $extra --out "benchmark/results/${name}_amls.txt" --quiet \
        2>&1 | tee "benchmark/results/${name}_run.log" | grep -E "^# AMLS|^# 计时|^# 内存" || true
done
echo "完成,结果见 benchmark/results/"
