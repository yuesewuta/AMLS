#!/usr/bin/env bash
# 基准对比运行脚本:对每个基准矩阵跑 AMLS(f_user 模式,前 200 阶量级)。
# 用法: bash tools/run_benchmark.sh [矩阵名...]   # 缺省全部;结果在 benchmark/results/
# 前置: 1) bash tools/fetch_benchmark.sh   2) python tools/make_reference.py
#       3) cmake --build build   (amls_solve.exe reference_tool.exe)
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
# η=thresholdFactor:分量模态截到关注频段的 η 倍),输出全部降阶模态,
# 与参考解 lam 列对比(见 compare_reference.py)。
# 每矩阵的 η/剖分深度为实测标定:
#   bcsstk13 半正定自由-自由结构,粗树空间贫瘠 → 细剖分(leafSize=100)+η=8;
#   af_shell3 50万 3D 壳:6 层近根稠密块爆内存 → 8 层;η=2/4 截断过紧
#   (relErr@200 29%/9.7%),η=8 → 2.3e-2;
#   boneS10 100万 3D:8 层 OOM → 9 层;η=8 仍 OOM(15.8GB 内存墙),η=4
#   → relErr@200 1.1e-1。
CSV=benchmark/f_user.csv
eta_for() {
    case "$1" in
        bcsstk13|af_shell3) echo 8 ;;
        boneS10) echo 4 ;;
        *) echo 4 ;;
    esac
}
nlevels_for() {
    case "$1" in
        af_shell3) echo 8 ;;
        boneS10) echo 9 ;;
        *) echo 6 ;;
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
    nl=$(nlevels_for "$name")
    extra=""
    [ "$name" = bcsstk13 ] && extra="--leaf-size 100"
    echo "    alpha=$alpha f_user=$fuser eta=$eta nlevels=$nl $extra"
    "$SOLVER" --K "$mtx" --mass-scale "$alpha" --fuser "$fuser" --eta "$eta" \
        --nlevels "$nl" $extra --out "benchmark/results/${name}_amls.txt" --quiet \
        2>&1 | tee "benchmark/results/${name}_run.log" | grep -E "^# AMLS|^# 计时|^# 内存" || true
done
echo "完成,结果见 benchmark/results/"
