#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 AMLS 求解结果与前 200 阶参考解。

读取 benchmark/results/<name>_amls.txt (每行 "序号 特征值",# 开头为注释)
与 benchmark/reference/<name>_ref200.npz,输出逐阶特征值/频率相对误差及汇总。

对比对象为 npz 中的 lam(物理特征值 K x = lam * M x,M = alpha*I)。
run_benchmark.sh 按 f_user.csv 的 alpha 传 --mass-scale,AMLS 求解的
正是该物理问题,其输出特征值直接与 lam 对比(alpha=1 时 lam == mu)。

用法: python tools/compare_reference.py [--dir benchmark] [矩阵名...]
"""
import argparse
from pathlib import Path

import numpy as np


def load_amls(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) >= 2:
            try:
                rows.append((int(float(p[0])), float(p[1])))
            except ValueError:
                continue
    rows.sort()
    return np.array([lam for _, lam in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark")
    ap.add_argument("names", nargs="*",
                    default=["bcsstk13", "s1rmq4m1", "bcsstk18", "smt",
                             "thermomech_TK", "thermomech_dK", "af_shell3",
                             "boneS10"])
    args = ap.parse_args()
    root = Path(args.dir)

    print(f"{'matrix':<16}{'n阶':>5}{'mu200_ref':>14}{'mu200_amls':>14}"
          f"{'relErr@200':>12}{'maxRelErr':>12}{'freqErr@200':>13}")
    for name in args.names:
        ref_p = root / "reference" / f"{name}_ref200.npz"
        out_p = root / "results" / f"{name}_amls.txt"
        if not ref_p.exists() or not out_p.exists():
            print(f"{name:<16}  缺少参考解或结果文件,跳过")
            continue
        z = np.load(ref_p)
        lam_ref, f_ref = z["lam"], z["f"]
        lam = load_amls(out_p)
        k = min(len(lam), len(lam_ref))
        # 刚体/近零模态用绝对误差保护;频率相对误差 ≈ 特征值相对误差的一半
        denom = np.maximum(np.abs(lam_ref[:k]), 1e-30)
        rel = np.abs(lam[:k] - lam_ref[:k]) / denom
        fre = abs(np.sqrt(max(lam[k-1], 0)) - np.sqrt(max(lam_ref[k-1], 0))) \
            / max(np.sqrt(abs(lam_ref[k-1])), 1e-30)
        print(f"{name:<16}{k:>5}{lam_ref[k-1]:>14.6e}{lam[k-1]:>14.6e}"
              f"{rel[-1]:>12.3e}{rel.max():>12.3e}{fre:>13.3e}")
        worst = np.argsort(rel)[-3:][::-1]
        print(f"{'':>16}误差最大三阶: " +
              ", ".join(f"#{i+1}:{rel[i]:.2e}" for i in worst))


if __name__ == "__main__":
    main()
