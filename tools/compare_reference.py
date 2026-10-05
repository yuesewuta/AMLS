#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 AMLS 求解结果与前 200 阶参考解。

读取 benchmark/results/<name>_amls.txt (每行 "序号 特征值",# 开头为注释)
与 benchmark/reference/<name>_ref200.npz,输出逐阶特征值/频率相对误差及汇总。

注意 run_benchmark.sh 传 M = I,AMLS 输出对应标准特征值 mu(K x = mu x),
故这里与 npz 中的 mu 对比;alpha 缩放只影响物理频率 f_user,不影响该对比。

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
                             "thermomech_TK", "thermomech_dK", "boneS10"])
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
        lam_ref, f_ref = z["mu"], z["f"]
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
