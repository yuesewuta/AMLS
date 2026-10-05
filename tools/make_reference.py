#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成基准参考解（只跑一次）。

对每个基准刚度矩阵 K（M = I）求前 200 阶最小特征对，产出:
  benchmark/bin/<name>.bin               mtx 转成的二进制(CSR),避免每次重读 mtx
  benchmark/reference/<name>_ref200.npz  前 200 阶特征值 mu(标准特征值 K x = mu x)、
                                         质量缩放因子 alpha、物理特征值 lam = mu/alpha、频率 f
  benchmark/f_user.csv                   每个矩阵的第 200 阶频率 f200 及输入频率 f_user

频率与 f_user 规则:
  M = alpha * I 时 K x = lam * M x 等价于标准特征值问题 K x = mu x (mu = lam/alpha),
  物理频率 f = sqrt(lam) / (2*pi)。
  f_user = 把 f200 写成科学计数法后向上取整到最高有效位:
           例如 f200 = 7821 Hz -> 7.8e3 -> f_user = 8000 (Hz)。

病态处理: 若求出的特征值跨度过大(如 1e-6 或 1e13),不重新求解,只选取 alpha
(10 的整数幂)把物理谱平移到正常量纲区间,等价于调整单位质量阵的大小。

用法:
  python tools/make_reference.py [--dir benchmark] [--only 名字] [--save-vectors]
"""
import argparse
import csv
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu, LinearOperator, ArpackNoConvergence

NEV = 200

MATRICES = [
    "bcsstk13",       # HB,          2003
    "s1rmq4m1",       # Cylshell,    5489
    "bcsstk18",       # HB,         11948
    "smt",            # TKK,        25710
    "thermomech_TK",  # Botonakis, 102158
    "thermomech_dK",  # Botanakis, 204316
    "boneS10",        # Oberwolfach,914898
]

MAGIC = b"AMLSBIN1"


def find_mtx(mtx_dir: Path, name: str) -> Path:
    hits = list(mtx_dir.glob(f"**/{name}.mtx"))
    if not hits:
        sys.exit(f"找不到 {name}.mtx,先运行 bash tools/fetch_benchmark.sh")
    return hits[0]


def load_stiffness(path: Path):
    """读入 mtx,取上三角对称化,返回 CSR。"""
    print(f"  读入 {path} ...", flush=True)
    A = sio.mmread(str(path))
    A = sp.csr_matrix(A)
    # mtx 中可能只存了上/下三角
    if (A != A.T).nnz != 0:
        A = A + A.T - sp.diags(A.diagonal())
    A = sp.csr_matrix(A)
    A.eliminate_zeros()
    return A


def write_bin(path: Path, A: sp.csr_matrix) -> None:
    """AMLSBIN1 格式(小端):
       magic[8] | version u32 | n i64 | nnz i64 | indptr i64[n+1] | indices i32[nnz] | data f64[nnz]
    """
    A = sp.csr_matrix(A)
    indptr = A.indptr.astype("<i8")
    indices = A.indices.astype("<i4")
    data = A.data.astype("<f8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Iqq", 1, A.shape[0], A.nnz))
        f.write(indptr.tobytes())
        f.write(indices.tobytes())
        f.write(data.tobytes())


def solve_reference(A):
    """求前 NEV 阶最小特征值 mu(K x = mu x)。n<=13 万用移位反演,更大用 which='SA'。"""
    n = A.shape[0]
    t0 = time.time()
    if n <= 130_000:
        # 小负移位:半正定(刚体模态)时 K - sigma*I 仍正定,分解稳定
        scale = max(float(abs(A).max()), float(np.max(np.abs(A.diagonal()))), 1.0)
        sigma = -1e-8 * scale
        try:
            print(f"  移位反演 (sigma={sigma:.3e}) 分解 {n} 阶 ...", flush=True)
            lu = splu((A - sigma * sp.identity(n, format="csc")).tocsc())
            op = LinearOperator(A.shape, lu.solve, dtype=A.dtype)
            mu = eigsh(A, k=NEV, sigma=sigma, which="LM", OPinv=op,
                       tol=0, return_eigenvectors=False)
        except (RuntimeError, ArpackNoConvergence) as e:
            print(f"  移位反演失败({e!r}),改用 which='SA' ...", flush=True)
            mu = _sa(A)
    else:
        mu = _sa(A)
    mu = np.sort(np.real(mu))
    print(f"  求解完成,{time.time()-t0:.1f}s,mu 范围 [{mu[0]:.3e}, {mu[-1]:.3e}]",
          flush=True)
    return mu


def _sa(A):
    n = A.shape[0]
    v0 = np.ones(n)
    mu = eigsh(A, k=NEV, which="SA", tol=1e-8, ncv=min(n, max(4 * NEV, 800)),
               maxiter=10 * n, v0=v0, return_eigenvectors=False)
    return np.sort(np.real(mu))


def f_of(lam):
    return np.sqrt(np.maximum(lam, 0.0)) / (2 * math.pi)


def f_user_of(f200: float) -> float:
    e = math.floor(math.log10(f200))
    return math.ceil(f200 / 10 ** e) * 10 ** e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark")
    ap.add_argument("--only", default=None, help="只处理指定矩阵")
    args = ap.parse_args()

    root = Path(args.dir)
    mtx_dir, bin_dir, ref_dir = root / "matrices", root / "bin", root / "reference"
    for d in (bin_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = [args.only] if args.only else MATRICES
    rows = []
    for name in names:
        print(f"[{name}]", flush=True)
        npz_path = ref_dir / f"{name}_ref{NEV}.npz"
        if npz_path.exists():
            print("  已有参考解,跳过(删除该 npz 可强制重算)", flush=True)
            with np.load(npz_path) as z:
                rows.append(dict(name=name, n=int(z["n"]), nnz=int(z["nnz"]),
                                 alpha=float(z["alpha"]),
                                 lam200=float(z["lam"][-1]),
                                 f200=float(z["f"][-1]),
                                 f_user=float(z["f_user"])))
            continue

        mtx = find_mtx(mtx_dir, name)
        A = load_stiffness(mtx)
        n, nnz = A.shape[0], A.nnz
        mu = solve_reference(A)

        # 质量缩放:alpha 取 10 的整数幂,把第 200 阶物理特征值拉回 1e4 量级
        lam200 = mu[-1]
        alpha = 1.0
        if not (1e-2 <= lam200 <= 1e8):
            alpha = 10.0 ** round(math.log10(lam200 / 1e4))
            print(f"  谱量纲异常(lambda200={lam200:.3e}),取 M = {alpha:g}·I 平衡", flush=True)
        lam = mu / alpha
        f = f_of(lam)
        f200 = float(f[-1])
        fu = f_user_of(f200) if f200 > 0 else float("nan")

        np.savez_compressed(
            npz_path, n=n, nnz=nnz, alpha=alpha, mu=mu, lam=lam, f=f,
            f_user=fu)
        write_bin(bin_dir / f"{name}.bin", A)
        print(f"  f200 = {f200:.6g} Hz -> f_user = {fu:.0f} Hz", flush=True)
        rows.append(dict(name=name, n=n, nnz=nnz, alpha=alpha,
                         lam200=float(lam[-1]), f200=f200, f_user=fu))
        del A

    with open(root / "f_user.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["matrix", "n", "nnz", "mass_scale_alpha",
                    "lambda_200", "f200_Hz", "f_user_Hz"])
        for r in rows:
            w.writerow([r["name"], r["n"], r["nnz"], r["alpha"],
                        f"{r['lam200']:.6e}", f"{r['f200']:.6e}",
                        f"{r['f_user']:.0f}"])
    print(f"\n汇总写入 {root/'f_user.csv'}")


if __name__ == "__main__":
    main()
