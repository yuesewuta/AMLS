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
    "af_shell3",      # Schenk_AFE, 504855
    "boneS10",        # Oberwolfach,914898
]

MAGIC = b"AMLSBIN1"


def find_mtx(mtx_dir: Path, name: str) -> Path:
    hits = list(mtx_dir.glob(f"**/{name}.mtx"))
    if not hits:
        sys.exit(f"找不到 {name}.mtx,先运行 bash tools/fetch_benchmark.sh")
    return hits[0]


def header_is_symmetric(path: Path) -> bool:
    """读 MatrixMarket banner,判断是否为对称(仅三角)存储。"""
    with open(path, "r", errors="replace") as f:
        banner = f.readline().lower()
    return banner.startswith("%%matrixmarket") and "symmetric" in banner


def load_stiffness(path: Path):
    """读入刚度矩阵 mtx 并对称化,返回 CSR。

    - symmetric 头:只存三角,scipy 自动镜像补齐。
    - general 头:可能是全存储的对称矩阵(如 thermomech_dK,上下三角各存一份,
      二者有 <=1e-4 的装配舍入差)。此时必须 (A+Aᵀ)/2 平均;若用 A+Aᵀ-diag
      会把所有非对角元翻倍,谱整体放大 2 倍。
    """
    print(f"  读入 {path} ...", flush=True)
    A = sp.csr_matrix(sio.mmread(str(path)))
    if not header_is_symmetric(path):
        scale = float(abs(A).max()) or 1.0
        asym = float(abs(A - A.T).max()) / scale
        if asym > 1e-6:
            sys.exit(f"{path.name}: general 存储但数值不对称(相对偏差 {asym:.2e}),"
                     "请人工确认该矩阵是否为刚度矩阵")
        print(f"  general 全存储,数值对称(相对偏差 {asym:.2e}),取 (A+Aᵀ)/2",
              flush=True)
        A = (A + A.T) * 0.5
    elif (A != A.T).nnz:
        A = A + A.T - sp.diags(A.diagonal())
    A.eliminate_zeros()
    return sp.csr_matrix(A)


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


def split_decoupled(A):
    """剥离解耦自由度(只有对角非零的行列)。

    ARPACK/LOBPCG 都是单/块迭代法,对简并特征值只返回一份副本
    (实测 bcsstk18 的 729 重 lambda=1 簇几乎全部丢失)。解耦自由度的
    特征值可以解析给出 lambda = K_ii(M=I),先从矩阵里剥掉,迭代法只在
    剩余耦合部分求,最后把解析值并回谱里,多重性才完整。
    返回 (耦合子矩阵, 解析特征值数组)。
    """
    off = A - sp.diags(A.diagonal())
    off.eliminate_zeros()
    r_off = np.asarray(off.getnnz(axis=1)).ravel()
    iso = np.where(r_off == 0)[0]
    if iso.size == 0:
        return A, np.array([])
    keep = np.ones(A.shape[0], dtype=bool)
    keep[iso] = False
    vals = np.asarray(A.diagonal())[iso]
    print(f"  剥离 {iso.size} 个解耦自由度(解析特征值 lambda=K_ii)",
          flush=True)
    return A[np.ix_(keep, keep)].tocsr(), vals


def solve_reference(A):
    """求前 NEV 阶最小特征值 mu(K x = mu x)。

    - 解耦自由度先剥离(见 split_decoupled),解析特征值最后并回;
    - n <= 13 万: splu 精确移位反演 + ARPACK;
    - 更大矩阵: LOBPCG + PyAMG 代数多重网格预条件(低内存;直接 LU 对 3D
      刚度矩阵 fill-in 过大,纯 ARPACK which='SA' 对内特征值收敛过慢)。
    """
    A, isoVals = split_decoupled(A)
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
        mu = _lobpcg_amg(A)
    mu = np.sort(np.concatenate([np.real(mu), isoVals]))
    print(f"  求解完成,{time.time()-t0:.1f}s,mu 范围 [{mu[0]:.3e}, {mu[-1]:.3e}]",
          flush=True)
    return mu


def _sa(A):
    n = A.shape[0]
    v0 = np.ones(n)
    mu = eigsh(A, k=NEV, which="SA", tol=1e-8, ncv=min(n, max(4 * NEV, 800)),
               maxiter=10 * n, v0=v0, return_eigenvectors=False)
    return np.sort(np.real(mu))


def _lobpcg_amg(A):
    """LOBPCG + smoothed-aggregation AMG 求最小 NEV 阶(多取 16 阶保证第 200 阶质量)。"""
    from scipy.sparse.linalg import lobpcg
    try:
        import pyamg
    except ImportError:
        sys.exit("大矩阵参考解需要 PyAMG:pip install pyamg(否则退回极慢的 which='SA')")

    n = A.shape[0]
    k = NEV + 16
    scale = max(float(np.max(np.abs(A.diagonal()))), 1.0)
    sigma = -1e-8 * scale  # 预条件用 K - sigma*I,半正定时也保证 SPD
    print(f"  构建 AMG 层级 (n={n}, nnz={A.nnz}) ...", flush=True)
    t0 = time.time()
    ml = pyamg.smoothed_aggregation_solver(
        (A - sigma * sp.identity(n, format="csc")).tocsr(),
        max_coarse=500, max_levels=12)
    print(f"  AMG 完成,{time.time()-t0:.1f}s,{len(ml.levels)} 层;开始 LOBPCG(k={k}) ...",
          flush=True)

    rng = np.random.default_rng(20261005)
    X = rng.standard_normal((n, k))
    # 预光滑初始空间(逐列,pyamg solve 只接受一维右端),显著加速收敛
    for j in range(k):
        X[:, j] = ml.solve(X[:, j], tol=1e-6, maxiter=3)
    X, _ = np.linalg.qr(X)
    def precond(B):
        out = np.empty_like(B)
        for j in range(B.shape[1]):
            out[:, j] = ml.solve(B[:, j], tol=1e-8, maxiter=2)
        return out
    M = LinearOperator(A.shape, matvec=precond, matmat=precond, dtype=A.dtype)
    t0 = time.time()
    w, _ = lobpcg(A, X, B=None, M=M, largest=False, tol=1e-9, maxiter=400,
                  retLambdaHistory=False)
    print(f"  LOBPCG 完成,{time.time()-t0:.1f}s", flush=True)
    return np.sort(np.real(w))[:NEV]


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
    csv_path = root / "f_user.csv"
    header = ["matrix", "n", "nnz", "mass_scale_alpha",
              "lambda_200", "f200_Hz", "f_user_Hz"]
    rows = []

    def write_csv():
        # 每算完一个矩阵就增量落盘,避免大矩阵中断丢失已完成结果
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in rows:
                w.writerow([r["name"], r["n"], r["nnz"], r["alpha"],
                            f"{r['lam200']:.6e}", f"{r['f200']:.6e}",
                            f"{r['f_user']:g}"])

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
            write_csv()
            continue

        mtx = find_mtx(mtx_dir, name)
        A = load_stiffness(mtx)
        n, nnz = A.shape[0], A.nnz
        mu = solve_reference(A)
        if mu.size < NEV:
            sys.exit(f"{name}: 合并解耦特征值后仍不足 {NEV} 阶")
        mu = mu[:NEV]   # 前 NEV 阶(解耦解析值已并入并升序)

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
        print(f"  f200 = {f200:.6g} Hz -> f_user = {fu:g} Hz", flush=True)
        rows.append(dict(name=name, n=n, nnz=nnz, alpha=alpha,
                         lam200=float(lam[-1]), f200=f200, f_user=fu))
        write_csv()
        del A

    print(f"\n汇总写入 {csv_path}")


if __name__ == "__main__":
    main()
