#!/usr/bin/env python3
"""Generate a consistent-mass M matrix sharing K's sparsity pattern.

Reads a symmetric/general MatrixMarket stiffness matrix K and writes an SPD
mass matrix M with the SAME nonzero pattern (the physical situation of a
FEM model: K and M assembled from the same mesh/shape functions).

Value construction (unit-shifted graph-Laplacian style, strictly diagonally
dominant hence SPD, positive-definiteness survives Schur complement):
    M_ij = -eps * |K_ij|                  (i != j, wherever K_ij is nonzero)
    M_ii = 1 + eps * sum_j |K_ij|         (strictly diagonally dominant by 1)
The pattern of M equals the pattern of K exactly.
"""
import argparse
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kfile', type=Path)
    ap.add_argument('mfile', type=Path)
    ap.add_argument('--eps', type=float, default=0.5,
                    help='Laplacian coupling weight (M = I + eps*L_K)')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='multiply whole M by this (mass scaling)')
    args = ap.parse_args()

    if args.eps <= 0:
        sys.exit('--eps must be positive')
    # 1. Read K as (i, j, |v|) over the stored half; symmetric K is assumed.
    rows, cols, banner_sym = -1, -1, False
    nz = []
    with args.kfile.open(errors='replace') as f:
        first = True
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            if first:
                parts = line.split()
                if len(parts) == 5 and parts[0] == '%%MatrixMarket':
                    banner = parts[1:]
                    if len(banner) >= 4:
                        rows, cols = int(banner[0]), int(banner[1])
                        banner_sym = banner[3] in ('symmetric',)
                else:
                    rows, cols = int(parts[0]), int(parts[1])
                first = False
                continue
            if '(' in line or ';' in line:
                line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = line.split()
            if len(parts) < 3:
                continue
            i, j, v = int(parts[0]) - 1, int(parts[1]) - 1, abs(float(parts[2]))
            nz.append((i, j, v))
    if rows < 0 or cols < 0 or rows != cols:
        sys.exit('could not parse dimensions / non-square K')
    # 2. Row sums of |K| (use both triangle halves if general storage).
    rowsum = [0.0] * rows
    diag = {}
    for i, j, v in nz:
        rowsum[i] += v
        rowsum[j] += v
        if i == j:
            diag[i] = diag.get(i, 0.0) + v
    # 3. Emit M.mtx (symmetric, use lower triangle only).
    out = []
    seen = set()
    eps = args.eps
    for i, j, v in nz:
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        out.append((i, j, -args.scale * eps * v))
    for i in range(rows):
        out.append((i, i, args.scale * (1.0 + eps * rowsum[i])))
    with args.mfile.open('w', encoding='utf-8') as f:
        f.write('%%MatrixMarket matrix coordinate real symmetric\n')
        f.write(f'{rows} {cols} {len(out)}\n')
        f.write('\n'.join(f'{i + 1} {j + 1} {v!r}' for i, j, v in out))
        f.write('\n')
    print(f'wrote {len(out)} records -> {args.mfile} '
          f'(n={rows}, pattern = K, SPD via strict diagonal dominance)')


if __name__ == '__main__':
    main()