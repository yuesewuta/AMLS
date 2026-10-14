#!/usr/bin/env python3
"""Abaqus *MATRIX OUTPUT, FORMAT=COORDINATE .mtx → Matrix Market 转换.

实测 Abaqus 2020 COORDINATE 文本格式（每行一个条目, 空白分隔, 有全矩阵两半）:
    i j value            # 例如 "1 2  1.318108974358976e+10"

Abaqus 特点（本转换器已处理）:
  1. 显式输出全矩阵两半（(1,2) 和 (2,1) 都写）, 且矩阵对称 → 去重保留上三角.
  2. K 中被边界条件约束的 DOF 对角线被写成 1e36 哨兵值 (Abaqus 大数法),
     M 不受影响. 默认把哨兵(=1e36 或其 MOR 变体)行/列删除, 得到约束后的
     子矩阵; 哨兵值也可用阈值参数调整.
  3. 删除行/列后索引会跳过空档, 重新编号为 1..N 连续 (Matrix Market 要求).
     可用 --remap 关闭 (保留原全局编号, 但 N 用最大下标).

输出 Matrix Market coordinate 格式（amls_solve 可读）:
    %%MatrixMarket matrix coordinate real symmetric
    N N NNZ
    i j value

用法:
  abaqus_mtx_to_mm.py K_STIF.mtx K.mtx --write-mask kmask.txt   # 记录删掉的 DOF
  abaqus_mtx_to_mm.py M_MASS.mtx M.mtx --read-mask kmask.txt    # M 复用同一约束集合
  abaqus_mtx_to_mm.py X.mtx X.mtx --keep-sentinel --sym general
"""
import argparse
import re
import sys

SENTINEL = 1e36
THRESHOLD = 1e30  # > 1e30 视为哨兵


def read_triplets(inp: str):
    """读取 COORDINATE 文本, 返回 (maxindex, [(i,j,v)...]) 1-based."""
    n = 0
    rows = []
    with open(inp, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("!"):
                continue
            p = re.split(r"[,\s]+", line)
            if len(p) < 3:
                continue
            try:
                r, c, v = int(p[0]), int(p[1]), float(p[2])
            except ValueError:
                continue
            n = max(n, r, c)
            rows.append((r, c, v))
    return n, rows


def convert(inp: str, outp: str, sym: str, drop_sentinel: bool,
            dedup: bool, remap: bool, mask_file: str = None,
            write_mask: str = None) -> None:
    n, rows = read_triplets(inp)
    if not rows:
        sys.stderr.write(f"[mtx] 未解析到任何条目: {inp}\n")
        sys.exit(2)

    # 哨兵集合: 对角线 > THRESHOLD 的下标 (1-based); 或用外部 mask 文件
    if mask_file:
        with open(mask_file, "r", encoding="utf-8") as f:
            sentinel_do = {int(x) for x in f.read().split()}
    else:
        sentinel_do = set()
        if drop_sentinel:
            diag = {r: v for r, c, v in rows if r == c}
            sentinel_do = {r for r, v in diag.items() if abs(v) > THRESHOLD}
    if not drop_sentinel:
        sentinel_do = set()

    # 过滤 + 去重 (保留上三角: 只留 r<=c)
    kept = []
    seen = set()
    for r, c, v in rows:
        if drop_sentinel and (r in sentinel_do or c in sentinel_do):
            continue
        if dedup:
            if r > c:
                r, c = c, r
            if (r, c) in seen:
                continue
            seen.add((r, c))
        kept.append((r, c, v))
    if dedup:
        kept.sort(key=lambda t: (t[0], t[1]))

    # 重映射 (默认): 连续编号 1..N。Matrix Market 要求下标 1..N, 且
    #   amls_solve 按对称展开, 编号必须无空档。
    if remap:
        old = sorted(set(r for r, c, v in kept) | set(c for r, c, v in kept))
        mp = {o: i + 1 for i, o in enumerate(old)}
        kept = [(mp[r], mp[c], v) for r, c, v in kept]
        final_n = len(old)
    else:
        final_n = n - len(sentinel_do) if not dedup else \
            max((max(r, c) for r, c, v in kept), default=0)

    with open(outp, "w", encoding="utf-8") as f:
        f.write(f"%%MatrixMarket matrix coordinate real {sym}\n")
        f.write(f"{final_n} {final_n} {len(kept)}\n")
        for r, c, v in kept:
            f.write(f"{r} {c} {v:.16e}\n")
    if write_mask:
        with open(write_mask, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in sorted(sentinel_do)) + "\n")
        sys.stderr.write(f"[mtx] mask -> {write_mask}: {len(sentinel_do)} DOF\n")
    sys.stderr.write(
        f"[mtx] {inp} -> {outp}: n={n} sentinel_dropped={len(sentinel_do)} "
        f"final_n={final_n} nnz={len(kept)} sym={sym}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--sym", choices=["symmetric", "general"], default="symmetric")
    ap.add_argument("--keep-sentinel", action="store_true",
                    help="不删除 1e36 哨兵行 (默认删除)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="不去重两半 (默认去重保留上三角)")
    ap.add_argument("--keep-labels", action="store_true",
                    help="不重映射为连续编号 (默认重映射)")
    ap.add_argument("--read-mask", metavar="FILE",
                    help="从 K 转换生成的 mask 文件读约束 DOF 集合")
    ap.add_argument("--write-mask", metavar="FILE",
                    help="把删掉的哨兵 DOF 写入 mask 文件 (供 M 复用)")
    a = ap.parse_args()
    convert(a.inp, a.out, a.sym,
            drop_sentinel=not a.keep_sentinel,
            dedup=not a.no_dedup,
            remap=not a.keep_labels,
            mask_file=a.read_mask,
            write_mask=a.write_mask)


if __name__ == "__main__":
    main()