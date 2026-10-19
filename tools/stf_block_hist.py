#!/usr/bin/env python3
"""stf 块尺寸直方图：统计 ABA_PAMS 导出 stf 的超节点块大小分布。
用法: python stf_block_hist.py a.stf [b.stf ...]
"""
import array, collections, sys


def read_stf(path):
    d = open(path, 'rb').read()
    head = array.array('I')
    head.frombytes(d[:24])
    nrow = head[0]
    nnz = head[1]
    need = 7 + 3 * nrow + nnz
    a = array.array('I')
    a.frombytes(d[:4 * need])
    assert len(a) >= need, f'stf truncated: {len(a)} ints < {need}'
    rp = a[6:6 + nrow + 1]
    assert rp[nrow] == nnz, f'rowptr end {rp[nrow]} != nnz {nnz}'
    return nrow, rp


def block_sizes(nrow, rp):
    nnzp = [rp[i + 1] - rp[i] for i in range(nrow)]
    sizes = []
    start = 0
    for i in range(1, nrow):
        if nnzp[i] != nnzp[i - 1] - 1:
            sizes.append(i - start)
            start = i
    sizes.append(nrow - start)
    return sizes


def main():
    for path in sys.argv[1:]:
        nrow, rp = read_stf(path)
        sizes = block_sizes(nrow, rp)
        hist = collections.Counter(sizes)
        total = sum(sizes)
        non3 = sum(c for s, c in hist.items() if s % 3 != 0)
        print(f'== {path}')
        print(f'   nrow={nrow}  blocks={len(sizes)}  '
              f'rows_in_blocks={total}')
        print(f'   hist={dict(sorted(hist.items()))}')
        bad = {s: c for s, c in sorted(hist.items()) if s % 3 != 0}
        if bad:
            print(f'   !! 非 3 倍数块: {bad}')
        else:
            print('   全部块为 3 的倍数')


if __name__ == '__main__':
    main()
