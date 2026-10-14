#!/usr/bin/env python3
"""abaqus_stf_tree.py — 从 Abaqus ABA_PAMS_EXPORT_SOLVER_DATA 导出的 stf
（AMS 超节点行序稀疏算子）重建消去树，输出带节点规模与连接关系注释的 markdown。

stf 二进制格式（实测，int32 小端，cant_20/cant_31 校验）:
  a[0]=nrow, a[1]=nnz, a[2..5]=元数据,
  a[6..6+nrow]=rowptr(nrow+1 项, nnzp),
  a[6+nrow+1 .. 6+nrow+1+2*nrow]=nrow 个 8 字节记录（每行 2×int32，对角相关，树重建不用）,
  其后 colidx(int32×nnz)。
  colidx 为**不含自身对角**的列号（0 基，行内升序；第 r 行首列 ≥ r+1），
  空行（nnzp 差 0）在流中不占位。

超节点块识别：块内行度逐行减 1（nnzp[i] == nnzp[i-1]-1），否则新块开始。
消去树父子：块 i 的行里出现的大于块末列的最小列号所在块 = parent(i)；
全部大于块末列的列号所在块 = 祖先耦合集合（含跨级直连）。
"""
import array, collections, sys, os

def read_stf(path):
    d = open(path, 'rb').read()
    head = array.array('I'); head.frombytes(d[:24])
    nrow = head[0]; nnz = head[1]
    need = 7 + 3*nrow + nnz
    a = array.array('I'); a.frombytes(d[:4*need])
    assert len(a) >= need, f'stf truncated: {len(a)} ints < {need}'
    rp = a[6:6+nrow+1]
    assert rp[nrow] == nnz, f'rowptr end {rp[nrow]} != nnz {nnz}'
    off = 7 + 3*nrow      # 6 头 + (nrow+1) rowptr + nrow 个 8 字节记录
    ci = a[off:off+nnz]
    assert min(ci) >= 0 and max(ci) < nrow
    return nrow, nnz, rp, ci

def blocks_and_tree(nrow, rp, ci):
    nnzp = [rp[i+1]-rp[i] for i in range(nrow)]
    blocks = []  # (start,end)
    start = 0
    for i in range(1, nrow):
        if nnzp[i] != nnzp[i-1] - 1:
            blocks.append((start, i-1)); start = i
    blocks.append((start, nrow-1))
    nb = len(blocks)
    blk_of_row = array.array('i', [0])*0
    blk_of_row = array.array('i', bytes(4*nrow))
    for bi, (s, e) in enumerate(blocks):
        for r in range(s, e+1):
            blk_of_row[r] = bi
    parent = [-1]*nb
    anc = [set() for _ in range(nb)]     # 祖先耦合块（非父也算）
    for bi, (s, e) in enumerate(blocks):
        cols = set()
        for r in range(s, e+1):
            cols.update(ci[rp[r]:rp[r+1]])
        up = sorted(c for c in cols if c > e)
        if not up:
            continue
        seen = set()
        for c in up:
            ab = blk_of_row[c]
            if ab in seen:
                continue
            seen.add(ab)
            anc[bi].add(ab)
        parent[bi] = blk_of_row[up[0]]
    return blocks, parent, anc

def depth_of(nb, parent):
    depth = [0]*nb
    for i in range(nb):
        d = 0; j = i
        while parent[j] != -1:
            j = parent[j]; d += 1
        depth[i] = d
    return depth

def main(stf_path, out_md, model_note=''):
    nrow, nnz, rp, ci = read_stf(stf_path)
    blocks, parent, anc = blocks_and_tree(nrow, rp, ci)
    nb = len(blocks)
    depth = depth_of(nb, parent)
    sizes = [e-s+1 for s, e in blocks]
    ndepth = max(depth)+1
    by_level = collections.Counter(depth)
    size_hist = collections.Counter(sizes)
    kids = collections.Counter(p for p in parent if p != -1)

    L = []
    L.append('# Abaqus 10万 DOF 悬臂梁 AMS 消去树（cant_31, 98,304 DOF）\n')
    if model_note:
        L.append(model_note + '\n')
    L.append('## 1. 数据来源与重建方法\n')
    L.append(f'- 算子：`ABA_PAMS_EXPORT_SOLVER_DATA` 探针导出的 `cant_31_stf`'
             f'（AMS 超节点行序刚度算子，已含多级缩聚后的跨级耦合），'
             f'nrow={nrow:,}，nnz={nnz:,}（不含对角，行内列号升序，第 r 行首列 ≥ r+1）。')
    L.append('- 超节点块识别：块内行度逐行 −1（`nnzp[i]==nnzp[i-1]-1`），'
             '行度跳变处即块边界（与 cant_20 重建同法）。')
    L.append('- 父子关系：块 i 行内出现的大于块末列的最小列号所在块为 parent(i)'
             '（消去树标准规则）；行内全部更高列号所在块构成祖先耦合集，'
             '含**跨级直连**（不经过父块）。\n')
    L.append('## 2. 总体形状\n')
    L.append('| 指标 | 值 |')
    L.append('|---|---|')
    L.append(f'| DOF | {nrow:,} |')
    L.append(f'| 超节点块数 | {nb:,} |')
    L.append(f'| 最长根叶链（级数） | {ndepth} |')
    L.append(f'| 顶层块数（森林根，无父块） | {sum(1 for x in parent if x==-1):,} |')
    L.append(f'| 块规模 min/中位/max | {min(sizes)} / {sorted(sizes)[nb//2]} / {max(sizes)} |')
    L.append(f'| 平均子块数（有子块的节点） | {sum(kids.values())/max(1,len(kids)):.2f} |')
    L.append(f'| 最大扇入（一个块的子块数） | {max(kids.values())} |')
    L.append(f'| 有跨级直连（祖先≠父）的块数 | '
             f'{sum(1 for i in range(nb) if any(a != parent[i] for a in anc[i])):,} |\n')
    L.append('## 3. 逐层规模表\n')
    L.append('| 距顶层级数 | 块数 | 块规模范围 | 该层 DOF 合计 |')
    L.append('|---|---|---|---|')
    for d in range(ndepth):
        idx = [i for i in range(nb) if depth[i] == d]
        ss = [sizes[i] for i in idx]
        L.append(f'| {d} | {len(idx):,} | {min(ss)}–{max(ss)} | {sum(ss):,} |')
    L.append('')
    L.append('## 4. 块规模分布\n')
    L.append('| 块规模（行数） | 块数 | 占比 |')
    L.append('|---|---|---|')
    for s in sorted(size_hist):
        c = size_hist[s]
        L.append(f'| {s} | {c:,} | {100*c/nb:.1f}% |')
    L.append('')
    L.append('## 5. 树形示意\n')
    children = {i: [] for i in range(nb)}
    for j, p in enumerate(parent):
        if p != -1:
            children[p].append(j)
    for i in children:
        children[i].sort(key=lambda c: -(sizes[c]))

    def annotate(b):
        s, e = blocks[b]
        coup = sum(1 for a in anc[b] if a != parent[b] and a > b)
        extra = f'，子块 {len(children[b])}' if children[b] else ''
        note = f'，↑直连更高块 {coup}' if coup else ''
        return f'[块{b}] 行{s}–{e}（{e-s+1} dof, 距顶层{depth[b]}级{extra}{note}）'

    root = nb - 1
    L.append('### 5.1 顶层展开（自最高层块向下，至 15 级以内，全部块）\n')
    L.append('```')
    L.append(annotate(root))
    shown_box = [0]
    def render_down(b, indent):
        if depth[b] > 15 or shown_box[0] > 60:
            return
        for c in children[b]:
            if shown_box[0] > 60 or depth[c] > 15:
                continue
            shown_box[0] += 1
            L.append(indent + '├─ ' + annotate(c))
            render_down(c, indent + '│  ')
    render_down(root, '')
    L.append('```')
    L.append('> 块编号按行序递增（叶→根）；「↑直连更高块」= 该块除父块外还直接耦合到的'
             '编号更大（更顶层）的块数，即跨级耦合宽度。近顶层区近于单链；'
             '中深层每层块数庞大（§3），无法全画。\n')

    # 叶→根链示例：取首块（最深的叶链之一）
    L.append('### 5.2 叶→根传播链示例（块 0 的父链）\n')
    L.append('```')
    chain = []
    j = 0
    while j != -1:
        chain.append(j); j = parent[j]
    L.append(f'链长 {len(chain)} 块（含根），每步上升一级（中间层省略）：')
    show_head, show_tail = 8, 5
    printed = set()
    for k, b in enumerate(chain):
        if k < show_head or k >= len(chain) - show_tail:
            printed.add(k)
            L.append('  ' + '  '*min(k, show_head+show_tail+1) + '└─ ' + annotate(b))
        elif k == show_head:
            L.append('    ……（省略 %d 块）……' % (len(chain) - show_head - show_tail))
    L.append('```')
    L.append('> AMS 恢复阶段沿此链自顶向下折叠：X_i = Φ_i·z_i + Σ_j Ψ_ij·X_j；'
             '链上每个块还可能直连链外更高块（§6），那些 Ψ 项同样要折叠'
             '（对应 PKU 移植中"全祖先对折叠"的教训）。\n')

    L.append('## 6. 连接关系统计\n')
    def bucket(d):
        if d <= 5:
            return str(d)
        if d <= 10:
            return '6–10'
        if d <= 20:
            return '11–20'
        if d <= 30:
            return '21–30'
        if d <= 40:
            return '31–40'
        if d <= 57:
            return '41–57'
        return '>57'
    nanc = collections.Counter()
    for i in range(nb):
        aa = sum(1 for a in anc[i] if a != parent[i] and a > i)
        nanc[bucket(aa)] += 1
    order = ['0','1','2','3','4','5','6–10','11–20','21–30','31–40','41–57','>57']
    L.append('### 6.1 每块的非父直接耦合数（除父块外还耦合到的更高块数）\n')
    L.append('| 非父耦合数 | 块数 | 占比 |')
    L.append('|---|---|---|')
    for k in order:
        if k in nanc:
            c = nanc[k]
            L.append(f'| {k} | {c:,} | {100*c/nb:.1f}% |')
    L.append('')

    up = collections.Counter(); down = collections.Counter()
    for i in range(nb):
        for a in anc[i]:
            if a == parent[i] or a <= i:
                continue
            d = depth[i] - depth[a]   # 正 = a 更靠近根
            tgt = up if d > 0 else (down if d < 0 else collections.Counter())
            if d == 0:
                continue
            tgt[bucket(abs(d))] += 1
    def emit_dir(name, cnt):
        L.append(f'#### {name}\n')
        L.append('| 层距 | 耦合对数 |')
        L.append('|---|---|')
        for k in order:
            if k in cnt:
                L.append(f'| {k} | {cnt[k]:,} |')
        L.append('')
    L.append('### 6.2 跨级直接耦合的层距分布\n')
    L.append('（只统计父块之外的直接耦合；层距 = 双方距顶层级数之差）\n')
    emit_dir('耦合到更靠近根的块（跨级上传方向）', up)
    emit_dir('耦合到更深的块（同类兄弟/远缘分隔符）', down)
    n0 = sum(1 for i in range(nb) if not any(a != parent[i] and a > i for a in anc[i]))
    L.append(f'> {n0:,} 个块（{100*n0/nb:.1f}%）无非父耦合（纯父子链）；其余 {nb-n0:,} 个块都有跨级直连，'
             '且主要指向距自己 1–10 层的更高块——AMS 的多级缩聚把远端耦合压进了'
             f'少数高层分隔符块。恢复阶段对所有祖先对折叠是必要的（实测 {nb-n0:,} 块有此需求）。\n')
    open(out_md, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'[ok] blocks={nb} depth={ndepth} -> {out_md}')
    print('size hist:', dict(sorted(size_hist.items())[:10]), '...')
    print('per-level:', {d: by_level[d] for d in range(ndepth)})

if __name__ == '__main__':
    stf = sys.argv[1] if len(sys.argv) > 1 else 'bench_abaqus/cant_31_stf'
    out = sys.argv[2] if len(sys.argv) > 2 else 'docs/abaqus_ams/cant_31_消去树.md'
    main(stf, out)
