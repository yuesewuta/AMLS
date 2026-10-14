#!/usr/bin/env python3
"""plot_stf_tree_graph.py — cant_31 stf 消去树图（分树/聚类风格）。

362 棵树各自独立布局（叶占宽、父在子重心上方），从左到右按树规模降序排列，
树间留空隙；每棵树整体垂直错位（伪随机 0~35），故树根不在同一高度。
节点圆面积 ∝ 块 DOF；同一棵树的节点与父子边同色（每树一色），
跨树/跨级直连边统一淡红。
"""
import sys, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abaqus_stf_tree import read_stf, blocks_and_tree, depth_of

stf = sys.argv[1] if len(sys.argv) > 1 else 'bench_abaqus/cant_31_stf'
out = sys.argv[2] if len(sys.argv) > 2 else 'docs/abaqus_ams/cant_31_消去树图.png'

nrow, nnz, rp, ci = read_stf(stf)
blocks, parent, anc = blocks_and_tree(nrow, rp, ci)
nb = len(blocks)
depth = np.array(depth_of(nb, parent))          # 距本树根的级数
sizes = np.array([e - s + 1 for s, e in blocks])
children = collections.defaultdict(list)
for j, p in enumerate(parent):
    if p != -1:
        children[p].append(j)
roots = [i for i in range(nb) if parent[i] == -1]

# ---- 树归属与规模 ----
root_of = np.arange(nb)
for i in range(nb):                              # parent[i] > i，倒序上溯
    r = i
    while parent[r] != -1:
        r = parent[r]
    root_of[i] = r
tree_members = collections.defaultdict(list)
for i in range(nb):
    tree_members[root_of[i]].append(i)
tree_width = {r: sum(1 for i in mem if not children[i]) for r, mem in tree_members.items()}
tree_height = {r: max(depth[mem]) for r, mem in tree_members.items()}
order = sorted(tree_members, key=lambda r: -tree_width[r])
ntrees = len(order)
print(f'blocks={nb} trees={ntrees} max tree height={max(tree_height.values())}')

# ---- 逐树布局：叶子顺序占宽（树间留隙），父=子重心；垂直伪随机错位 ----
x = np.zeros(nb)
y = np.zeros(nb)
rng = np.random.default_rng(42)
y_off = {}
slot = 0.0
GAP = 4.0
for t, r in enumerate(order):
    mem = tree_members[r]
    for i in sorted(mem, reverse=True):          # 叶从左到右
        if not children[i]:
            x[i] = slot
            slot += 1.0
    for i in sorted(mem, reverse=True):
        if children[i]:
            x[i] = np.mean([x[c] for c in children[i]])
    off = float(rng.uniform(0, 35))
    y_off[r] = off
    y[mem] = -depth[mem] - off
    slot += GAP
nleaves = slot
print(f'total leaf slots={nleaves:.0f}')

# ---- 颜色：每树一色 ----
cmap = plt.cm.gist_ncar
tree_colors = {r: cmap(0.05 + 0.90 * t / max(1, ntrees - 1)) for t, r in enumerate(order)}
node_colors = [tree_colors[root_of[i]] for i in range(nb)]

# ---- 边 ----
seg_parent, col_parent = [], []
for i in range(nb):
    if parent[i] != -1:
        seg_parent.append([(x[i], y[i]), (x[parent[i]], y[parent[i]])])
        col_parent.append(tree_colors[root_of[i]])
seg_cross = [[(x[i], y[i]), (x[a], y[a])]
             for i in range(nb) for a in anc[i] if a != parent[i]]
print(f'parent edges={len(seg_parent)} cross edges={len(seg_cross)}')

# ---- 画图 ----
fig, ax = plt.subplots(figsize=(40, 20), dpi=150)
ax.add_collection(LineCollection(seg_cross, colors='crimson', linewidths=0.15,
                                 alpha=0.06, zorder=1))
ax.add_collection(LineCollection(seg_parent, colors=col_parent, linewidths=0.3,
                                 alpha=0.75, zorder=2))
ax.scatter(x, y, s=sizes * 7 + 2, c=node_colors, linewidths=0, zorder=3, alpha=0.95)
ax.set_xlim(-3, nleaves + 3)
ax.set_ylim(-max(tree_height.values()) - 40, 4)
ax.set_axis_off()

# 图例
handles = [Line2D([0], [0], marker='o', linestyle='', markersize=np.sqrt(d * 7 + 2),
                  markerfacecolor='0.4', markeredgecolor='none', label=f'{d} dof')
           for d in (3, 6, 9, 12, 24)]
leg1 = ax.legend(handles=handles, title='节点面积 ∝ 块 DOF', loc='upper left',
                 frameon=False, labelspacing=1.1, borderpad=0.8)
ax.add_artist(leg1)
handles2 = [Line2D([0], [0], color='0.4', marker='o', linestyle='', lw=0,
                   markerfacecolor=cmap(0.3), label='节点/父子边：同树同色（362 树各一色）'),
            Line2D([0], [0], color='crimson', lw=1.0, alpha=0.6, label='跨级直连边（含跨树）')]
ax.legend(handles=handles2, loc='upper right', frameon=False)

# 局部放大：最大树的深层
biggest = order[0]
bx = [x[i] for i in tree_members[biggest]]
by = [y[i] for i in tree_members[biggest]]
x0, x1 = min(bx), min(bx) + 60
y0, y1 = -(y_off[biggest] + 167), -(y_off[biggest] + 126)
axins = ax.inset_axes([0.60, 0.40, 0.28, 0.36])
axins.add_collection(LineCollection(seg_cross, colors='crimson', linewidths=0.4,
                                    alpha=0.25, zorder=1))
axins.add_collection(LineCollection(seg_parent, colors=col_parent, linewidths=0.7,
                                    alpha=0.9, zorder=2))
axins.scatter(x, y, s=sizes * 7 + 2, c=node_colors, linewidths=0.3,
              edgecolors='k', zorder=3)
axins.set_xlim(x0, x1); axins.set_ylim(y0, y1)
axins.set_xticks([]); axins.set_yticks([])
axins.set_title('局部放大（最大树深层，面积 ∝ DOF）', fontsize=11)
ax.indicate_inset_zoom(axins, edgecolor='0.2')

ax.set_title(f'Abaqus AMS 消去树（cant_31, {nrow:,} DOF, {nb:,} 块 / {ntrees} 棵树）— '
             '同树同色, 节点面积 ∝ 块 DOF, 红=跨级直连', fontsize=15)
fig.tight_layout()
fig.savefig(out)
print('[ok]', out)
