#!/usr/bin/env python3
"""plot_stf_tree_plotly.py — cant_31 stf 消去树交互式 plotly 页面（聚类式散布版）。

布局：
  1) 每棵树独立「径向树布局」：根在簇中心，叶子按角度扇分，节点半径=距根级数（+抖动）；
  2) 全局按树半径贪心圆 packing（绕轨搜索），簇间留大间距；
  3) 只画父子边；同树同色（黄金角 HLS 调色板，色相拉满、相邻树色差最大）；
     根节点用菱形与普通圆形节点区分；节点面积 ∝ 块 DOF。
交互：
  - 悬停显示块信息；WebGL 渲染；滚轮缩放/拖拽平移；
  - 左上「按树筛选」下拉：选树 → 该树保持高亮、其余淡化到 5% 透明度。
输出独立 HTML（plotly.js 走 CDN）。
"""
import sys, os, collections, colorsys
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abaqus_stf_tree import read_stf, blocks_and_tree, depth_of

stf = sys.argv[1] if len(sys.argv) > 1 else 'bench_abaqus/cant_31_stf'
out = sys.argv[2] if len(sys.argv) > 2 else 'docs/abaqus_ams/cant_31_tree.html'

GAP = 12.0
rng = np.random.default_rng(7)

nrow, nnz, rp, ci = read_stf(stf)
blocks, parent, anc = blocks_and_tree(nrow, rp, ci)
nb = len(blocks)
depth = np.array(depth_of(nb, parent))
sizes = np.array([e - s + 1 for s, e in blocks])
ncoup = np.array([sum(1 for a in anc[i] if a != parent[i]) for i in range(nb)])
children = collections.defaultdict(list)
for j, p in enumerate(parent):
    if p != -1:
        children[p].append(j)
root_of = np.arange(nb)
for i in range(nb):
    r = i
    while parent[r] != -1:
        r = parent[r]
    root_of[i] = r
tree_members = collections.defaultdict(list)
for i in range(nb):
    tree_members[root_of[i]].append(i)
ntrees = len(tree_members)
print(f'blocks={nb} trees={ntrees}')

# ---- 1) 每树径向布局 ----
x = np.zeros(nb); y = np.zeros(nb)
tree_R = {}
for r, mem in tree_members.items():
    nleaf = sum(1 for i in mem if not children[i])
    ang = {}
    s = 0
    for i in mem:
        if not children[i]:
            ang[i] = 2 * np.pi * (s + 0.5) / nleaf
            s += 1
    for i in sorted(mem, key=lambda j: -depth[j]):
        if children[i]:
            ang[i] = np.mean([ang[c] for c in children[i]])
    for i in mem:
        rad = depth[i] * (1.0 + rng.normal(0, 0.02))
        a = ang[i] + rng.normal(0, 0.02 / max(depth[i], 1))
        x[i] = rad * np.cos(a)
        y[i] = rad * np.sin(a)
    tree_R[r] = max(depth[mem]) + 2.0

# ---- 2) 全局圆 packing（绕轨搜索） ----
order = sorted(tree_members, key=lambda r: -tree_R[r])
center = {}
placed_c = []
pc = pr = None
N_ANG = 64
for t, r in enumerate(order):
    R = tree_R[r]
    if pc is None:
        cx, cy = 0.0, 0.0
    else:
        cx = cy = None
        for ci in range(len(placed_c)):
            d = placed_c[ci][2] + R + GAP
            angs = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False) + 0.13 * t
            cand = np.stack([placed_c[ci][0] + d * np.cos(angs),
                             placed_c[ci][1] + d * np.sin(angs)], axis=1)
            dist = np.hypot(pc[:, 0][:, None] - cand[:, 0],
                            pc[:, 1][:, None] - cand[:, 1])
            good = np.all(dist >= (pr + R + GAP)[:, None], axis=0)
            if good.any():
                cx, cy = cand[np.argmax(good)]
                break
        assert cx is not None, f'packing failed at tree #{t} R={R}'
    center[r] = (cx, cy)
    placed_c.append((cx, cy, R))
    pc = np.array([(c[0], c[1]) for c in placed_c])
    pr = np.array([c[2] for c in placed_c])
for r, mem in tree_members.items():
    x[mem] += center[r][0]
    y[mem] += center[r][1]
print(f'packing done, extent ±{max(max(abs(c[0]), abs(c[1])) + c[2] for c in placed_c):.0f}')

# ---- 3) 颜色：黄金角 HLS，相邻树色相差 ~137°，明暗交替 ----
tree_hex = []
for t in range(ntrees):
    h = (t * 137.508) % 360 / 360.0
    l = 0.42 if t % 2 == 0 else 0.58
    rgb = colorsys.hls_to_rgb(h, l, 0.75)
    tree_hex.append('#%02x%02x%02x' % tuple(int(c * 255) for c in rgb))
tree_of = {r: t for t, r in enumerate(order)}

# ---- 4) traces：每树 边+节点 两条；节点 trace 内根节点用菱形 ----
fig = go.Figure()
tree_trace_idx = {}    # tree t -> [edge_trace_idx, node_trace_idx]
for t, r in enumerate(order):
    mem = tree_members[r]
    exs, eys = [], []
    for i in mem:
        if parent[i] != -1:
            exs += [x[i], x[parent[i]], None]
            eys += [y[i], y[parent[i]], None]
    fig.add_trace(go.Scattergl(x=exs, y=eys, mode='lines',
                               line=dict(width=1.2, color=tree_hex[t]),
                               opacity=0.7, hoverinfo='skip', showlegend=False))
    hover = [f'块{i}（树{t}{"，根" if i == r else ""}）<br>'
             f'行 {blocks[i][0]}–{blocks[i][1]}（{sizes[i]} dof）<br>'
             f'距顶层 {depth[i]} 级 · 非父直连 {ncoup[i]}'
             for i in mem]
    sym = ['diamond' if i == r else 'circle' for i in mem]
    msz = [(sizes[i] * 2.2 + 2) * (1.8 if i == r else 1.0) for i in mem]
    fig.add_trace(go.Scattergl(x=x[mem], y=y[mem], mode='markers', customdata=hover,
                               marker=dict(size=msz, symbol=sym,
                                           color=tree_hex[t], opacity=0.95,
                                           line=dict(width=0)),
                               hovertemplate='%{customdata}<extra></extra>',
                               showlegend=False, name=f'树{t}'))
    tree_trace_idx[t] = [len(fig.data) - 2, len(fig.data) - 1]
print('traces done:', len(fig.data))

# ---- 5) 按树筛选下拉 ----
ntr = len(fig.data)
buttons = [dict(method='restyle',
                label='显示全部',
                args=[{'opacity': [1.0] * ntr}, list(range(ntr))])]
for t, r in enumerate(order):
    ops = [0.06] * ntr
    for idx in tree_trace_idx[t]:
        ops[idx] = 1.0
    buttons.append(dict(method='restyle', label=f'树{t}（{len(tree_members[r])}块）',
                        args=[{'opacity': ops}, list(range(ntr))]))
fig.update_layout(
    updatemenus=[dict(type='dropdown', buttons=buttons, x=0.0, xanchor='left',
                      y=1.12, yanchor='top', showactive=True,
                      pad=dict(t=2, b=2))])
fig.update_layout(
    title=f'Abaqus AMS 消去树（cant_31, {nrow:,} DOF, {nb:,} 块 / {ntrees} 棵树）— '
          '径向簇 + 圆 packing，同树同色（菱形=根），面积 ∝ 块 DOF',
    template='plotly_white',
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    margin=dict(l=10, r=10, t=70, b=10),
)
fig.update_yaxes(scaleanchor='x', scaleratio=1)
fig.write_html(out, include_plotlyjs='cdn')
print('[ok]', out, f'{os.path.getsize(out)/1e6:.1f} MB')
