#!/usr/bin/env python3
"""Plot per-dept duration scatter from AMLS chrome-trace JSON.

Reads an AMLS trace file (AMLS_TRACE=out.json) whose events carry
args{node, lvl} and dur in microseconds, groups events by formula step
(name), and writes:
  - one scatter plot per step: x = depth from node to ROOT (root=0,
    leaf=max; trace stores lvl leaf=0, so we plot maxLvl - lvl),
    y = duration seconds
  - a per-depth aggregate table (count, max, sum) for every step
  - an overview plot with a shared x-axis that merges all steps' per-depth
    total (sum) durations, so near-root blocking depth is visible at once.

The step/tag split follows the AMLS formulation:
  GE:  N1 (constraint modes u = -K̂ii⁻¹ K_i,a),
       N3 (Mtemp = M_i,a + M̂_ii u),
       N4 (stiffness update  K̄ += K_hi,i u        ),
       N5 (mass update      M̄ += M_hi,i u + uᵀ Mtemp),
       N2.free (coupling column release),
       DONE (dependency/adaptation task, contains no numbers).
  Cond: N7 (local spectrum + mode selection),
        N8 (mass-coupling seed  μ̃),
        N9N10 (fold + propagate),
        freeMtemp/freeDiag/freeU (memory release).
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STEP_LABEL = {
    'N1': 'N1 u=-Kii^-1 K_i,a', 'N3': 'N3 Mtemp=M_i,a+Mii u',
    'N4.batch': 'N4 batch Ke += K_hi,i u', 'N5.batch': 'N5 batch Me += M_hi,i u + u^T Mtemp',
    'N4.coup': 'N4 per-pair coupling', 'N5.coup': 'N5 per-pair coupling',
    'N4.diag': 'N4 per-pair diagonal', 'N5.diag': 'N5 per-pair diagonal',
    'N2.free': 'N2 free coupling col', 'N7': 'N7 local spectrum',
    'N8': 'N8 seed mass coupling', 'N9': 'N9 fold mu~*Z_i',
    'N10': 'N10 propagate mu~ += mu~*u', 'N9N10': 'N9N10 fold+propagate',
    'N8N9N10': 'N8+N9+N10 fused', 'freeMtemp': 'free Mtemp',
    'freeDiag': 'free diag blocks', 'freeU': 'free u cols',
}


def parse_trace(path):
    """Stream the chrome-trace events (one JSON object per line)."""
    events = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line or line in ('{', '[', ']', '}') or line.startswith('{"traceEvents"'):
                continue
            line = line.rstrip(',').rstrip()
            m = re.match(r'^\{.*\}$', line)
            if not m or '"ph":"X"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get('ph') != 'X':
                continue
            dur = float(obj.get('dur', 0)) * 1e-6
            args = obj.get('args', {})
            events.append({
                'cat': obj.get('cat', ''),
                'name': obj.get('name', ''),
                'dur': dur,
                'node': args.get('node', -1),
                'lvl': args.get('lvl', -1),
            })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace', type=Path)
    ap.add_argument('--out', type=Path, default=Path('depth_scatter'))
    ap.add_argument('--ymax', type=float, default=0.0,
                    help='cap shared y-axis for overview (seconds)')
    args = ap.parse_args()

    events = parse_trace(args.trace)
    # lvl in the trace is leaf=0 (depth from leaves upward). The user-facing
    # depth is "distance from node i up to the root": root=0, leaves=maxLevel.
    max_lvl = max(e['lvl'] for e in events if e['lvl'] >= 0) if events else 0
    def depth_of(e):
        return max_lvl - e['lvl'] if e['lvl'] >= 0 else -1
    steps = ['N1', 'N3', 'N4.batch', 'N5.batch', 'N4.coup', 'N5.coup',
             'N4.diag', 'N5.diag', 'N2.free', 'N7', 'N8', 'N9', 'N10',
             'N9N10', 'N8N9N10', 'freeMtemp', 'freeDiag', 'freeU']
    # Keep only step events with a valid depth and duration.
    by_step = defaultdict(list)
    for e in events:
        if e['name'] in steps:
            d = depth_of(e)
            if d >= 0 and e['dur'] >= 0:
                by_step[e['name']].append({**e, 'depth': d})

    args.out.mkdir(parents=True, exist_ok=True)
    # Per-step scatter plots (each event = one point).
    for name, evs in sorted(by_step.items()):
        xs = np.array([e['depth'] for e in evs], dtype=float)
        ys = np.array([e['dur'] for e in evs], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(xs, ys, s=14, alpha=0.55, edgecolors='none')
        ax.set_xlabel('depth from node to root (root=0, leaf=max)')
        ax.set_ylabel('duration (s)')
        ax.set_title(f'{name} — {STEP_LABEL.get(name, name)}  (n={len(evs)})')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / f'{name}.png', dpi=130)
        plt.close(fig)
        print(f'[step] {name}: {len(evs)} events')

    # Per-depth aggregate table per step (sum and count).
    agg_rows = []
    for name in sorted(by_step):
        by_lvl = defaultdict(list)
        for e in by_step[name]:
            by_lvl[e['depth']].append(e['dur'])
        for lvl in sorted(by_lvl):
            durs = by_lvl[lvl]
            agg_rows.append({'step': name, 'depth_to_root': lvl, 'count': len(durs),
                             'sum_s': round(sum(durs), 6),
                             'max_s': round(max(durs), 6)})
    with (args.out / 'depth_aggregate.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step', 'depth_to_root', 'count', 'sum_s', 'max_s'])
        w.writeheader()
        w.writerows(agg_rows)
    print(f'[agg] wrote {len(agg_rows)} rows -> {args.out / "depth_aggregate.csv"}')

    # Overview: per-depth total (sum across steps) to expose blocking depth.
    groups = ['N1', 'N3', 'N4.batch', 'N5.batch', 'N2.free', 'N7',
              'N8', 'N9', 'N10', 'freeMtemp', 'freeDiag']
    depths = sorted({e['depth'] for name in groups for e in by_step.get(name, [])})
    sums = {name: defaultdict(float) for name in groups}
    cnts = {name: defaultdict(int) for name in groups}
    for name in groups:
        for e in by_step.get(name, []):
            sums[name][e['depth']] += e['dur']
            cnts[name][e['depth']] += 1
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(depths))
    for name in groups:
        series = np.array([sums[name].get(d, 0.0) for d in depths])
        counts = np.array([cnts[name].get(d, 0) for d in depths])
        label = STEP_LABEL.get(name, name)
        if np.any(series > 0):
            ax.bar(depths, series, bottom=bottom, width=0.82,
                   label=f'{label} (n={int(counts.sum())})', alpha=0.9)
            bottom += series
    ax.set_xlabel('depth from node to root (root=0, leaf=max)')
    ax.set_ylabel('total step duration (s)')
    ax.set_title('Per-depth total duration by formula step (stacked)')
    if args.ymax > 0:
        ax.set_ylim(0, args.ymax)
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / 'overview_stacked.png', dpi=140)
    plt.close(fig)

    with (args.out / 'depth_totals.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['depth_to_root'] + groups)
        w.writeheader()
        for i, d in enumerate(depths):
            row = {'depth_to_root': d}
            for name in groups:
                row[name] = round(sums[name].get(d, 0.0), 6)
            w.writerow(row)
    print(f'[overview] wrote -> {args.out / "overview_stacked.png"} '
          f'{args.out / "depth_totals.csv"}')


if __name__ == '__main__':
    main()