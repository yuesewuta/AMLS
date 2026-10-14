import json
from pathlib import Path
import numpy as np
p = Path('benchmark/results/optimization_check/controlled_runs/summary.json')
if not p.exists(): raise SystemExit('no completed runs')
runs = json.loads(p.read_text())['runs']
for matrix in ('thermomech_TK', 'boneS10'):
    subset = [r for r in runs if r['name'].startswith(matrix) and r['exitcode'] == 0]
    if len(subset) >= 2:
        a, b = [np.loadtxt(r['argv'][r['argv'].index('--out') + 1], ndmin=2)[:, 1] for r in subset[:2]]
        delta = np.max(np.abs(a-b))
        scaled = np.max(np.abs(a-b)/np.maximum(np.abs(a), 1e-12))
        print(matrix, 'same_tree_eigenvalues', len(a), 'max_absolute', delta, 'max_scaled', scaled)
