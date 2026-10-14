import json
import re
import statistics
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / 'benchmark/results/optimization_check/final_runs'
summary = json.loads((folder / 'summary.json').read_text())
results = []
for run in summary['runs']:
    stdout = Path(run['stdout']).read_text(errors='replace')
    stderr = Path(run['stderr']).read_text(errors='replace')
    timing_line = next((line for line in stdout.splitlines() if '# 计时' in line), '')
    timing = {key: float(value) for key, value in re.findall(r'(\w+)=([\d.eE+-]+)', timing_line)}
    stages = {key: {'wall': float(wall), 'cpu': float(cpu)} for key, wall, cpu in re.findall(r'stage=(\w+) done wall_seconds=([\d.]+) cpu_seconds=([\d.]+)', stderr)}
    structure = re.search(r'p=(\d+)\s+树节点=(\d+)', stdout)
    peak = re.search(r'峰值工作集=([\d.]+)GB', stdout)
    residuals = re.search(r'# 残差: 第1阶=([\d.eE+-]+) 第(\d+)阶=([\d.eE+-]+) 末阶=([\d.eE+-]+)', stdout)
    matrix = 'boneS10' if run['name'].startswith('boneS10') else 'thermomech_TK'
    result = {'name': run['name'], 'matrix': matrix,
              'case': run['name'].removeprefix(matrix + '_').rsplit('_r', 1)[0],
              'sha256': run['executable_sha256'], 'exitcode': run['exitcode'],
              'wall': run['wall_seconds'], 'timing': timing, 'stages': stages,
              'p': int(structure[1]) if structure else None,
              'nodes': int(structure[2]) if structure else None,
              'peak_GiB': float(peak[1]) if peak else None,
              'adapt': next((line for line in stderr.splitlines() if '[adapt]' in line), ''),
              'n7': [{'node': int(node), 'n': int(n), 'mt': int(mt), 'threads': int(nt), 'wall': float(w)} for node, n, mt, nt, w in re.findall(r'\[N7\] node=(\d+) n=(\d+) mt=(\d+) threads=(\d+) wall=([\d.]+)', stderr)]}
    final = run.get('final_sample') or {}
    if 'cpu_seconds' in final:
        result['cpu_seconds'] = final['cpu_seconds']
        result['aggregate_CPU_percent'] = 100 * final['cpu_seconds'] / run['wall_seconds'] / 16
    if residuals:
        result['residuals'] = {'first': float(residuals[1]), 'mode': int(residuals[2]), 'at_mode': float(residuals[3]), 'last': float(residuals[4])}
    if run['exitcode'] == 0:
        vals = np.loadtxt(root / run['argv'][run['argv'].index('--out') + 1], ndmin=2)[:, 1]
        ref = np.loadtxt(root / f'benchmark/reference/{matrix}_cpp200.txt', ndmin=2)[:, -1]
        k = min(200, len(vals), len(ref))
        err = np.abs(vals[:k] - ref[:k]) / np.maximum(np.abs(ref[:k]), 1e-30)
        nonzero = np.abs(ref[:k]) > max(1e-12, np.abs(ref[:k]).max() * 1e-12)
        result['accuracy'] = {'count': k, 'relerr200': float(err[-1]), 'max_nonzero_relerr': float(err[nonzero].max())}
    results.append(result)
groups = []
for matrix in ('thermomech_TK', 'boneS10'):
    for case in sorted({r['case'] for r in results if r['matrix'] == matrix}):
        rows = [r for r in results if r['matrix'] == matrix and r['case'] == case]
        g = {'matrix': matrix, 'case': case, 'repeats': len(rows), 'exits': [r['exitcode'] for r in rows]}
        for key in ('wall', 'peak_GiB', 'cpu_seconds', 'aggregate_CPU_percent'):
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                g[key] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
        for key in ('partition', 'phase2', 'reduced', 'backtransform', 'normalize'):
            values = [r['timing'][key] for r in rows if key in r['timing']]
            if values:
                g[key] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
        for stage in ('read', 'solve'):
            values = [r['stages'][stage]['wall'] for r in rows if stage in r['stages']]
            if values:
                g[stage + '_wall'] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
        g['accuracy'] = [r.get('accuracy') for r in rows]
        g['residuals'] = [r.get('residuals') for r in rows]
        groups.append(g)
equivalence = []
for matrix in ('thermomech_TK', 'boneS10'):
    rows = [r for r in results if r['matrix'] == matrix and r['exitcode'] == 0]
    for category, cases in [('bisect', ['pair_bisect', 'batch_bisect']), ('nd', ['snapshot_nd', 'batch_nd', 'nd_adapt_n7serial', 'nd_adapt_n7parallel'])]:
        subset = [r for r in rows if r['case'] in cases]
        if not subset:
            continue
        base = np.loadtxt(root / f"benchmark/results/optimization_check/final_outputs/{subset[0]['name']}.txt", ndmin=2)[:, 1]
        for row in subset[1:]:
            vals = np.loadtxt(root / f"benchmark/results/optimization_check/final_outputs/{row['name']}.txt", ndmin=2)[:, 1]
            delta = np.abs(vals - base)
            equivalence.append({'matrix': matrix, 'tree': category, 'baseline': subset[0]['name'], 'comparison': row['name'], 'count': len(vals), 'max_abs': float(delta.max()), 'max_scaled': float((delta / np.maximum(1.0, np.abs(base))).max())})
analysis = {'completed_runs': len(results), 'runs': results, 'groups': groups, 'same_tree_equivalence': equivalence}
(folder / 'final_analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'completed_runs': len(results), 'groups': groups, 'equivalence': equivalence}, ensure_ascii=False, indent=2))
