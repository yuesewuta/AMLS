import csv
import json
import re
from pathlib import Path
import numpy as np
import sys

folder = Path(sys.argv[1])
summary = json.loads((folder / 'summary.json').read_text())
results = []
for run in summary['runs']:
    stdout = Path(run['stdout']).read_text(errors='replace')
    stderr = Path(run['stderr']).read_text(errors='replace')
    timing = dict((key, float(value)) for key, value in re.findall(r'(\w+)=([\d.eE+-]+)', next((line for line in stdout.splitlines() if '# 计时' in line), '')))
    completed = re.search(r'p=(\d+)\s+树节点=(\d+)', stdout)
    memory = re.search(r'峰值工作集=([\d.]+)GB', stdout)
    stages = {key: {'wall': float(wall), 'cpu': float(cpu)} for key, wall, cpu in re.findall(r'stage=(\w+) done wall_seconds=([\d.]+) cpu_seconds=([\d.]+)', stderr)}
    result = {'name': run['name'], 'exitcode': run['exitcode'], 'process_wall': run['wall_seconds'], 'timing': timing, 'stages': stages, 'p': int(completed[1]) if completed else None, 'nodes': int(completed[2]) if completed else None, 'peak_GB': float(memory[1]) if memory else None}
    argv = run['argv']
    if run['exitcode'] == 0 and '--out' in argv:
        output = Path(argv[argv.index('--out') + 1])
        vals = np.loadtxt(output, ndmin=2)[:, 1]
        name = 'boneS10' if 'boneS10' in run['name'] else 'thermomech_TK'
        ref = np.loadtxt(f'benchmark/reference/{name}_cpp200.txt', ndmin=2)[:, -1]
        k = min(200, len(vals), len(ref))
        rel = np.abs(vals[:k] - ref[:k]) / np.maximum(np.abs(ref[:k]), 1e-30)
        result['reference'] = {'count': k, 'relerr_200': float(rel[-1]), 'maxrelerr_200': float(rel.max()), 'maxrelerr_nonzero': float(rel[np.abs(ref[:k]) > max(1e-12, np.max(np.abs(ref[:k])) * 1e-12)].max()), 'mu1': float(vals[0]), 'mu200': float(vals[k-1])}
    result['n7'] = [{'node': int(node), 'size': int(n), 'mt': int(mt), 'threads': int(threads), 'wall': float(wall)} for node, n, mt, threads, wall in re.findall(r'\[N7\] node=(\d+) n=(\d+) mt=(\d+) threads=(\d+) wall=([\d.]+)', stderr)]
    result['adapt'] = next((line for line in stderr.splitlines() if '[adapt]' in line), '')
    results.append(result)
(folder / 'analysis.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(json.dumps(results, indent=2, ensure_ascii=False))
