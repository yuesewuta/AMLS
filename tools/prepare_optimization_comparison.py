import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = ['build/amls_solve.exe', '--mass-scale', '1', '--fuser', '0.2', '--eta', '4', '--threads', '16', '--quiet']
clean = {'AMLS_NO_ADAPT': None, 'AMLS_NO_MT': None, 'AMLS_ADAPT': None, 'AMLS_BATCH_OFF': None, 'AMLS_N7_SERIAL': None, 'AMLS_DEBUG_ADAPT': '1', 'AMLS_PROGRESS': '1', 'AMLS_TRACE': None}
entries = []
for matrix, leaf in [('thermomech_TK', '500'), ('boneS10', '500')]:
    cases = [('pair_bisect', 'bisect', dict(AMLS_BATCH_OFF='1', AMLS_NO_ADAPT='1')),
             ('batch_bisect', 'bisect', dict(AMLS_NO_ADAPT='1')),
             ('batch_nd', 'nd', dict(AMLS_NO_ADAPT='1')),
             ('batch_nd_tail', 'nd', {})]
    if matrix == 'boneS10':
        leaf = '223'
    for label, partition, override in cases:
        name = matrix + '_' + label
        args = base + ['--K', f'benchmark/matrices/{matrix}/{matrix}.mtx', '--partition', partition, '--nlevels', '12' if matrix == 'boneS10' else '6', '--leaf-size', leaf, '--out', f'benchmark/results/optimization_check/{name}.txt']
        entries.append({'name': name, 'argv': args, 'env': dict(clean, **override)})
(root / 'benchmark/results/optimization_check/commands.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
