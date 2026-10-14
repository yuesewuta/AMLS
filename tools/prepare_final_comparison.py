import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'benchmark/results/optimization_check/final_outputs'
out.mkdir(exist_ok=True)
clean = {'AMLS_NO_ADAPT': None, 'AMLS_NO_MT': None, 'AMLS_ADAPT': None,
         'AMLS_BATCH_OFF': None, 'AMLS_N7_SERIAL': None, 'AMLS_NO_FUSE': None,
         'AMLS_DEBUG_ADAPT': '1', 'AMLS_PROGRESS': '1', 'AMLS_TRACE': None}
cases = [
    ('pair_bisect', 'bisect', {'AMLS_BATCH_OFF': '1', 'AMLS_NO_ADAPT': '1'}),
    ('batch_bisect', 'bisect', {'AMLS_NO_ADAPT': '1'}),
    ('batch_nd', 'nd', {'AMLS_NO_ADAPT': '1'}),
    ('nd_adapt_n7serial', 'nd', {'AMLS_N7_SERIAL': '1'}),
    ('nd_adapt_n7parallel', 'nd', {}),
    ('nd_noadapt_batch', 'nd', {'AMLS_NO_ADAPT': '1'}),
]
# 22 组实验被中断于 boneS10 r1 尾部（运行 17 记录未生成）；重新排程，
# 令重要对照尽早完成。snapshot 已测得，只需补 r2 及其余配置。
entries = []
prior = ['nd_adapt_n7serial', 'nd_adapt_n7parallel', 'batch_nd', 'batch_bisect', 'pair_bisect', 'snapshot_nd']
for matrix, leaf, levels in [('thermomech_TK', '500', '6'), ('boneS10', '223', '12')]:
    # 以每矩阵一个串行组完成所有重复；为将 ND 对照集中，先排 ND 相关，
    # 再排 bisect 对，最后是 snapshot 重复。
    bycase = {label: [] for label, _, _ in cases}
    for repeat in range(1, 3):
        ordered = cases if repeat == 1 else list(reversed(cases))
        for label, partition, overrides in ordered:
            if label in prior and repeat == 1:
                continue
            bycase[label].append((repeat, partition, overrides))
    for label, items in bycase.items():
        for repeat, partition, overrides in items:
            name = f'{matrix}_{label}_r{repeat}'
            exe = 'build/amls_solve.exe'
            if label == 'snapshot_nd':
                exe = 'benchmark/results/optimization_check/binaries/amls_static_snapshot.exe'
            argv = [exe, '--K', f'benchmark/matrices/{matrix}/{matrix}.mtx',
                    '--mass-scale', '1', '--fuser', '0.2', '--eta', '4',
                    '--threads', '16', '--quiet', '--partition', partition,
                    '--nlevels', levels, '--leaf-size', leaf, '--out',
                    f'benchmark/results/optimization_check/final_outputs/{name}.txt']
            entries.append({'name': name, 'argv': argv, 'env': dict(clean, **overrides)})
(root / 'benchmark/results/optimization_check/final_commands2.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
print(f'Prepared {len(entries)} sequential comparisons (resume)')
