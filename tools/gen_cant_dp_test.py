#!/usr/bin/env python3
"""生成两类非 dp=3 模型的 Abaqus inp，用于验证 AMS 超节点块原子=网格节点 DOF 数：
  cant_shell.inp   — S4R 壳单元悬臂板（6 DOF/节点 → dp=6 预期）
  cant_plane2d.inp — CPS4 平面应力悬臂梁（2 DOF/节点 → dp=2 预期）
用法: python tools/gen_cant_dp_test.py  (输出到 bench_abaqus/)
"""
import os


def emit_shell(path, nx=40, nw=4, L=100.0, W=10.0):
    """S4R 悬臂板: 网格 (nx+1)*(nw+1) 节点, 6 DOF/节点."""
    out = ['*HEADING', 'cantilever plate, S4R shell, 6 dof per node', '*NODE']
    nid = lambda i, j: j * (nx + 1) + i + 1
    for j in range(nw + 1):
        for i in range(nx + 1):
            out.append(f'{nid(i,j)}, {i*L/nx:.6f}, {j*W/nw:.6f}, 0.0')
    out.append('*ELEMENT, TYPE=S4R, ELSET=EALL')
    for j in range(nw):
        for i in range(nx):
            out.append(f'{j*nx+i+1}, {nid(i,j)}, {nid(i+1,j)}, '
                       f'{nid(i+1,j+1)}, {nid(i,j+1)}')
    out += ['*NSET, NSET=fixed']
    out.append(', '.join(str(nid(0, j)) for j in range(nw + 1)))
    out += [
        '*MATERIAL, NAME=steel', '*ELASTIC', '210e9, 0.3', '*DENSITY',
        '7800.0', '*SHELL SECTION, ELSET=EALL, MATERIAL=steel', '1.0, 5', '*BOUNDARY',
        'fixed, 1, 6', '*STEP, NAME=freq', '*FREQUENCY, EIGENSOLVER=AMS',
        '6, 0.0, 200.0', '*END STEP', '*END INPUT',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    n = (nx + 1) * (nw + 1)
    print(f'{path}: {n} nodes, {6*n} dof')


def emit_plane(path, nx=40, nh=4, L=100.0, H=10.0):
    """CPS4 平面应力悬臂梁: (nx+1)*(nh+1) 节点, 2 DOF/节点."""
    out = ['*HEADING', 'cantilever beam, CPS4 plane stress, 2 dof per node',
           '*NODE']
    nid = lambda i, j: j * (nx + 1) + i + 1
    for j in range(nh + 1):
        for i in range(nx + 1):
            out.append(f'{nid(i,j)}, {i*L/nx:.6f}, {j*H/nh:.6f}')
    out.append('*ELEMENT, TYPE=CPS4, ELSET=EALL')
    for j in range(nh):
        for i in range(nx):
            out.append(f'{j*nx+i+1}, {nid(i,j)}, {nid(i+1,j)}, '
                       f'{nid(i+1,j+1)}, {nid(i,j+1)}')
    out += ['*NSET, NSET=fixed']
    out.append(', '.join(str(nid(0, j)) for j in range(nh + 1)))
    out += [
        '*SOLID SECTION, ELSET=EALL, MATERIAL=steel', '1.0',
        '*MATERIAL, NAME=steel', '*ELASTIC', '210e9, 0.3', '*DENSITY',
        '7800.0', '*BOUNDARY', 'fixed, 1, 2', '*STEP, NAME=freq',
        '*FREQUENCY, EIGENSOLVER=AMS', '6, 0.0, 200.0', '*END STEP',
        '*END INPUT',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    n = (nx + 1) * (nh + 1)
    print(f'{path}: {n} nodes, {2*n} dof')


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(__file__), '..', 'bench_abaqus')
    emit_shell(os.path.join(base, 'cant_shell.inp'))
    emit_plane(os.path.join(base, 'cant_plane2d.inp'))
