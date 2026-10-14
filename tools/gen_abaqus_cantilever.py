#!/usr/bin/env python3
"""生成 Abaqus 悬臂块模型 input 文件（规则 C3D8 网格）。

用于 AMLS vs Abaqus AMS 公平对比：
  * 同一模型：*FREQUENCY, EIGENSOLVER=AMS 求特征值（参考解）
  * *MATRIX GENERATE + *MATRIX OUTPUT 导出 K/M（喂给 amls_solve）
坐标单位 m，材料钢 E=210e9 Pa, nu=0.3, rho=7800 kg/m^3（SI）。

用法:
  gen_abaqus_cantilever.py --nx NX --ny NY --nz NZ --nev NEV -o out.inp
约束面: x=0 平面所有节点固支（U1=U2=U3=0），消去刚体模态，K 正定。
"""
import argparse
import sys

BLOCK_HEAD = "*HEADING\ncantilever block, C3D8, unit elements\n"


def gen_model(nx, ny, nz, nev, fmax=10.0):
    nxn, nyn = nx + 1, ny + 1
    nnodes = nxn * nyn * (nz + 1)
    nelem = nx * ny * nz

    node_of = lambda i, j, k: 1 + i + nxn * (j + nyn * k)

    parts = []
    parts.append(BLOCK_HEAD)
    parts.append("*NODE, NSET=ALLNODES\n")
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                parts.append(f"{node_of(i,j,k)}, {i}, {j}, {k}\n")
    parts.append("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = node_of(i, j, k)
                n2 = node_of(i + 1, j, k)
                n3 = node_of(i + 1, j + 1, k)
                n4 = node_of(i, j + 1, k)
                n5 = node_of(i, j, k + 1)
                n6 = node_of(i + 1, j, k + 1)
                n7 = node_of(i + 1, j + 1, k + 1)
                n8 = node_of(i, j + 1, k + 1)
                eid = 1 + i + nx * (j + ny * k)
                parts.append(
                    f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")
    parts.append("*SOLID SECTION, ELSET=EALL, MATERIAL=Steel\n")
    parts.append("*MATERIAL, NAME=Steel\n")
    parts.append("*ELASTIC\n210e9, 0.3\n")
    parts.append("*DENSITY\n7800.0\n")

    # 固支面：x=0 平面（i==0）
    fixed = [node_of(0, j, k) for k in range(nz + 1) for j in range(ny + 1)]
    parts.append("*NSET, NSET=fixed\n")
    for idx in range(0, len(fixed), 8):
        parts.append(", ".join(str(n) for n in fixed[idx:idx + 8]) + "\n")
    parts.append("*BOUNDARY\nfixed, 1, 3\n")

    parts.append("*STEP, NAME=freq\n")
    parts.append(f"*FREQUENCY, EIGENSOLVER=AMS\n")
    parts.append(f"{nev}, 0.0, {fmax}\n")
    parts.append("*END STEP\n")

    # MATRIX OUTPUT 的文件后缀编号对应分析步编号。
    parts.append("*STEP, NAME=matgen, PERTURBATION\n")
    parts.append("*MATRIX GENERATE, STIFFNESS, MASS\n")
    parts.append("*MATRIX OUTPUT, STIFFNESS, MASS, FORMAT=COORDINATE\n")
    parts.append("*END STEP\n")

    text = "".join(parts)
    ndof = nnodes * 3
    return text, nnodes, nelem, ndof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--nev", type=int, default=20)
    ap.add_argument("--fmax", type=float, default=10.0,
                    help="AMS 最高关注频率，单位 Hz (默认 10)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if min(a.nx, a.ny, a.nz, a.nev) <= 0 or not (0 < a.fmax < float("inf")):
        ap.error("网格数、模态数和最高频率必须为正且有限")

    text, nnodes, nelem, ndof = gen_model(a.nx, a.ny, a.nz, a.nev, a.fmax)
    with open(a.out, "w") as f:
        f.write(text)
    sys.stderr.write(
        f"[gen] out={a.out} nodes={nnodes} elems={nelem} dof={ndof} "
        f"fixed_face_nodes={(a.ny+1)*(a.nz+1)} bytes={len(text)}\n")


if __name__ == "__main__":
    main()