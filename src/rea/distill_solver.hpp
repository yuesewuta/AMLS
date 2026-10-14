#pragma once
// REA 版缩聚特征求解（Abaqus pme_SmpReducedEigenSolver 语义的稠密适配）。
//
// 逆向依据（docs/abaqus_ams/逆向证据/pme_core_pseudocode.txt）：
//   Solve@pme_SmpReducedEigenSolver [99398]:
//     SmpDistillation → OneSubspaceIterationWithRigidBodyModes
//     → SmpRitzProjectionOnV1(K) → SmpRitzProjectionOnV1(M) → 稠密 Ritz 求解
//   SolveSubtreeEigenProblem [99859]: 子树算子以 σ=阈值/1000 缩放，
//     标准特征求解取 θ∈[0.001,1e36] ⟺ λ ≤ 蒸馏阈值（θ=σ/λ 倒数关系）。
//   TruncateEigenSolution [100516]: 按频段下界(0x280)剔除 + nev(0x2c0) 截断。
//
// 本模块在"缩聚已完成的整体缩减笔 (K_r, M_r)"上复现该流程：
//   1. 蒸馏过滤：剔除 Rayleigh 商超过蒸馏阈值 λ2 = cutoff2²·dVar9 的基向量；
//   2. 一次子空间迭代：Y = (K_r−σM_r)⁻¹ M_r，M_r 正交归一（shift σ 模拟
//      ABA_PAMS_GLOBALSHIFT，奇异时自动 escal­ation）；
//   3. Ritz 分析：投影特征求解，频段 (bandLo, bandHi] 截断 + nev 截断；
//   4. 块 Gram-Schmidt M_r 正交归一，嵌回原始 p 维缩减坐标（被剔除列补零），
//      使 PKU step4 恢复路径无需改动。
#include <cstdint>

#include "core/types.hpp"

namespace amls {
namespace rea {

struct DistillParams {
    double fmaxHz = 0.0;       // 关注频率上限 (Hz)  — std_pamsDriver param_5
    double fminHz = 0.0;       // 关注频率下限 (Hz)  — param_6
    double cutoff2 = 1.0;      // 蒸馏阈值倍数        — param_8（λ2 = cutoff2²·dVar9）
    double cutoff3 = 1.0;      // 初始子空间倍数      — param_9（保留 ≤ cutoff3²·dVar9）
    int64_t nev = 0;           // 请求模态数          — param_10（0 = 不按数截断）
    double shift = 0.0;        // 全局 shift          — ABA_PAMS_GLOBALSHIFT
    bool subspaceIteration = true;   // 一次子空间迭代（Abaqus 默认开）
    int64_t extractOrder = 0;        // 额外数截断（对齐 PKU extractOrder，0 = 不限）
};

struct DistillResult {
    DefaultTraits::Vector eigenvalues;        // λ = ω² 升序
    DefaultTraits::DenseMatrix eigenvectors;  // p×k，M_r 正交归一（嵌回全 p 维）
    int64_t inputSize = 0;                    // p（缩聚后基向量总数）
    int64_t distilledSize = 0;                // p'（蒸馏+初始过滤后）
};

// 对缩减笔 (Kr, Mr) 执行 蒸馏过滤 → (可选)一次子空间迭代 → Ritz → 截断 → 归一。
// Kr/Mr 为 p×p 稠密对称阵（K_r 半正定，M_r 正定）。输入不被修改。
// 若过滤后无基向量或频段内无模态，返回空 result（eigenvalues.size()==0）。
DistillResult distillRitz(const DefaultTraits::DenseMatrix& Kr,
                          const DefaultTraits::DenseMatrix& Mr,
                          const DistillParams& p);

}  // namespace rea
}  // namespace amls
