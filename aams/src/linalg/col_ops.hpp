#pragma once
// 列压缩矩阵算子：opsColSolve / opsColAccGemv / opsColAccGemm / opsColDenseProd
// —— 移植自 third_party/amlsPKU utils/col_ops，GEMM 底层走宿主 cblas
// （大乘法经 dgemmColsRaw 借线程），回代走 netlib LAPACK dsytrf/dsytrs。
#include <cstdint>
#include <mutex>
#include <vector>

#include "core/types.hpp"
#include "linalg/colcomp_mat.hpp"

namespace aams {

/// 稠密对称不定分解（Bunch–Kaufman）：A = P·L·D·Lᵀ·Pᵀ（原地覆盖 A 下三角）。
/// 返回 info（0 成功）；ipiv 为换主元数组（size = n）。
int32_t sytrfFactor(DefaultTraits::DenseMatrix& a, std::vector<int32_t>& ipiv);

/// 用 dsytrf 因子对多右端 B（n × nrhs，原地覆盖）回代求解。返回 info。
int32_t sytrsSolve(int32_t n, const DefaultTraits::DenseMatrix& lu,
                   const std::vector<int32_t>& ipiv, double* b, int32_t nrhs);

/// X = (LDLᵀ)⁻¹ · D，列压缩约束量求解。
///
/// X 仅保留 D 的实际有效列，单次多右端 dsytrs 回代（存储列连续）；不复制
/// 负状态预留列；支持 X 与 D 别名。
///
/// @param n    分解阶数（D 为 n 行）
/// @param lu   分解因子（dsytrf 输出，n × n 列主序）
/// @param ipiv 换主元数组（dsytrf 输出，size = n）
/// @param D    列压缩右端
/// @param X    列压缩结果（不带符号系数）
bool opsColSolve(int64_t n, const DefaultTraits::DenseMatrix& lu,
                 const std::vector<int32_t>& ipiv, const ColCompMat& D,
                 ColCompMat& X);

/// A += op(B) · C，纯列压缩乘法（gemv 流式，无回代）。
///
/// 对 C 的每个实际有效列 gemv 得一列乘积后立即映射散加到 A，不显式乘积
/// 矩阵；C 的全 0 占位列整列跳过；A 中被写入的占位列随即激活。
/// transB=true 计算 Bᵀ·C（乘积行号取 B 有效列原始列号），false 计算 B·C。
/// A 存储列须覆盖 C 的实际有效列（预留列定理运行时断言）。
bool opsColAccGemv(const ColCompMat& B, bool transB, const ColCompMat& C,
                   ColCompMat& A);

/// A += op(B) · C，纯列压缩乘法（gemm）。
///
/// 与 opsColAccGemv 同语义，但把 C 的实际有效列收集成连续右端后单次 gemm
/// 显式乘积，再按列映射散加到 A。
bool opsColAccGemm(const ColCompMat& B, bool transB, const ColCompMat& C,
                   ColCompMat& A);

/// 批量非转置累加：targets[i] += L · psis[i]（i = 0..psis.size()-1）。
///
/// CompressUpdate 专用：固定左块 L 与节点，向全部祖先目标块更新。
/// 把各 Ψ 的有效列拼接成一次 dgemm（乘积 Y 按列分组），再逐目标持锁
/// 散加——调用开销与锁持有时间从 O(|anc|) 次乘法降为 1 次乘法 + 散加。
/// 各 Ψ 行数须等于 L.totalCols（与逐对 opsColAcc 契约一致）；
/// L / Ψ 的 0 有效列按恒零贡献跳过，对应目标仍需调用方倒计数。
/// mtxs[i] 为散加 targets[i] 时持有的锁（可为 nullptr 表示无锁）。
bool opsColAccBatch(const ColCompMat& L,
                    const std::vector<const ColCompMat*>& psis,
                    const std::vector<ColCompMat*>& targets,
                    const std::vector<std::mutex*>& mtxs);

/// 批量转置累加：targets[i] += Bᵀ · Cs[i]（i = 0..Cs.size()-1）。
///
/// GaussKAccum / GaussMAccum 专用：固定 B（Ψ_j 或 M_temp_j），向全部
/// k 目标块更新。各 C 有效列拼成一次 dgemm（prod = Bᵀ·C_all），再逐
/// 目标持锁按 B 有效行散射。B 占位列的乘积行被丢弃（与逐对 gemv 的
/// 语义一致——占位贡献恒 0）。mtx 为 nullptr 表示无锁。
bool opsColAccBatchTrans(const ColCompMat& B,
                         const std::vector<const ColCompMat*>& Cs,
                         const std::vector<ColCompMat*>& targets,
                         const std::vector<std::mutex*>& mtxs);

/// 输出 opsColAcc gemv/gemm 剖析计数（AAMS_PROFILE=1 时生效）。
void colOpsDumpStats();

/// 按配置分发到 opsColAccGemv / opsColAccGemm（useGemv 对应
/// AmlsInternalParam::useColAccgemv）。
inline bool opsColAcc(bool useGemv, const ColCompMat& B, bool transB,
                      const ColCompMat& C, ColCompMat& A) {
    return useGemv ? opsColAccGemv(B, transB, C, A)
                   : opsColAccGemm(B, transB, C, A);
}

/// A += op(B) · C，稠密左端 × 列压缩右端（gemv 流式）。
///
/// B 为稠密矩阵（如特征向量 Φ），零拷贝读 B.data()；用于
/// μ = 0 + Φᵀ·M_temp 这类创建型累加（A 预先零初始化）。
bool opsColAccGemv(const DefaultTraits::DenseMatrix& B, bool transB,
                   const ColCompMat& C, ColCompMat& A);

/// A += op(B) · C，稠密左端 × 列压缩右端（gemm）。
bool opsColAccGemm(const DefaultTraits::DenseMatrix& B, bool transB,
                   const ColCompMat& C, ColCompMat& A);

inline bool opsColAcc(bool useGemv, const DefaultTraits::DenseMatrix& B,
                      bool transB, const ColCompMat& C, ColCompMat& A) {
    return useGemv ? opsColAccGemv(B, transB, C, A)
                   : opsColAccGemm(B, transB, C, A);
}

/// out = A · B，列压缩 × 稠密 = 稠密（消费侧乘积，覆盖写）。
///
/// A 满列且全部激活时直接用原缓冲单次 gemm（零拷贝）；否则 gather A 有效
/// 列成连续左端、按原始列号行 gather B，单次 gemm；A 的预留零列不参与
/// 计算（贡献恒 0）。用于子空间缩聚消费 μ·Φ_j。
bool opsColDenseProd(const ColCompMat& A, const DefaultTraits::DenseMatrix& B,
                     DefaultTraits::DenseMatrix& out);

/// useColAccgemv 开关（对应 PKU AmlsInternalParam::useColAccgemv，默认 gemv）。
bool& useColAccGemv();

} // namespace aams
