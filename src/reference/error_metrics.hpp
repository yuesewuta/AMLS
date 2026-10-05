// 误差指标（验证方案 §1）：特征值相对误差（主,≤4%）、残差、向量夹角。
#pragma once
#include <cstdint>
#include <vector>

#include "core/types.hpp"

namespace amls {

struct ErrorMetrics {
    /// 特征值相对误差 |θ−λ_ref|/|λ_ref|（主验收指标）。
    static std::vector<double> eigenvalueRelErr(
        const DefaultTraits::Vector& theta,
        const DefaultTraits::Vector& lambdaRef);

    /// 最大特征值相对误差。
    static double maxEigenvalueRelErr(const DefaultTraits::Vector& theta,
                                      const DefaultTraits::Vector& lambdaRef);

    /// 残差 r = ‖Kz−θMz‖/(‖Kz‖+|θ|‖Mz‖)。
    static double residual(const DefaultTraits::SparseMatrix& K,
                           const DefaultTraits::SparseMatrix& M,
                           double theta,
                           const DefaultTraits::Vector& z);

    /// 频率相对误差（对外报告口径）:|√θ−√λ|/√λ。
    static std::vector<double> frequencyRelErr(
        const DefaultTraits::Vector& theta,
        const DefaultTraits::Vector& lambdaRef);
};

} // namespace amls
