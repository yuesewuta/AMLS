#pragma once
// 参考解（验证方案 §1）：按规模分级。
//   n ≤ denseMax: LAPACK DSYGVD 全谱（金标准）
//   n > denseMax: Spectra SymGEigsShiftSolver（移位反演,σ=0 下 Cholesky 模式）
#include <cstdint>

#include "core/types.hpp"

namespace amls {

class ReferenceSolver {
public:
    using SparseMatrix = DefaultTraits::SparseMatrix;
    using Vector = DefaultTraits::Vector;
    using DenseMatrix = DefaultTraits::DenseMatrix;

    explicit ReferenceSolver(int64_t denseMax = 5000)
        : denseMax_(denseMax) {}

    /// 求 (K, M) 低端 nev 个特征对（升序）。K 需 SPD（σ=0 移位反演）。
    /// 返回特征值;特征向量可选（verify 只需特征值作主指标）。
    Vector smallestEigenvalues(const SparseMatrix& K, const SparseMatrix& M,
                               int64_t nev, DenseMatrix* V = nullptr);

private:
    int64_t denseMax_;
};

} // namespace amls
