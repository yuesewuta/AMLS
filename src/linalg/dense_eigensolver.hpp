#pragma once
// 稠密广义特征求解接口（设计文档 §4.5）——全部为小规模稠密问题:
// 局部谱问题 + 投影问题。
#include <stdexcept>

#include "core/types.hpp"

namespace amls {

template <typename T>
class IDenseEigensolver {
public:
    using Scalar = typename T::Scalar;
    using DenseMatrix = typename T::DenseMatrix;
    using Vector = typename T::Vector;

    /// 全部特征对:K z = w_j M z_j,升序;K/M 被覆盖。
    virtual void solveAll(DenseMatrix& K, DenseMatrix& M, Vector& w,
                          DenseMatrix& V) = 0;

    /// 半开区间 (lo, hi] 内全部特征对（按值）,升序。
    /// 空区间返回 m=0。
    virtual void solveInterval(DenseMatrix& K, DenseMatrix& M, Scalar lo,
                               Scalar hi, Vector& w, DenseMatrix& V) = 0;

    /// 索引区间 [il, iu]（1-based,含端点）的特征对,升序。
    virtual void solveIndex(DenseMatrix& K, DenseMatrix& M, int64_t il,
                            int64_t iu, Vector& w, DenseMatrix& V) = 0;

    virtual ~IDenseEigensolver() = default;
};

/// LAPACK 实现:dsygvd（全谱）/ dsygvx（区间）。
template <typename T>
class LapackDenseEigensolver final : public IDenseEigensolver<T> {
public:
    using Scalar = typename T::Scalar;
    using DenseMatrix = typename T::DenseMatrix;
    using Vector = typename T::Vector;

    void solveAll(DenseMatrix& K, DenseMatrix& M, Vector& w,
                  DenseMatrix& V) override;

    void solveInterval(DenseMatrix& K, DenseMatrix& M, Scalar lo, Scalar hi,
                       Vector& w, DenseMatrix& V) override;

    void solveIndex(DenseMatrix& K, DenseMatrix& M, int64_t il, int64_t iu,
                    Vector& w, DenseMatrix& V) override;
};

} // namespace amls
