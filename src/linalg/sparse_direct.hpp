#pragma once
// 稀疏对称直接解接口（设计文档 §4.5）——块消元的唯一合法算子形态（P5/P11 硬约束:
// 无显式求逆 API）。
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Sparse>

#include "core/types.hpp"

namespace amls {

template <typename T>
class ISparseDirectSolver {
public:
    using SparseMatrix = typename T::SparseMatrix;
    using DenseMatrix  = typename T::DenseMatrix;

    /// Cholesky/LDLᵀ 分解（K 对称正定;检测到非正主元即抛异常,P10）。
    virtual void factorize(const SparseMatrix& A) = 0;

    /// 多右端求解 A X = B(B/X 稠密,列数任意——典型为边界列数 bᵢ 或 kᵢ,P2/P5)。
    virtual void solveWithFactor(const DenseMatrix& B, DenseMatrix& X) const = 0;

    virtual ~ISparseDirectSolver() = default;
};

/// Eigen SimplicialLDLT 实现（v1 默认）。仅支持 SPD;
/// 遇到零/负主元抛异常（范围外输入保护）。
template <typename T>
class EigenSimplicialLDLTSolver final : public ISparseDirectSolver<T> {
public:
    using SparseMatrix = typename T::SparseMatrix;
    using DenseMatrix  = typename T::DenseMatrix;

    void factorize(const SparseMatrix& A) override;
    void solveWithFactor(const DenseMatrix& B, DenseMatrix& X) const override;

    /// 非零主元个数（诊断）。
    int rank() const;

private:
    Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                          Eigen::AMDOrdering<typename T::Index>>
        llt_;
};

/// 稠密 SPD 分解（分割节点对角块用;接口同构,便于替换 CHOLMOD）。
template <typename T>
class DenseLDLTSolver final : public ISparseDirectSolver<T> {
public:
    using SparseMatrix = typename T::SparseMatrix;
    using DenseMatrix  = typename T::DenseMatrix;

    void factorize(const SparseMatrix& A) override;
    void solveWithFactor(const DenseMatrix& B, DenseMatrix& X) const override;

    /// 直接分解稠密 SPD 矩阵（分割节点对角块用,避免稀疏回绕）。
    void factorizeDense(const DenseMatrix& A);

private:
    Eigen::LDLT<DenseMatrix> llt_;
    typename T::Index n_ = 0;
};

} // namespace amls
