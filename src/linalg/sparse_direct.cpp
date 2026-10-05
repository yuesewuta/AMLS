#include "linalg/sparse_direct.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/SparseCholesky>

namespace amls {

template <typename T>
void EigenSimplicialLDLTSolver<T>::factorize(const SparseMatrix& A) {
    llt_.compute(A);
    if (llt_.info() != Eigen::Success)
        throw std::runtime_error(
            "EigenSimplicialLDLT: 分解失败（K 非正定或奇异;请检查输入或对 K 做移位）");
    // P10 断言：SPD 输入不允许出现非正主元
    const auto& D = llt_.vectorD();
    for (typename T::Index i = 0; i < D.size(); ++i) {
        if (!(D[i] > 0.0) || !std::isfinite(D[i]))
            throw std::runtime_error(
                "EigenSimplicialLDLT: 检测到非正主元（输入超出 SPD 范围）");
    }
}

template <typename T>
void EigenSimplicialLDLTSolver<T>::solveWithFactor(const DenseMatrix& B,
                                                   DenseMatrix& X) const {
    X = llt_.solve(B);
    if (llt_.info() != Eigen::Success)
        throw std::runtime_error("EigenSimplicialLDLT: 求解失败");
}

template <typename T>
int EigenSimplicialLDLTSolver<T>::rank() const {
    return static_cast<int>(llt_.vectorD().size());
}

template <typename T>
void DenseLDLTSolver<T>::factorize(const SparseMatrix& A) {
    factorizeDense(DenseMatrix(A));
}

template <typename T>
void DenseLDLTSolver<T>::factorizeDense(const DenseMatrix& A) {
    llt_.compute(A);
    if (llt_.info() != Eigen::Success)
        throw std::runtime_error("DenseLDLT: 分解失败");
    const auto& D = llt_.vectorD();
    for (typename T::Index i = 0; i < D.size(); ++i) {
        if (!(D[i] > 0.0) || !std::isfinite(D[i]))
            throw std::runtime_error(
                "DenseLDLT: 检测到非正主元（输入超出 SPD 范围,P10）");
    }
    n_ = A.rows();
}

template <typename T>
void DenseLDLTSolver<T>::solveWithFactor(const DenseMatrix& B,
                                         DenseMatrix& X) const {
    X = llt_.solve(B);
    if (llt_.info() != Eigen::Success)
        throw std::runtime_error("DenseLDLT: 求解失败");
}

template class EigenSimplicialLDLTSolver<DefaultTraits>;
template class DenseLDLTSolver<DefaultTraits>;

} // namespace amls
