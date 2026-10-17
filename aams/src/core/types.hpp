#pragma once
// 基础类型（纯代数,无语义）——见设计文档 §4.1
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace aams {

template <typename S, typename I>
struct Traits {
    using Scalar = S;
    using Index  = I;
    using SparseMatrix = Eigen::SparseMatrix<S, Eigen::ColMajor, I>;
    using DenseMatrix  = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector       = Eigen::Matrix<S, Eigen::Dynamic, 1>;
    using Permutation  = std::vector<I>;   // perm[new] = old
};

using DefaultTraits = Traits<double, int64_t>;

/// 矩阵对 (K, M)：K 对称,M 对称正定（设计文档 §4.1 pencil.h）
template <typename T>
struct Pencil {
    typename T::SparseMatrix K;
    typename T::SparseMatrix M;
};

} // namespace aams
