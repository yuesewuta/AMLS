#pragma once
// 图划分接口（设计文档 §4.3）:递归二分 → 分割节点树 + 重排列。
#include "amls/separator_tree.hpp"
#include "core/types.hpp"

namespace amls {

template <typename T>
class IPartitioner {
public:
    struct Result {
        SeparatorTree tree;
        typename T::Permutation perm;   // perm[new] = old
    };

    /// 对 A = |K|+|M| 的邻接结构做递归二分,最多 nLevels 层;
    /// 子图规模 ≤ leafSize 时停止（叶节点）。
    virtual Result recursiveBisection(const typename T::SparseMatrix& A,
                                      int nLevels, int64_t leafSize) = 0;
    virtual ~IPartitioner() = default;
};

/// METIS 实现：METIS_PartGraphRecursive 二分 + METIS_ComputeVertexSeparator
/// 取使两子图不连通的顶点分隔（分割节点的唯一定义）。
template <typename T>
class MetisPartitioner final : public IPartitioner<T> {
public:
    using Result = typename IPartitioner<T>::Result;
    Result recursiveBisection(const typename T::SparseMatrix& A, int nLevels,
                              int64_t leafSize) override;
};

} // namespace amls
