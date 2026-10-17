#pragma once
// 图划分接口（设计文档 §4.3）:递归二分 → 分割节点树 + 重排列。
#include "ams/separator_tree.hpp"
#include "core/types.hpp"

namespace aams {

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
    /// 单次 METIS_NodeND + 消去树分组（amalgamation ≥ leafSize）：
    /// 一次调用产出整棵分割树（可多叉），替代 O(2^nLevels) 次递归二分。
    virtual Result nestedDissection(const typename T::SparseMatrix& A,
                                    int64_t leafSize) = 0;
    /// NodeND + 符号分析（LDL 列结构）+ exact 超节点分组 + ≤maxBlockDofs
    /// 受限拆分：产出细粒度消去块树（保留单子链）。
    virtual Result supernodePartition(const typename T::SparseMatrix& A,
                                      int64_t maxBlockDofs) = 0;
    virtual ~IPartitioner() = default;
};

/// METIS 实现：METIS_PartGraphRecursive 二分 + METIS_ComputeVertexSeparator
/// 取使两子图不连通的顶点分隔（分割节点的唯一定义）；
/// 或 METIS_NodeND 单次嵌套剖分 + 消去树/超节点合并建树。
template <typename T>
class MetisPartitioner final : public IPartitioner<T> {
public:
    using Result = typename IPartitioner<T>::Result;
    Result recursiveBisection(const typename T::SparseMatrix& A, int nLevels,
                              int64_t leafSize) override;
    Result nestedDissection(const typename T::SparseMatrix& A,
                            int64_t leafSize) override;
    Result supernodePartition(const typename T::SparseMatrix& A,
                              int64_t maxBlockDofs) override;
};

} // namespace aams
