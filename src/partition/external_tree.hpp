#pragma once
// 外部树剖分器 —— 从 Abaqus/外部提供的剖分树文件构造 SeparatorTree。
//
// 用途：把 Abaqus AMS 实测的超节点/消去树（或其近似）喂给 PKU 求解管线，
// 复用全部下游（高斯消去/子空间缩聚/恢复），只替换剖分来源。
//
// 树文件格式（文本，每行一个节点，id 按 0..N-1）：
//   角色 子节点数 子节点id...  |  DOF列表（该节点在原始坐标系下的方程号）
// 角色: L=Leaf(叶, 无子), S=Separator(内点, ≥2子), R=Root(根, 无父)
// 例：
//   L 0 | 0 1 2
//   L 0 | 3 4 5
//   R 2 0 1 | 6 7 8
// 约定：后序编号（祖先 id 恒大于后代 id），叶子先行。
//       DOF 集合全体互不相交、并集 = [0, nTotal)，与 PKU 约定一致。
//
// 也支持"仅排列"模式（--tree-file 为一行 perm: old0 old1 ... oldN-1，
// perm[new]=old），此时树由内部 METIS_NodeND 生成，仅排列被外部覆盖。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "amls/separator_tree.hpp"
#include "partition/partitioner.hpp"

namespace amls {

template <typename T>
class ExternalTreePartitioner final : public IPartitioner<T> {
public:
    using Result = typename IPartitioner<T>::Result;

    /// 从树文件构造剖分结果。
    /// treeFile: 树文件路径。若为空则退化为内部 nestedDissection。
    Result fromFile(const typename T::SparseMatrix& A, const std::string& treeFile,
                    int64_t leafSize);
    Result recursiveBisection(const typename T::SparseMatrix& A, int nLevels,
                              int64_t leafSize) override {
        return nestedDissection(A, leafSize);
    }
    Result nestedDissection(const typename T::SparseMatrix& A,
                            int64_t leafSize) override {
        return fromFile(A, "", leafSize);
    }
    Result supernodePartition(const typename T::SparseMatrix& A,
                              int64_t maxBlockDofs) override {
        return fromFile(A, "", maxBlockDofs);
    }
};

} // namespace amls
