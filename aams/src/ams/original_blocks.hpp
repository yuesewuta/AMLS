#pragma once
// 原始子块抽取（PKU separate_tree 的 blockK_/blockM_ 语义）。
//
// 从重排后 pencil 一次性抽取分割树全部节点的对角子块与 (后代 i, 祖先 a)
// 耦合子块，供 GaussEliminationModule 消去与子空间缩聚只读使用。
// 原始矩阵在构造期扫描一遍即丢弃引用；K、M 按各自稀疏模式独立抽取
//（不做 PKU 的 AaddZeroB 模式统一）。
//
// 语义约定：
//   - 树节点 indices 为重排坐标下的全局行列号；块内局部行/列号 = 升序
//     索引列表中的位置（构造时排序固化）。
//   - subK(i,a) 为 n_i × n_a（行 = i 空间、列 = a 空间）；K 对称下
//     subK(i,a)ᵀ = K(a 空间 × i 空间)。
//   - 无结构耦合的 (i,a) 对返回形状正确的空矩阵。
//   - 不可比（互非祖先）节点对之间出现耦合 → 结构非法，构造抛
//     std::invalid_argument（分隔树性质：子树间只经共同祖先耦合）。
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ams/separator_tree.hpp"
#include "core/types.hpp"

namespace aams {

class OriginalBlocks {
public:
    using SparseMatrix = DefaultTraits::SparseMatrix;

    /// reordered：已按树节点索引重排的 (K, M)。树结构非法（索引越界/
    /// 重复/不完备、不可比节点耦合、K-M 形状不一致）抛 std::invalid_argument。
    OriginalBlocks(const Pencil<DefaultTraits>& reordered,
                   const SeparatorTree& tree);

    int64_t nNodes() const { return N_; }
    const SeparatorTree& tree() const { return tree_; }
    int64_t size(int64_t i) const;
    /// 节点 i 的升序全局行列号（重排坐标）。
    const std::vector<int64_t>& indices(int64_t i) const;

    /// 节点 i 的对角子块（n_i × n_i）。
    const SparseMatrix& diagK(int64_t i) const;
    const SparseMatrix& diagM(int64_t i) const;
    /// (i,a) 耦合子块：i 为后代、a 为其祖先（n_i × n_a）。
    /// a 不是 i 的祖先时抛 std::out_of_range。
    const SparseMatrix& subK(int64_t i, int64_t a) const;
    const SparseMatrix& subM(int64_t i, int64_t a) const;
    bool hasCouplingK(int64_t i, int64_t a) const;
    bool hasCouplingM(int64_t i, int64_t a) const;
    /// 全部 (i, 祖先) 对数（诊断）。
    int64_t nCouplingPairs() const { return nPairs_; }

private:
    int64_t key_(int64_t i, int64_t a) const { return i * N_ + a; }
    /// O(1) 祖先判定（DFS 进入/离开区间）。
    bool isAncestor_(int64_t a, int64_t d) const {
        return a == d || (tin_[static_cast<size_t>(a)] < tin_[static_cast<size_t>(d)] &&
                          tout_[static_cast<size_t>(d)] < tout_[static_cast<size_t>(a)]);
    }
    /// 单遍双扫抽取：pass1 计数 reserve，pass2 按列升序插入。
    void extract_(const SparseMatrix& A,
                  std::unordered_map<int64_t, SparseMatrix>& diag,
                  std::unordered_map<int64_t, SparseMatrix>& coup);

    const SeparatorTree& tree_;
    int64_t N_ = 0;
    std::vector<std::vector<int64_t>> idx_;  // 每节点升序索引副本
    std::vector<int64_t> nodeOf_, localOf_;  // 全局 dof → (节点, 块内局部号)
    std::vector<int64_t> tin_, tout_;        // DFS 区间（祖先判定）
    std::unordered_map<int64_t, SparseMatrix> kDiag_, mDiag_;
    std::unordered_map<int64_t, SparseMatrix> kCoup_, mCoup_;  // key i·N+a
    int64_t nPairs_ = 0;
};

} // namespace aams
