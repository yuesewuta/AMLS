#pragma once
// μ̃（子空间缩聚右耦合中间量）内存存储——缩聚右耦合结果的独立载体
//（SubspaceCondensation 自身不再保存这些块）。
//
// 键 = (后代 c, 祖先 a)；值 = ColComp（k_c 行 × a 空间压缩列）。三个动作
// 对应依赖关系拆解表：
//   N8 种子：μ̃(c,a)  = Z_cᵀ·Mtemp(c,a)        （c 的全部祖先 a）
//   N10 传播：μ̃(c,a) += μ̃(c,i)·u(i,a)         （i 在 c→a 路径上）
//   N9 折叠：m(c,i)  = μ̃(c,i)·Z_i             （最终降阶质量块，k_c×k_i）
//
// 并发设计：键在 prepare 全部预建（值仅留列模式，行数待种子确定），运行期
// 无结构变更、无 rehash，find 返回的引用稳定。每键的写者链 = 种子(c) → 沿
// c→a 路径的 fold(i) 序列；树路径全序 + DAG 依赖（fold(i) ← fold(孩子)、
// fold(i) ← N8(全部后代)）保证任意时刻单写者——无条带锁、无原子。
// 列模式（符号预算，无精确抵消假定）：
//   pattern(c,a) = rowM(a,c) ∪ ⋃_{i∈path c→a（含 c 不含 a）} rowK(a,i)
// 其中 rowK/rowM 为 GE 符号填充 pass 的 (祖先,后代) 块终态非零行位图
//（rowK(a,i) = u(i,a) 的列模式，rowM(a,c) = Mtemp(c,a) 中 M 耦合的列模式）。
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "amls/separator_tree.hpp"
#include "core/types.hpp"
#include "linalg/colcomp.hpp"

namespace amls {

class MassCouplingStore {
public:
    using DenseMatrix = DefaultTraits::DenseMatrix;
    /// (祖先 a, 后代 i) → 该耦合块终态非零行位图（GE 符号 pass 产物）。
    using RowMaskFn = std::function<const ColMask&(int64_t a, int64_t i)>;

    /// 预建全部 (c, a) 祖先对的槽位（模式为空的对不建——其 μ̃ 恒零，
    /// 折叠产出零块，装配端"缺块即零"语义下可省）。行数 = 0，待种子定。
    /// 同时建立列索引 columnIndex_[a] = {c : (c,a) 有槽位}（fold(a) 遍历）。
    void prepare(const SeparatorTree& tree, const RowMaskFn& rowK,
                 const RowMaskFn& rowM,
                 const std::function<int64_t(int64_t node)>& nodeSize) {
        clear();
        N_ = tree.nNodes();
        columnIndex_.assign(static_cast<size_t>(N_), {});
        int64_t keys = 16;
        for (int64_t c = 0; c < N_; ++c)
            keys += static_cast<int64_t>(tree.ancestors(c).size());
        blocks_.reserve(static_cast<size_t>(keys));
        for (int64_t c = 0; c < N_; ++c) {
            for (const int64_t a : tree.ancestors(c)) {
                ColMask mask = rowM(a, c);
                for (int64_t i = c; i != a; i = tree.node(i).parent)
                    mask.merge(rowK(a, i));
                if (mask.count() == 0)
                    continue;
                ColComp blk;
                blk.full = nodeSize(a);
                blk.ident = mask.count() == blk.full;
                if (!blk.ident)
                    blk.cols = mask.indices();
                blk.data = DenseMatrix::Zero(0, blk.storedCols());
                blocks_.emplace(key_(c, a), std::move(blk));
                columnIndex_[static_cast<size_t>(a)].push_back(c);
            }
        }
    }

    /// 趟间清空（两趟阈值估计的趟 B 前调用）。
    void clear() {
        blocks_.clear();
        columnIndex_.clear();
        N_ = 0;
    }

    /// 列节点 a 的后代槽位列表（fold(a) 遍历的 c 集合）。
    const std::vector<int64_t>& columnEntries(int64_t a) const {
        return columnIndex_[static_cast<size_t>(a)];
    }

    bool has(int64_t c, int64_t a) const {
        return blocks_.find(key_(c, a)) != blocks_.end();
    }

    /// N8 种子：μ̃(c,a) = eigenvectorsCᵀ·Mtemp(c,a)。k_c = eigenvectorsC.cols()
    /// 为 0 时跳过（槽位保持 0 行，后续传播/折叠均为空操作）。
    void seed(int64_t c, int64_t a, const DenseMatrix& eigenvectorsC,
              const ColComp& mtemp) {
        const int64_t kc = eigenvectorsC.cols();
        if (kc <= 0)
            return;
        const auto it = blocks_.find(key_(c, a));
        if (it == blocks_.end())
            return;
        ColComp& blk = it->second;
        blk.data = DenseMatrix::Zero(kc, blk.storedCols());
        blockops::accumulateTransposedProduct(blk, eigenvectorsC, mtemp);
    }

    /// N10 传播：μ̃(c,a) += μ̃(c,i)·u(i,a)。源/目标槽位不存在（模式空）
    /// 或行数为 0（k_c=0）时为空操作。
    void propagate(int64_t c, int64_t i, int64_t a, const ColComp& u) {
        const auto src = blocks_.find(key_(c, i));
        if (src == blocks_.end())
            return;
        const auto dst = blocks_.find(key_(c, a));
        if (dst == blocks_.end())
            return;
        blockops::accumulateProduct(dst->second, src->second, u);
    }

    /// N9 折叠：m(c,i) = μ̃(c,i)·eigenvectorsI（k_c×k_i）。
    /// 槽位缺失 / k_c=0 / k_i=0 时返回 0×0（调用方跳过装配）。
    DenseMatrix fold(int64_t c, int64_t i,
                     const DenseMatrix& eigenvectorsI) const {
        DenseMatrix out;
        const auto it = blocks_.find(key_(c, i));
        if (it == blocks_.end() || it->second.data.rows() == 0 ||
            eigenvectorsI.cols() == 0)
            return out;
        out = DenseMatrix::Zero(it->second.data.rows(),
                                eigenvectorsI.cols());
        blockops::accumulateProduct(out, it->second, eigenvectorsI);
        return out;
    }

    /// 折叠完成后释放列 i 的全部 μ̃(·,i)（值缩容，表项与列索引保留——
    /// 运行期无结构变更）。
    void releaseColumn(int64_t i) {
        for (const int64_t c : columnIndex_[static_cast<size_t>(i)])
            if (const auto it = blocks_.find(key_(c, i)); it != blocks_.end())
                it->second.shrink();
    }

    /// 诊断：当前 μ̃ 数据总字节数。
    double denseBytes() const {
        double bytes = 0.0;
        for (const auto& [_, blk] : blocks_) {
            bytes += blk.data.size() * sizeof(double);
            bytes += static_cast<double>(blk.cols.size()) * sizeof(int32_t);
        }
        return bytes;
    }

private:
    int64_t key_(int64_t c, int64_t a) const { return c * N_ + a; }

    int64_t N_ = 0;
    std::unordered_map<int64_t, ColComp> blocks_;
    std::vector<std::vector<int64_t>> columnIndex_;
};

} // namespace amls
