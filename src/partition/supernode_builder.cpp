#include "partition/supernode_builder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace amls {
namespace sym {

namespace {

// exact/fundamental 风格超节点分组: 返回列区间 [l, r) 的列表（每块一个）。
// 判据: 列 j 与 j+1 同块 iff
//   parent(j)=j+1 && colCount(j)==colCount(j+1)   (列模式严格相同)
//   附加 childCount(j+1)<=1（fundamental 风格, 不跨标量树分支）
// 注: 同节点 3 DOF 的行模式完全相同 → colCount 相等 → 合并为 3 DOF 块
//     （与 Abaqus stf 实测的 3/6/9/12 块分布一致）。
std::vector<std::pair<int64_t, int64_t>> findExactRanges(
    const SymbolicAnalysis& sa, bool preserveBranchBoundaries) {
    const int64_t n = static_cast<int64_t>(sa.columnCount.size());
    std::vector<int64_t> childCount(static_cast<size_t>(n), 0);
    for (int64_t j = 0; j < n; ++j)
        if (sa.parentColumn[static_cast<size_t>(j)] >= 0)
            ++childCount[static_cast<size_t>(
                sa.parentColumn[static_cast<size_t>(j)])];

    std::vector<std::pair<int64_t, int64_t>> ranges;
    int64_t l = 0;
    while (l < n) {
        int64_t r = l + 1;
        while (r < n) {
            const int64_t j = r - 1;
            const bool same =
                sa.parentColumn[static_cast<size_t>(j)] == r &&
                sa.columnCount[static_cast<size_t>(j)] ==
                    sa.columnCount[static_cast<size_t>(r)] &&
                (!preserveBranchBoundaries ||
                 childCount[static_cast<size_t>(r)] <= 1);
            if (!same) break;
            ++r;
        }
        ranges.emplace_back(l, r);
        l = r;
    }
    return ranges;
}

// 把超大 exact 块按连续列拆成 <= maxBlockDofs 的段。
std::vector<std::pair<int64_t, int64_t>> splitContiguous(
    const std::vector<std::pair<int64_t, int64_t>>& ranges,
    int64_t maxBlockDofs) {
    std::vector<std::pair<int64_t, int64_t>> out;
    for (const auto& [l, r] : ranges) {
        const int64_t len = r - l;
        if (len <= maxBlockDofs) {
            out.emplace_back(l, r);
        } else {
            int64_t cur = l;
            while (cur < r) {
                const int64_t next = std::min(r, cur + maxBlockDofs);
                out.emplace_back(cur, next);
                cur = next;
            }
        }
    }
    return out;
}

}  // namespace

BlockForest buildBlockForest(const SymbolicAnalysis& sa, int64_t maxBlockDofs) {
    if (maxBlockDofs < 1) maxBlockDofs = 1;
    const int64_t n = static_cast<int64_t>(sa.columnCount.size());
    auto ranges = findExactRanges(sa, /*preserveBranchBoundaries=*/true);
    ranges = splitContiguous(ranges, maxBlockDofs);

    const int64_t nb = static_cast<int64_t>(ranges.size());
    BlockForest forest;
    forest.blocks.resize(static_cast<size_t>(nb));
    forest.blockOfColumn.assign(static_cast<size_t>(n), -1);
    // 1. 基本块信息: begin/end
    for (int64_t b = 0; b < nb; ++b) {
        auto& blk = forest.blocks[static_cast<size_t>(b)];
        blk.begin = ranges[static_cast<size_t>(b)].first;
        blk.end = ranges[static_cast<size_t>(b)].second;
        for (int64_t j = blk.begin; j < blk.end; ++j)
            forest.blockOfColumn[static_cast<size_t>(j)] = b;
    }
    // 2. 块父关系: 块 B 的父 = 块内所有列的外部父列所在块中, 块号最大者
    //    （后序编号: 祖先 id 恒大于后代 id; 最大块号 = 最高层祖先）。
    //    消去树性质保证这些外部父都落在同一祖先链上; 取最高者作为直接父,
    //    其余成为 B 的跨级祖先（经直接父链可达, 祖先可比自然满足）。
    for (int64_t b = 0; b < nb; ++b) {
        auto& blk = forest.blocks[static_cast<size_t>(b)];
        int64_t parentBlock = -1;
        // 从 begin 列沿 etree 父链向上（跳过块内父列），第一个外部父列所在块
        // 即块 B 的 etree 父块。exact 分组保证块内列构成 etree 链，因此
        // 从 begin 出发的父链出口唯一确定块父。
        {
            int64_t j = blk.begin;
            const int64_t jend = blk.end;
            int64_t guard = 0;
            while (j >= blk.begin && j < jend && guard++ < (jend - blk.begin) + 1) {
                const int64_t p = sa.parentColumn[static_cast<size_t>(j)];
                if (p < 0) break;          // 根列
                const int64_t bp = forest.blockOfColumn[static_cast<size_t>(p)];
                if (bp != b) { parentBlock = bp; break; }  // 外部父
                j = p;                       // 沿链在块内前进
                if (j < blk.begin || j >= jend) break;  // 防越界
            }
        }
        blk.parent = parentBlock;
    }
    // 3. children 聚合
    for (int64_t b = 0; b < nb; ++b) {
        const int64_t par = forest.blocks[static_cast<size_t>(b)].parent;
        if (par >= 0)
            forest.blocks[static_cast<size_t>(par)].children.push_back(b);
    }
    // 4. roots
    for (int64_t b = 0; b < nb; ++b)
        if (forest.blocks[static_cast<size_t>(b)].parent == -1)
            forest.roots.push_back(b);

    // 5. 后序（迭代 DFS, 避免深链递归爆栈）; 祖先 id 恒大于后代
    std::vector<char> visited(static_cast<size_t>(nb), 0);
    forest.postorder.clear();
    forest.postorder.reserve(static_cast<size_t>(nb));
    for (const int64_t root : forest.roots) {
        std::vector<std::pair<int64_t, size_t>> st{{root, 0}};
        while (!st.empty()) {
            auto& top = st.back();
            const int64_t id = top.first;
            const auto& kids = forest.blocks[static_cast<size_t>(id)].children;
            if (top.second < kids.size()) {
                const int64_t c = kids[top.second];
                ++top.second;
                if (visited[static_cast<size_t>(c)])
                    throw std::runtime_error("supernode_builder: 环/共享子块");
                visited[static_cast<size_t>(c)] = 1;
                st.push_back({c, 0});
            } else {
                forest.postorder.push_back(id);
                st.pop_back();
            }
        }
    }
    // 后序重编号: 使祖先 id > 后代 id（遍历顺序=后序时天然满足, 直接使用）
    // 注: 上面 DFS 后序压入的顺序已保证"子先于父", 直接作为最终块 id 序。
    // 重新映射 blockOfColumn 与 children/parent。
    std::vector<int64_t> remap(static_cast<size_t>(nb), -1);
    for (int64_t i = 0; i < nb; ++i)
        remap[static_cast<size_t>(forest.postorder[static_cast<size_t>(i)])] = i;
    BlockForest out;
    out.blocks.resize(static_cast<size_t>(nb));
    out.blockOfColumn.assign(static_cast<size_t>(n), -1);
    for (int64_t old = 0; old < nb; ++old) {
        const int64_t nid = remap[static_cast<size_t>(old)];
        auto& nbk = out.blocks[static_cast<size_t>(nid)];
        const auto& obk = forest.blocks[static_cast<size_t>(old)];
        nbk.begin = obk.begin; nbk.end = obk.end;
        nbk.parent = obk.parent >= 0 ? remap[static_cast<size_t>(obk.parent)] : -1;
        for (const int64_t c : obk.children)
            nbk.children.push_back(remap[static_cast<size_t>(c)]);
        for (int64_t j = obk.begin; j < obk.end; ++j)
            out.blockOfColumn[static_cast<size_t>(j)] = nid;
        out.maxOwnedDofs =
            std::max(out.maxOwnedDofs, obk.end - obk.begin);
    }
    for (const int64_t r : forest.roots)
        out.roots.push_back(remap[static_cast<size_t>(r)]);
    // 新块号就是按后序分配的（remap[postorder[i]]=i），新 id 下的后序即
    // 恒等序列；存旧 id 在语义上是打乱的顺序（虽是全排列不越界）。
    out.postorder.resize(static_cast<size_t>(nb));
    std::iota(out.postorder.begin(), out.postorder.end(), 0);
    // 校验: 全覆盖、无重复、父唯一、maxOwnedDofs<=maxBlockDofs
    std::vector<int64_t> seen(static_cast<size_t>(n), -1);
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = out.blocks[static_cast<size_t>(b)];
        if (blk.end - blk.begin > maxBlockDofs)
            throw std::runtime_error("supernode_builder: 块超限");
        for (int64_t j = blk.begin; j < blk.end; ++j) {
            if (seen[static_cast<size_t>(j)] != -1)
                throw std::runtime_error("supernode_builder: DOF 重复归属");
            seen[static_cast<size_t>(j)] = b;
        }
    }
    for (int64_t j = 0; j < n; ++j)
        if (seen[static_cast<size_t>(j)] == -1)
            throw std::runtime_error("supernode_builder: DOF 未覆盖");
    // 原始消去边祖先可比自检（设计文档 §4.1 硬约束）:
    // 对 sa 中每个 (j, p=parentColumn[j]), 块 bj 与块 bp 必须可比
    // （bj==bp 或 bj 是 bp 后代/祖先）。
    {
        // 祖先判定: 沿 parent 链
        auto isAncestor = [&](int64_t a, int64_t b) -> bool {
            for (int64_t x = b; x != -1; x = out.blocks[static_cast<size_t>(x)].parent)
                if (x == a) return true;
            return false;
        };
        for (int64_t j = 0; j < n; ++j) {
            const int64_t p = sa.parentColumn[static_cast<size_t>(j)];
            if (p < 0) continue;
            const int64_t bj = out.blockOfColumn[static_cast<size_t>(j)];
            const int64_t bp = out.blockOfColumn[static_cast<size_t>(p)];
            if (bj == bp) continue;
            if (!isAncestor(bj, bp) && !isAncestor(bp, bj)) {
                throw std::runtime_error(
                    "supernode_builder: 原始消去边 (j=" + std::to_string(j) +
                    ", p=" + std::to_string(p) + ") 块 " +
                    std::to_string(bj) + "/" + std::to_string(bp) +
                    " 祖先不可比");
            }
        }
    }
    return out;
}

void dumpBlockForestStats(const BlockForest& forest) {
    const int64_t nb = static_cast<int64_t>(forest.blocks.size());
    int64_t nLeaf = 0, nSingle = 0, maxDepth = 0, maxFan = 0;
    std::vector<int64_t> depth(static_cast<size_t>(nb), 0);
    // 深度须自顶向下传播：逆后序（父先子后）遍历
    for (auto rit = forest.postorder.rbegin(); rit != forest.postorder.rend();
         ++rit) {
        const int64_t id = *rit;
        const auto& blk = forest.blocks[static_cast<size_t>(id)];
        if (blk.children.empty()) ++nLeaf;
        if (blk.children.size() == 1) ++nSingle;
        maxFan = std::max(maxFan, static_cast<int64_t>(blk.children.size()));
        for (const int64_t c : blk.children)
            depth[static_cast<size_t>(c)] = depth[static_cast<size_t>(id)] + 1;
    }
    for (const int64_t d : depth) maxDepth = std::max(maxDepth, d);
    std::fprintf(stderr,
                 "[supernode] blocks=%lld roots=%lld leaf=%lld single=%lld "
                 "maxDepth=%lld maxFan=%lld maxOwned=%lld\n",
                 static_cast<long long>(nb),
                 static_cast<long long>(forest.roots.size()),
                 static_cast<long long>(nLeaf),
                 static_cast<long long>(nSingle),
                 static_cast<long long>(maxDepth),
                 static_cast<long long>(maxFan),
                 static_cast<long long>(forest.maxOwnedDofs));
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        // 规模分布
        std::vector<int64_t> sizes;
        for (const auto& blk : forest.blocks)
            sizes.push_back(blk.end - blk.begin);
        std::sort(sizes.rbegin(), sizes.rend());
        std::fprintf(stderr, "[supernode] top-sizes:");
        for (int64_t i = 0;
             i < static_cast<int64_t>(std::min<int64_t>(12, sizes.size()));
             ++i)
            std::fprintf(stderr, " %lld",
                         static_cast<long long>(sizes[static_cast<size_t>(i)]));
        std::fprintf(stderr, "\n");
        // 打印含 dof 36..45 的块及其父
        for (int64_t b = 0; b < nb; ++b) {
            const auto& blk = forest.blocks[static_cast<size_t>(b)];
            if (blk.begin <= 45 && 36 <= blk.end)
                std::fprintf(stderr,
                             "[supernode] blk%lld [%lld,%lld) parent=%lld kids=%lld\n",
                             (long long)b, (long long)blk.begin,
                             (long long)blk.end, (long long)blk.parent,
                             (long long)blk.children.size());
        }
    }
}

}  // namespace sym
}  // namespace amls
