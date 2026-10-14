#include "rea/aggregation.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace amls {
namespace sym {

namespace {

// 可变树节点（聚合工作表示）。DOF 用重排坐标连续区间 [dofBegin, dofEnd) 表示。
struct AggNode {
    int64_t dofBegin = -1;
    int64_t dofEnd = -1;
    int64_t parent = -1;
    std::vector<int64_t> children;
    bool collapsed = false;   // 本轮已被折叠（Abaqus Collapse2 marker）
    bool dead = false;        // 已被父吸收，逻辑删除
    int64_t size() const { return dofEnd - dofBegin; }
};

struct MutableForest {
    std::vector<AggNode> nodes;
    std::vector<int64_t> roots;
    // 原始 dof → 块 id 仅用于一致性校验；聚合过程用区间合并维护。
};

// 从 BlockForest 建可变森林。BlockForest 的 blocks[] 按后序编号、区间不重叠、
// children 已含全部子块 id。
MutableForest fromBlocks(const BlockForest& base) {
    const int64_t n = static_cast<int64_t>(base.blocks.size());
    MutableForest mf;
    mf.nodes.resize(static_cast<size_t>(n));
    for (int64_t b = 0; b < n; ++b) {
        const auto& blk = base.blocks[static_cast<size_t>(b)];
        auto& nd = mf.nodes[static_cast<size_t>(b)];
        nd.dofBegin = blk.begin;
        nd.dofEnd = blk.end;
        nd.parent = blk.parent;
        nd.children = blk.children;
    }
    mf.roots = base.roots;
    return mf;
}

// 把可变森林重建为 BlockForest（重新后序编号，祖先 id>后代，区间全覆盖互斥）。
BlockForest toBlocks(const MutableForest& mf, int64_t nTotalDofs) {
    // 收集存活节点，计算后序（子先父）。
    const int64_t N = static_cast<int64_t>(mf.nodes.size());
    std::vector<int64_t> post;   // 存活节点后序（节点 id）
    std::vector<char> state(static_cast<size_t>(N), 0);  // 0=未访问 1=灰(在栈) 2=黑(完成)
    std::vector<std::pair<int64_t, size_t>> st;
    for (const int64_t r : mf.roots) {
        if (mf.nodes[static_cast<size_t>(r)].dead) continue;
        if (state[static_cast<size_t>(r)] == 2) continue;
        st.push_back({r, 0});
        state[static_cast<size_t>(r)] = 1;
        while (!st.empty()) {
            auto& top = st.back();
            const int64_t id = top.first;
            const auto& nd = mf.nodes[static_cast<size_t>(id)];
            if (top.second < nd.children.size()) {
                const int64_t c = nd.children[top.second];
                ++top.second;
                if (mf.nodes[static_cast<size_t>(c)].dead) continue;
                if (state[static_cast<size_t>(c)] == 1)
                    throw std::runtime_error(
                        "aggregateAbaqusStyle: 聚合后出现环 (节点 " +
                        std::to_string(c) + " 已在栈中)");
                if (state[static_cast<size_t>(c)] == 2) continue;
                state[static_cast<size_t>(c)] = 1;
                st.push_back({c, 0});
            } else {
                if (!nd.dead) post.push_back(id);
                state[static_cast<size_t>(id)] = 2;
                st.pop_back();
            }
        }
    }
    // 后序: 存活节点, 子先父后。存活的节点就是聚合后的块。
    BlockForest out;
    const int64_t nb = static_cast<int64_t>(post.size());
    out.blocks.resize(static_cast<size_t>(nb));
    out.blockOfColumn.assign(static_cast<size_t>(nTotalDofs), -1);
    // post[] 已是子先父后的顺序；反向映射 old_id → 新块号（新块号也按后序，
    // 父块号恒大于子块号）。
    std::vector<int64_t> newId(static_cast<size_t>(N), -1);
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t old = post[static_cast<size_t>(i)];
        newId[static_cast<size_t>(old)] = i;
    }
    for (int64_t i = 0; i < nb; ++i) {
        const int64_t old = post[static_cast<size_t>(i)];
        const auto& ond = mf.nodes[static_cast<size_t>(old)];
        auto& nbk = out.blocks[static_cast<size_t>(i)];
        nbk.begin = ond.dofBegin;
        nbk.end = ond.dofEnd;
        nbk.parent = ond.parent >= 0 ? newId[static_cast<size_t>(ond.parent)] : -1;
        for (const int64_t c : ond.children)
            if (!mf.nodes[static_cast<size_t>(c)].dead)
                nbk.children.push_back(newId[static_cast<size_t>(c)]);
        for (int64_t j = ond.dofBegin; j < ond.dofEnd; ++j) {
            if (out.blockOfColumn[static_cast<size_t>(j)] != -1)
                throw std::runtime_error("aggregateAbaqusStyle: DOF 重复归属");
            out.blockOfColumn[static_cast<size_t>(j)] = i;
        }
        out.maxOwnedDofs = std::max(out.maxOwnedDofs, ond.dofEnd - ond.dofBegin);
    }
    for (int64_t j = 0; j < nTotalDofs; ++j)
        if (out.blockOfColumn[static_cast<size_t>(j)] == -1)
            throw std::runtime_error("aggregateAbaqusStyle: DOF 未覆盖");
    for (const int64_t r : mf.roots)
        out.roots.push_back(newId[static_cast<size_t>(r)]);
    // 新块号就是按后序分配的（newId[post[i]]=i），新 id 下的后序即恒等序列；
    // 直接存旧 id 会越界（旧 id 可超出存活块数）。
    out.postorder.resize(static_cast<size_t>(nb));
    std::iota(out.postorder.begin(), out.postorder.end(), 0);
    return out;
}

}  // namespace

BlockForest aggregateAbaqusStyle(const BlockForest& base,
                                 const AggregationParams& p) {
    if (base.blocks.empty()) return base;
    const int64_t nTotal =
        static_cast<int64_t>(base.blockOfColumn.size());
    MutableForest mf = fromBlocks(base);

    // ---- Pass 1: AggregateLeaves（Abaqus 语义，[R] 54412）----
    // 对每个"全部子都是叶子"的节点 P：若 P.size + Σchild.size 不超过
    // leafAggrMaxDofs（且 ≤ maxBlockDofs），把全部叶子子吸收进 P，P 变叶。
    if (p.aggregateLeaves) {
        // 自底向上（后序）：后序 id 即 blocks 顺序中父在子后。
        // 反复直到不动点（吸收后新变叶的父可能继续被其上父吸收）。
        bool changed = true;
        int guard = 0;
        while (changed && guard++ < 10000) {
            changed = false;
            if (std::getenv("AMLS_DUMP_TREE"))
                std::fprintf(stderr, "[rea-aggr] leaves pass guard=%d\n", guard);
            for (int64_t i = static_cast<int64_t>(mf.nodes.size()) - 1; i >= 0; --i) {
                auto& nd = mf.nodes[static_cast<size_t>(i)];
                if (nd.dead || nd.parent < 0) continue;  // 只处理非根
                if (nd.children.empty()) continue;        // 已是叶
                bool allLeaves = true;
                int64_t total = nd.dofEnd - nd.dofBegin;
                // Abaqus AggregateLeaves 只合并"与父区间连续邻接"的叶子子块：
                // 子块区间必须紧贴父区间（子块 dofEnd == 父 dofBegin 或
                // 父 dofEnd == 子 dofBegin），否则合并会产生区间空洞/重叠。
                // 对后序编号（父 id > 子 id）且消去树父列在子列之后，
                // 子块区间在父块之前：要求子 dofEnd == 父 dofBegin。
                int64_t minLo = nd.dofBegin;
                bool contiguous = true;
                for (const int64_t c : nd.children) {
                    const auto& cn = mf.nodes[static_cast<size_t>(c)];
                    if (cn.dead) continue;
                    if (!cn.children.empty()) { allLeaves = false; break; }
                    if (cn.dofEnd != nd.dofBegin) { contiguous = false; break; }
                    total += cn.dofEnd - cn.dofBegin;
                    minLo = std::min(minLo, cn.dofBegin);
                }
                if (!allLeaves || !contiguous) continue;
                if (total > p.leafAggrMaxDofs && p.leafAggrMaxDofs > 0) continue;
                if (total > p.maxBlockDofs) continue;
                if (p.leafAggrMaxDofs <= 0) continue;
                // 吸收: P 的区间扩展到覆盖全部叶（叶区间连续且紧贴 P 之前）。
                int64_t lo = nd.dofBegin;
                for (const int64_t c : nd.children) {
                    const auto& cn = mf.nodes[static_cast<size_t>(c)];
                    lo = std::min(lo, cn.dofBegin);
                }
                nd.dofBegin = lo;
                for (const int64_t c : nd.children) mf.nodes[static_cast<size_t>(c)].dead = true;
                nd.children.clear();   // 变叶
                changed = true;
            }
        }
    }

    // ---- Pass 2: Collapse2 单子链折叠（Abaqus 语义，[R] 101821）----
    // 对"恰 1 个子"的内部节点 X：若 X.size + 子.size ≤ collapseMaxDofs
    // （且 ≤ maxBlockDofs），把子吸收进 X（X 区间扩展到覆盖子），
    // X 保留子之子的孩子；循环至不动点。
    if (p.collapseMode != 0) {
        bool changed = true;
        int guard = 0;
        while (changed && guard++ < 10000) {
            changed = false;
            int64_t nCollapse = 0;
            if (std::getenv("AMLS_DUMP_TREE"))
                std::fprintf(stderr, "[rea-aggr] collapse pass guard=%d\n", guard);
            for (int64_t i = 0; i < static_cast<int64_t>(mf.nodes.size()); ++i) {
                auto& nd = mf.nodes[static_cast<size_t>(i)];
                if (nd.dead) continue;
                if (nd.children.size() != 1) continue;
                const int64_t c = nd.children[0];
                auto& cn = mf.nodes[static_cast<size_t>(c)];
                if (cn.dead) continue;
                // 连续约束：子块区间必须紧贴父块（消去树父列在子列之后，
                // 重排坐标下子区间在父区间之前且相邻）。不连续则跳过——
                // 合并会产生区间空洞/重叠，破坏 DOF 连续段约定。
                if (cn.dofEnd != nd.dofBegin) continue;
                const int64_t combined = nd.size() + cn.size();
                if (p.collapseMaxDofs > 0 && combined > p.collapseMaxDofs) continue;
                if (combined > p.maxBlockDofs) continue;
                // 吸收子：区间 [cn.dofBegin, nd.dofEnd)，孩子的孩子成为 X 的孩子。
                // 只更新仍存活的孩子（死节点忽略）。
                nd.dofBegin = cn.dofBegin;
                std::vector<int64_t> liveKids;
                for (const int64_t gc : cn.children)
                    if (!mf.nodes[static_cast<size_t>(gc)].dead) {
                        mf.nodes[static_cast<size_t>(gc)].parent = i;
                        liveKids.push_back(gc);
                    }
                nd.children = std::move(liveKids);
                cn.dead = true;
                changed = true;
                ++nCollapse;
            }
            if (std::getenv("AMLS_DUMP_TREE") && nCollapse > 0)
                std::fprintf(stderr, "[rea-aggr] collapse pass %d: %lld collapses\n",
                             guard, (long long)nCollapse);
        }
    }

    if (std::getenv("AMLS_DUMP_TREE")) {
        int64_t alive = 0;
        for (const auto& nd : mf.nodes)
            if (!nd.dead) ++alive;
        std::fprintf(stderr, "[rea-aggr] pre-toBlocks alive=%lld total=%lld\n",
                     (long long)alive, (long long)mf.nodes.size());
    }

    BlockForest out = toBlocks(mf, nTotal);
    // 残留校验：最大块 ≤ maxBlockDofs
    for (const auto& blk : out.blocks)
        if (blk.end - blk.begin > p.maxBlockDofs)
            throw std::runtime_error("aggregateAbaqusStyle: 聚合后块超限");
    return out;
}

AggregationStats aggregationStats(const BlockForest& forest) {
    AggregationStats s;
    s.blocks = static_cast<int64_t>(forest.blocks.size());
    s.roots = static_cast<int64_t>(forest.roots.size());
    std::vector<int64_t> depth(static_cast<size_t>(s.blocks), 0);
    std::unordered_map<int64_t, int64_t> hist;
    // 深度须自顶向下传播：逆后序（父先子后）遍历
    for (auto rit = forest.postorder.rbegin(); rit != forest.postorder.rend();
         ++rit) {
        const int64_t id = *rit;
        const auto& blk = forest.blocks[static_cast<size_t>(id)];
        const int64_t sz = blk.end - blk.begin;
        hist[sz]++;
        if (blk.children.empty()) s.leaves++;
        if (blk.children.size() == 1) s.singleChildren++;
        for (const int64_t c : blk.children)
            depth[static_cast<size_t>(c)] = depth[static_cast<size_t>(id)] + 1;
    }
    for (const int64_t d : depth) s.maxDepth = std::max(s.maxDepth, d);
    s.sizeHistogram.assign(hist.begin(), hist.end());
    std::sort(s.sizeHistogram.begin(), s.sizeHistogram.end());
    return s;
}

void dumpAggregationStats(const BlockForest& forest, const char* tag) {
    AggregationStats s = aggregationStats(forest);
    std::fprintf(stderr,
                 "[rea-aggr] %s: blocks=%lld roots=%lld leaves=%lld "
                 "single=%lld maxDepth=%lld\n",
                 tag, (long long)s.blocks, (long long)s.roots,
                 (long long)s.leaves, (long long)s.singleChildren,
                 (long long)s.maxDepth);
    std::fprintf(stderr, "[rea-aggr] %s size-dist:", tag);
    for (const auto& [sz, cnt] : s.sizeHistogram)
        if (cnt != 0)
            std::fprintf(stderr, " %lld:%lld", (long long)sz, (long long)cnt);
    std::fprintf(stderr, "\n");
}

}  // namespace sym
}  // namespace amls