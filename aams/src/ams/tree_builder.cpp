#include "ams/tree_builder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <metis.h>

#include "partition/supernode_builder.hpp"
#include "partition/symbolic_analyzer.hpp"
#include "ams/aggregation.hpp"

namespace aams {
namespace rea {

namespace {

// 与 metis_partitioner.cpp extractGraphFast 一致的无向 CSR 提取（idx_t 类型，
// 必须匹配 METIS_NodeND 的 idx_t* 参数；否则指针步进错乱导致段错误）。
struct Csr {
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
};
Csr extractGraph(const DefaultTraits::SparseMatrix& A) {
    const int64_t n = A.rows();
    DefaultTraits::SparseMatrix compressed = A;
    compressed.makeCompressed();
    const auto& C = compressed;
    // 行度计数（off-diag (r,c) 贡献 c 到 r 的邻接）
    std::vector<idx_t> rdeg(static_cast<size_t>(n) + 1, 0);
    for (int64_t c = 0; c < n; ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(C, c); it; ++it)
            if (it.row() != c)
                ++rdeg[static_cast<size_t>(it.row()) + 1];
    for (int64_t i = 0; i < n; ++i)
        rdeg[static_cast<size_t>(i) + 1] += rdeg[static_cast<size_t>(i)];
    std::vector<idx_t> radj(static_cast<size_t>(rdeg.back()));
    std::vector<idx_t> cur(rdeg.begin(), rdeg.end() - 1);
    for (int64_t c = 0; c < n; ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(C, c); it; ++it)
            if (it.row() != c)
                radj[static_cast<size_t>(cur[static_cast<size_t>(it.row())]++)] =
                    static_cast<idx_t>(c);
    // adj(v) = 列 v 行号 ∪ radj[v]，两侧各自升序 → 归并去重
    Csr g;
    g.xadj.assign(static_cast<size_t>(n) + 1, 0);
    for (int64_t v = 0; v < n; ++v) {
        const int64_t cb = C.outerIndexPtr()[v], ce = C.outerIndexPtr()[v + 1];
        const idx_t rb = rdeg[static_cast<size_t>(v)],
                      re = rdeg[static_cast<size_t>(v) + 1];
        int64_t cnt = 0;
        int64_t p = cb, q = rb;
        while (p < ce || q < re) {
            idx_t val;
            const idx_t colVal = p < ce ? C.innerIndexPtr()[p] : idx_t(-1);
            const idx_t rowVal = q < re ? radj[static_cast<size_t>(q)] : idx_t(-1);
            if (q >= re || (p < ce && colVal < rowVal)) { val = colVal; ++p; }
            else if (p >= ce || rowVal < colVal)        { val = rowVal; ++q; }
            else                                        { val = colVal; ++p; ++q; }
            if (val == static_cast<idx_t>(v)) continue;   // 去自环
            g.adjncy.push_back(val);
            ++cnt;
        }
        g.xadj[static_cast<size_t>(v) + 1] =
            g.xadj[static_cast<size_t>(v)] + static_cast<idx_t>(cnt);
    }
    return g;
}

// 网格节点图：把连续 dp 个 DOF 收缩为一个顶点（同一网格节点的 DOF 通过
// 泊松耦合两两相连，构成 dp×dp 稠密对角块）。边双向收集后逐顶点去重。
Csr extractNodeGraph(const DefaultTraits::SparseMatrix& A, int64_t dp,
                     int64_t nv) {
    DefaultTraits::SparseMatrix compressed = A;
    compressed.makeCompressed();
    std::vector<std::vector<idx_t>> adj(static_cast<size_t>(nv));
    for (int64_t c = 0; c < compressed.rows(); ++c) {
        const int64_t nc = c / dp;
        for (DefaultTraits::SparseMatrix::InnerIterator it(compressed, c); it;
             ++it) {
            const int64_t r = it.row();
            if (r == c) continue;
            const int64_t nr = r / dp;
            if (nr == nc) continue;
            adj[static_cast<size_t>(nr)].push_back(static_cast<idx_t>(nc));
            adj[static_cast<size_t>(nc)].push_back(static_cast<idx_t>(nr));
        }
    }
    Csr g;
    g.xadj.assign(static_cast<size_t>(nv) + 1, 0);
    for (int64_t v = 0; v < nv; ++v) {
        auto& a = adj[static_cast<size_t>(v)];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        g.adjncy.insert(g.adjncy.end(), a.begin(), a.end());
        g.xadj[static_cast<size_t>(v) + 1] =
            static_cast<idx_t>(g.adjncy.size());
    }
    return g;
}

// 连续 dp 分组布局检测：n % dp == 0 且组内耦合边覆盖率 ≥99%。
// 不要求组内两两全耦合：Abaqus 导出的 mtx 会丢弃数值极小的跨向耦合
// （cant_20 中 1495/8820 节点缺 (0,1)/(0,2) 对、7 组全缺），缺失的耦合
// 由符号分析阶段的消元填充补回，不影响 3-DOF 超节点形成。
bool detectGroupedLayout(const DefaultTraits::SparseMatrix& A, int64_t dp) {
    const int64_t n = A.rows();
    if (n % dp != 0) return false;
    std::vector<uint8_t> hasEdge(static_cast<size_t>(n / dp), 0);
    for (int64_t c = 0; c < n; ++c) {
        const int64_t t = c / dp;
        if (hasEdge[static_cast<size_t>(t)]) continue;
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
            if (it.row() / dp == t && it.row() != c) {
                hasEdge[static_cast<size_t>(t)] = 1;
                break;
            }
    }
    int64_t nEmpty = 0;
    for (uint8_t e : hasEdge)
        if (!e) ++nEmpty;
    // 容忍 ≤1% 的孤立组（导出丢值），仍有组内边的组 ≥99% 即认定分组布局
    return nEmpty * 100 <= static_cast<int64_t>(hasEdge.size());
}


}  // namespace

IPartitioner<DefaultTraits>::Result buildAbaqusTree(
    const DefaultTraits::SparseMatrix& A, const AbaqusTreeOptions& opt) {
    using Result = IPartitioner<DefaultTraits>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0) return res;

    // ---- 1. 网格节点图收缩 + METIS_NodeND 排列（GenerateOrder ↣ NodeND）----
    // Abaqus 在网格节点图上跑 NodeND（stf 块以 3-DOF 为主），DOF 排列 =
    // 节点排列按 dp 展开成 perm[new]=old。
    int dp = opt.dofsPerNode;
    if (dp <= 0) {
        dp = 1;
        if (n % 3 == 0 && detectGroupedLayout(A, 3)) dp = 3;
        if (std::getenv("AMLS_DUMP_TREE"))
            std::fprintf(stderr, "[rea-tree] 自动节点收缩宽度 dp=%d\n", dp);
    } else if (dp > 1 &&
               !(n % dp == 0 && detectGroupedLayout(A, dp))) {
        throw std::runtime_error(
            "rea::buildAbaqusTree: dofsPerNode=" + std::to_string(dp) +
            " 布局校验失败（n 不整除或组内 dofs 未两两耦合）");
    }
    const int64_t nv = dp > 1 ? n / dp : n;
    Csr g = dp > 1 ? extractNodeGraph(A, dp, nv) : extractGraph(A);
    idx_t nvtxs = static_cast<idx_t>(nv);
    std::vector<idx_t> mperm(static_cast<size_t>(nv)), iperm(static_cast<size_t>(nv));
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = opt.metisSeed;
    const int r = METIS_NodeND(&nvtxs, g.xadj.data(), g.adjncy.data(),
                               nullptr, opts, mperm.data(), iperm.data());
    if (r != METIS_OK)
        throw std::runtime_error("rea::buildAbaqusTree: METIS_NodeND 失败");
    std::vector<int64_t> permND(static_cast<size_t>(n));
    if (dp > 1) {
        for (int64_t pos = 0; pos < nv; ++pos)
            for (int64_t d = 0; d < dp; ++d)
                permND[static_cast<size_t>(dp * pos + d)] =
                    dp * static_cast<int64_t>(mperm[static_cast<size_t>(pos)]) + d;
    } else {
        for (int64_t v = 0; v < n; ++v)
            permND[static_cast<size_t>(v)] = mperm[static_cast<size_t>(v)];
    }

    // ---- 2. 符号分析（BuildTreeAfterMetis ↣ etree + columnCount）----
    // dp>1 时在网格节点层做符号分析：A_n = Rᵀ A R（dp×dp dof 块求和；
    // R 为节点→dof 聚合，A SPD ⇒ A_n SPD）。exact 超节点以节点为原子，
    // 块大小均为 dp 的倍数——与 Abaqus stf 分布（cant_20:
    // 3:6739/6:1033/9:70/12:48/18:5/24:3，全为 3 的倍数）一致。若在 dof
    // 层做消除分析，导出 mtx 丢失的组内耦合会让三重奏被 exact 检测拆开。
    sym::SymbolicAnalysis sa;
    if (dp > 1) {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(A.nonZeros()));
        for (int64_t c = 0; c < n; ++c)
            for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
                trips.emplace_back(it.row() / dp, c / dp, it.value());
        DefaultTraits::SparseMatrix An(nv, nv);
        An.setFromTriplets(trips.begin(), trips.end());
        std::vector<int64_t> nodePerm(static_cast<size_t>(nv));
        for (int64_t v = 0; v < nv; ++v)
            nodePerm[static_cast<size_t>(v)] = mperm[static_cast<size_t>(v)];
        if (std::getenv("AMLS_DUMP_TREE"))
            std::fprintf(stderr,
                         "[rea-tree] step2 analyze (node level, nv=%lld)\n",
                         (long long)nv);
        sa = sym::analyzeSymmetrized(An, nodePerm);
    } else {
        if (std::getenv("AMLS_DUMP_TREE"))
            std::fprintf(stderr, "[rea-tree] step2 analyze\n");
        sa = sym::analyzeSymmetrized(A, permND);
    }
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step2 done etree size=%lld\n",
                     (long long)sa.parentColumn.size());

    // ---- 3. exact 基础超节点（CreateFundamentalSuperNodes ↣ buildBlockForest）----
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step3 base blocks\n");
    // dp>1: 块上限按节点数换算（maxBlockDofs 为 dof 单位）
    sym::BlockForest base = sym::buildBlockForest(
        sa, dp > 1 ? std::max<int64_t>(1, opt.maxBlockDofs / dp)
                   : opt.maxBlockDofs);
    if (dp > 1) {
        // 节点块 [b,e) → dof 块 [dp*b, dp*e)（节点排列展开后连续）；
        // blockOfColumn 同步重建为 dof 层（聚合 Pass 以它取 nTotal）。
        for (auto& blk : base.blocks) {
            blk.begin *= dp;
            blk.end *= dp;
        }
        base.blockOfColumn.assign(static_cast<size_t>(n), -1);
        for (int64_t b = 0; b < static_cast<int64_t>(base.blocks.size()); ++b)
            for (int64_t j = base.blocks[static_cast<size_t>(b)].begin;
                 j < base.blocks[static_cast<size_t>(b)].end; ++j)
                base.blockOfColumn[static_cast<size_t>(j)] = b;
    }
    if (std::getenv("AMLS_DUMP_TREE")) {
        std::fprintf(stderr, "[rea-tree] step3 done base blocks=%lld\n",
                     (long long)base.blocks.size());
        sym::dumpAggregationStats(base, "base");
    }

    // ---- 4. Abaqus 聚合（Collapse2 + AggregateLeaves + Cutoff）----
    sym::AggregationParams ap;
    ap.collapseMaxDofs = opt.collapseMaxDofs;
    ap.leafAggrMaxDofs = opt.leafAggrMaxDofs;
    ap.maxBlockDofs   = opt.maxBlockDofs;
    ap.minBlockDofs   = opt.minBlockDofs;
    ap.collapseMode   = opt.collapseMode;
    ap.aggregateLeaves = opt.aggregateLeaves;
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step4 aggregate\n");
    sym::BlockForest forest = sym::aggregateAbaqusStyle(base, ap);
    if (std::getenv("AMLS_DUMP_TREE"))
        sym::dumpAggregationStats(forest, "aggregated");

    // ---- 5. 块森林 → SeparatorTree（原始坐标）----
    // 多根处理：SeparatorTree 只支持单根（postOrder/validate 依赖）。
    // Abaqus 支持森林（numberOfTrees），但后续 solver 需要单根；
    // 这里把多个根合并为一个超根。超根是纯结构合并点，indices 必须为空——
    // 各根的 DOF 仍归各根节点所有，若超根再收一遍会造成 DOF 重复归属。
    // opt.superRootDofs>0 仅是显式授权开关（0 = 拒绝合成，直接报错）。
    int64_t nb = static_cast<int64_t>(forest.blocks.size());
    bool needSuperRoot = static_cast<int64_t>(forest.roots.size()) > 1;
    if (needSuperRoot && opt.superRootDofs <= 0)
        throw std::runtime_error(
            "rea::buildAbaqusTree: 多根森林需要 superRootDofs>0（合成单根）");
    std::vector<int64_t> nodeOfBlock(static_cast<size_t>(nb), -1);
    // 先建普通块节点
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        std::vector<int64_t> idx;
        idx.reserve(static_cast<size_t>(blk.end - blk.begin));
        for (int64_t j = blk.begin; j < blk.end; ++j)
            idx.push_back(permND[static_cast<size_t>(j)]);
        const NodeRole role =
            blk.children.empty() ? NodeRole::Leaf : NodeRole::Separator;
        nodeOfBlock[static_cast<size_t>(b)] =
            res.tree.addNode(role, std::move(idx));
    }
    // 父/子关系
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        if (blk.children.empty()) continue;
        std::vector<int64_t> kids;
        kids.reserve(blk.children.size());
        for (const int64_t c : blk.children)
            kids.push_back(nodeOfBlock[static_cast<size_t>(c)]);
        for (const int64_t c : kids)
            res.tree.setParent(c, nodeOfBlock[static_cast<size_t>(b)]);
        res.tree.setChildren(nodeOfBlock[static_cast<size_t>(b)], std::move(kids));
    }
    // 多根合成超根（空 indices 的结构合并点）
    if (needSuperRoot) {
        std::vector<int64_t> rootKids;
        rootKids.reserve(forest.roots.size());
        for (const int64_t r : forest.roots)
            rootKids.push_back(nodeOfBlock[static_cast<size_t>(r)]);
        const int64_t superId =
            res.tree.addNode(NodeRole::Root, std::vector<int64_t>{});
        for (const int64_t c : rootKids)
            res.tree.setParent(c, superId);
        res.tree.setChildren(superId, std::move(rootKids));
    }
    res.tree.relabelRoot();

    // ---- 6. 后序排列 perm[new]=old（重排坐标 → 原始坐标）----
    // nd.indices 已是原始坐标（step5 存的就是 permND[列]），perm 直接是
    // 后序节点 indices 的顺接；再套一次 permND 会双重映射。
    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (const int64_t id : order) {
        const TreeNode& nd = res.tree.node(static_cast<size_t>(id));
        for (const int64_t j : nd.indices)
            res.perm.push_back(j);
    }

    // ---- 7. 原始 K 全边祖先可比自检 ----
    // A 的边 (r,c) 与节点 indices 同为原始坐标，直接用原始 dof 索引 dofBlock。
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        auto isAncestor = [&](int64_t a, int64_t b) -> bool {
            for (int64_t x = b; x != -1;
                 x = res.tree.node(static_cast<size_t>(x)).parent)
                if (x == a) return true;
            return false;
        };
        std::vector<int64_t> dofBlock(static_cast<size_t>(n), -1);
        for (int64_t id = 0; id < res.tree.nNodes(); ++id)
            for (const int64_t d : res.tree.node(static_cast<size_t>(id)).indices)
                dofBlock[static_cast<size_t>(d)] = id;
        int64_t nBad = 0;
        for (int64_t c = 0; c < A.outerSize(); ++c)
            for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it) {
                const int64_t rr = it.row();
                if (rr == c) continue;
                const int64_t br = dofBlock[static_cast<size_t>(rr)];
                const int64_t bc = dofBlock[static_cast<size_t>(c)];
                if (br == bc) continue;
                if (!isAncestor(br, bc) && !isAncestor(bc, br)) ++nBad;
            }
        std::fprintf(stderr, "[rea-tree] K全边不可比计数=%lld\n",
                     (long long)nBad);
    }
    return res;
}

}  // namespace rea
}  // namespace aams