#include "rea/tree_builder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <metis.h>

#include "partition/supernode_builder.hpp"
#include "partition/symbolic_analyzer.hpp"
#include "rea/aggregation.hpp"

namespace amls {
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

}  // namespace

IPartitioner<DefaultTraits>::Result buildAbaqusTree(
    const DefaultTraits::SparseMatrix& A, const AbaqusTreeOptions& opt) {
    using Result = IPartitioner<DefaultTraits>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0) return res;

    // ---- 1. METIS_NodeND 排列（GenerateOrder ↣ NodeND）----
    Csr g = extractGraph(A);
    idx_t nvtxs = static_cast<idx_t>(n);
    std::vector<idx_t> mperm(static_cast<size_t>(n)), iperm(static_cast<size_t>(n));
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = opt.metisSeed;
    const int r = METIS_NodeND(&nvtxs, g.xadj.data(), g.adjncy.data(),
                               nullptr, opts, mperm.data(), iperm.data());
    if (r != METIS_OK)
        throw std::runtime_error("rea::buildAbaqusTree: METIS_NodeND 失败");
    std::vector<int64_t> permND(static_cast<size_t>(n));
    for (int64_t v = 0; v < n; ++v)
        permND[static_cast<size_t>(v)] = mperm[static_cast<size_t>(v)];

    // ---- 2. 符号分析（BuildTreeAfterMetis ↣ etree + columnCount）----
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step2 analyze\n");
    sym::SymbolicAnalysis sa = sym::analyzeSymmetrized(A, permND);
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step2 done etree size=%lld\n",
                     (long long)sa.parentColumn.size());

    // ---- 3. exact 基础超节点（CreateFundamentalSuperNodes ↣ buildBlockForest）----
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step3 base blocks\n");
    sym::BlockForest base = sym::buildBlockForest(sa, opt.maxBlockDofs);
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
}  // namespace amls