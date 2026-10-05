#include "partition/partitioner.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <metis.h>

namespace amls {

namespace {

/// 从对称稀疏矩阵抽取无向图 CSR（去重、去自环）,顶点权 1。
struct Csr {
    std::vector<idx_t> xadj, adjncy;
};

Csr extractGraph(const DefaultTraits::SparseMatrix& A) {
    const int64_t n = A.rows();
    std::vector<std::unordered_set<int64_t>> nbr(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it) {
            const int64_t r = it.row();
            if (r == c)
                continue;
            nbr[static_cast<size_t>(r)].insert(c);
            nbr[static_cast<size_t>(c)].insert(r);
        }
    }
    Csr g;
    g.xadj.resize(static_cast<size_t>(n) + 1, 0);
    for (int64_t i = 0; i < n; ++i)
        g.xadj[static_cast<size_t>(i) + 1] =
            g.xadj[static_cast<size_t>(i)] +
            static_cast<idx_t>(nbr[static_cast<size_t>(i)].size());
    g.adjncy.resize(static_cast<size_t>(g.xadj.back()));
    for (int64_t i = 0; i < n; ++i) {
        idx_t p = g.xadj[static_cast<size_t>(i)];
        for (int64_t j : nbr[static_cast<size_t>(i)])
            g.adjncy[static_cast<size_t>(p++)] = static_cast<idx_t>(j);
    }
    return g;
}

/// 诱导子图。
Csr inducedSubgraph(const Csr& g, const std::vector<idx_t>& verts,
                    std::vector<idx_t>& localOf) {
    const idx_t m = static_cast<idx_t>(verts.size());
    localOf.assign(g.xadj.size() - 1, -1);
    for (idx_t i = 0; i < m; ++i)
        localOf[static_cast<size_t>(verts[static_cast<size_t>(i)])] = i;

    Csr sub;
    sub.xadj.resize(static_cast<size_t>(m) + 1, 0);
    std::vector<idx_t> tmp;
    for (idx_t i = 0; i < m; ++i) {
        const idx_t v = verts[static_cast<size_t>(i)];
        tmp.clear();
        for (idx_t p = g.xadj[static_cast<size_t>(v)];
             p < g.xadj[static_cast<size_t>(v) + 1]; ++p) {
            const idx_t u = g.adjncy[static_cast<size_t>(p)];
            if (localOf[static_cast<size_t>(u)] >= 0)
                tmp.push_back(localOf[static_cast<size_t>(u)]);
        }
        std::sort(tmp.begin(), tmp.end());
        sub.xadj[static_cast<size_t>(i) + 1] =
            sub.xadj[static_cast<size_t>(i)] + static_cast<idx_t>(tmp.size());
        sub.adjncy.insert(sub.adjncy.end(), tmp.begin(), tmp.end());
    }
    return sub;
}

struct BisectResult {
    std::vector<idx_t> part;  // 0 / 1 / 2(=分隔)
};

/// 二分 + 顶点分隔:part[] 中 2 = 使 0/1 两子图不连通的分隔顶点。
BisectResult bisect(Csr g) {
    const idx_t n = static_cast<idx_t>(g.xadj.size() - 1);
    BisectResult res;
    res.part.assign(static_cast<size_t>(n), 0);
    if (n == 0)
        return res;

    idx_t nvtxs = n, ncon = 1, nparts = 2, edgecut = 0;
    std::vector<idx_t> where(static_cast<size_t>(n), 0);
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = 42;   // 固定种子:验证可复现
    const int r = METIS_PartGraphRecursive(
        &nvtxs, &ncon, g.xadj.data(), g.adjncy.data(), nullptr, nullptr,
        nullptr, &nparts, nullptr, nullptr, opts, &edgecut, where.data());
    if (r != METIS_OK)
        throw std::runtime_error("METIS_PartGraphRecursive 失败");

    // 由划分计算顶点分隔（METIS 原生接口;part[] 输出 0/1/2）
    idx_t sepsize = 0;
    const int r2 = METIS_ComputeVertexSeparator(
        &nvtxs, g.xadj.data(), g.adjncy.data(), nullptr, opts, &sepsize,
        res.part.data());
    if (r2 != METIS_OK)
        throw std::runtime_error("METIS_ComputeVertexSeparator 失败");
    return res;
}

} // namespace

template <typename T>
typename IPartitioner<T>::Result
MetisPartitioner<T>::recursiveBisection(const typename T::SparseMatrix& A,
                                        int nLevels, int64_t leafSize) {
    using Result = typename IPartitioner<T>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0)
        return res;
    Csr g = extractGraph(A);

    // 递归:返回子树根节点 id;verts 为子图顶点（原始全局编号）
    std::function<int64_t(const std::vector<idx_t>&, int)> recurse =
        [&](const std::vector<idx_t>& verts, int depth) -> int64_t {
        const int64_t m = static_cast<int64_t>(verts.size());

        if (m <= leafSize || depth >= nLevels || m < 4) {
            std::vector<int64_t> idx(verts.begin(), verts.end());
            return res.tree.addNode(NodeRole::Leaf, std::move(idx));
        }

        std::vector<idx_t> localOf;
        Csr sub = inducedSubgraph(g, verts, localOf);

        BisectResult b = bisect(sub);
        std::vector<idx_t> side[2], sep;
        for (idx_t i = 0; i < static_cast<idx_t>(verts.size()); ++i) {
            const idx_t p = b.part[static_cast<size_t>(i)];
            if (p == 2)
                sep.push_back(verts[static_cast<size_t>(i)]);
            else
                side[p].push_back(verts[static_cast<size_t>(i)]);
        }

        // 退化防护：分隔为空或某一侧为空/过小 → 不再二分
        const int64_t minSide = std::min(side[0].size(), side[1].size());
        if (sep.empty() || minSide < 2) {
            std::vector<int64_t> idx(verts.begin(), verts.end());
            return res.tree.addNode(NodeRole::Leaf, std::move(idx));
        }

        std::vector<int64_t> sepIdx(sep.begin(), sep.end());
        const int64_t sepId =
            res.tree.addNode(NodeRole::Separator, std::move(sepIdx));
        const int64_t l = recurse(side[0], depth + 1);
        const int64_t r = recurse(side[1], depth + 1);
        res.tree.setChildren(sepId, l, r);
        res.tree.setParent(l, sepId);
        res.tree.setParent(r, sepId);
        return sepId;
    };

    std::vector<idx_t> all(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        all[static_cast<size_t>(i)] = static_cast<idx_t>(i);
    recurse(all, 0);
    res.tree.relabelRoot();

    // 重排列：按后序拼接各节点索引 → perm[new] = old
    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (int64_t id : order) {
        const TreeNode& nd = res.tree.node(id);
        res.perm.insert(res.perm.end(), nd.indices.begin(), nd.indices.end());
    }
    if (!res.tree.validate(n))
        throw std::runtime_error("MetisPartitioner: 树校验失败");
    return res;
}

template class MetisPartitioner<DefaultTraits>;

} // namespace amls
