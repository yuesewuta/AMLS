#include "partition/partitioner.hpp"
#include "partition/symbolic_analyzer.hpp"
#include "partition/supernode_builder.hpp"

#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

/// 快速无向图 CSR：列结构 + 一次计数排序转置合并（对任意存储模式正确，
/// 不假设对称存储；O(nnz) 无哈希）。NodeND 路径专用——递归二分路径
/// 保留原实现以固定基线。
Csr extractGraphFast(const DefaultTraits::SparseMatrix& A) {
    const int64_t n = A.rows();
    // outerIndexPtr 的相邻边界仅在压缩存储下表示有效元素区间。
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

template <typename T>
typename IPartitioner<T>::Result
MetisPartitioner<T>::nestedDissection(const typename T::SparseMatrix& A,
                                      int64_t leafSize) {
    using Result = typename IPartitioner<T>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0)
        return res;
    if (leafSize < 1)
        leafSize = 1;
    Csr g = extractGraphFast(A);

    // ---- 单次 NodeND：mperm[new] = old，iperm[old] = new
    idx_t nvtxs = static_cast<idx_t>(n);
    std::vector<idx_t> mperm(static_cast<size_t>(n)),
        iperm(static_cast<size_t>(n));
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = 42;
    const int r = METIS_NodeND(&nvtxs, g.xadj.data(), g.adjncy.data(),
                               nullptr, opts, mperm.data(), iperm.data());
    if (r != METIS_OK)
        throw std::runtime_error("METIS_NodeND 失败");

    // ---- 消去树（Liu 算法，作用于 ND 位置空间的图；填充边只在
    // 消去树祖先链上 ⇒ 按消去树切分的子树天然满足"只耦合祖先"）----
    // METIS 的 iperm 是顶点到消去位置的映射（ometis.c）。
    std::vector<int64_t> posOf(static_cast<size_t>(n)),
        atPos(static_cast<size_t>(n));
    for (int64_t v = 0; v < n; ++v) {
        const int64_t p =
            static_cast<int64_t>(iperm[static_cast<size_t>(v)]);
        posOf[static_cast<size_t>(v)] = p;
        atPos[static_cast<size_t>(p)] = v;
    }
    std::vector<int64_t> parent(static_cast<size_t>(n), -1),
        ancestor(static_cast<size_t>(n), -1);
    for (int64_t j = 0; j < n; ++j) {
        const idx_t v = static_cast<idx_t>(atPos[static_cast<size_t>(j)]);
        for (idx_t p = g.xadj[static_cast<size_t>(v)];
             p < g.xadj[static_cast<size_t>(v) + 1]; ++p) {
            const int64_t k = posOf[static_cast<size_t>(g.adjncy[static_cast<size_t>(p)])];
            if (k >= j)
                continue;
            int64_t x = k;
            while (ancestor[static_cast<size_t>(x)] != -1 &&
                   ancestor[static_cast<size_t>(x)] != j) {
                const int64_t nx = ancestor[static_cast<size_t>(x)];
                ancestor[static_cast<size_t>(x)] = j;
                x = nx;
            }
            if (ancestor[static_cast<size_t>(x)] == -1) {
                parent[static_cast<size_t>(x)] = j;
                ancestor[static_cast<size_t>(x)] = j;
            }
        }
    }
    std::vector<std::vector<int64_t>> echild(static_cast<size_t>(n));
    std::vector<int64_t> eroots;
    for (int64_t j = 0; j < n; ++j) {
        if (parent[static_cast<size_t>(j)] >= 0)
            echild[static_cast<size_t>(parent[static_cast<size_t>(j)])]
                .push_back(j);
        else
            eroots.push_back(j);
    }
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        int64_t maxDeg = 0, maxDegId = -1;
        for (int64_t j = 0; j < n; ++j) {
            const int64_t d =
                static_cast<int64_t>(echild[static_cast<size_t>(j)].size());
            if (d > maxDeg) {
                maxDeg = d;
                maxDegId = j;
            }
        }
        std::fprintf(stderr,
                     "[tree] etree roots=%lld maxDeg=%lld at pos=%lld\n",
                     static_cast<long long>(eroots.size()),
                     static_cast<long long>(maxDeg),
                     static_cast<long long>(maxDegId));
    }

    // ---- 按 ≥leafSize 合并（amalgamation）：后序累袋，袋满切出超节点。
    // 袋 = 子树内尚未切走的顶点段（消去树连通段，作为分割集合有效：
    // 其下各切出子树只经袋/更高祖先耦合）。----
    struct Sup {
        std::vector<int64_t> verts;      // 原图顶点号
        std::vector<int64_t> children;   // 超节点 id
        bool dead = false;
    };
    std::vector<Sup> sups;
    sups.reserve(static_cast<size_t>(n / leafSize + eroots.size() + 2));
    std::vector<std::vector<int64_t>> bags(static_cast<size_t>(n));
    std::vector<std::vector<int64_t>> cuts(static_cast<size_t>(n));
    std::vector<int64_t> topSups;
    for (const int64_t er : eroots) {
        // 迭代后序（先取子再自增，push_back 重分配不悬垂栈顶引用）
        std::vector<int64_t> post;
        std::vector<std::pair<int64_t, size_t>> st{{er, 0}};
        while (!st.empty()) {
            auto& top = st.back();
            if (top.second < echild[static_cast<size_t>(top.first)].size()) {
                const int64_t c =
                    echild[static_cast<size_t>(top.first)][top.second];
                ++top.second;
                st.push_back({c, 0});
            } else {
                post.push_back(top.first);
                st.pop_back();
            }
        }
        for (const int64_t u : post) {
            std::vector<int64_t> bag(1, atPos[static_cast<size_t>(u)]);
            std::vector<int64_t> cut;
            for (const int64_t c : echild[static_cast<size_t>(u)]) {
                auto& cb = bags[static_cast<size_t>(c)];
                bag.insert(bag.end(), cb.begin(), cb.end());
                cb.clear();
                auto& cc = cuts[static_cast<size_t>(c)];
                cut.insert(cut.end(), cc.begin(), cc.end());
                cc.clear();
            }
            if (static_cast<int64_t>(bag.size()) >= leafSize &&
                parent[static_cast<size_t>(u)] != -1) {
                Sup s;
                s.verts = std::move(bag);
                s.children = std::move(cut);
                cuts[static_cast<size_t>(u)] = {
                    static_cast<int64_t>(sups.size())};
                sups.push_back(std::move(s));
            } else {
                bags[static_cast<size_t>(u)] = std::move(bag);
                cuts[static_cast<size_t>(u)] = std::move(cut);
            }
        }
        // 消去树顶：袋非空（至少含根位置顶点）→ 顶层超节点
        Sup s;
        s.verts = std::move(bags[static_cast<size_t>(er)]);
        s.children = std::move(cuts[static_cast<size_t>(er)]);
        topSups.push_back(static_cast<int64_t>(sups.size()));
        sups.push_back(std::move(s));
    }
    // 不连通图（多棵消去树）：顶层超节点并入单一根（各连通分量间
    // 本无耦合，合并有效）
    if (topSups.size() > 1) {
        Sup root;
        for (const int64_t t : topSups) {
            Sup& s = sups[static_cast<size_t>(t)];
            root.verts.insert(root.verts.end(), s.verts.begin(),
                              s.verts.end());
            root.children.insert(root.children.end(), s.children.begin(),
                                 s.children.end());
            s.dead = true;
        }
        sups.push_back(std::move(root));
    }

    // ---- 单子链向上合并（内点 ≥2 子；收敛于无单子节点）----
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& s : sups) {
            if (s.dead || s.children.size() != 1)
                continue;
            Sup& c = sups[static_cast<size_t>(s.children[0])];
            s.verts.insert(s.verts.end(), c.verts.begin(), c.verts.end());
            s.children = std::move(c.children);
            c.dead = true;
            changed = true;
        }
    }
    const int64_t rootSup = static_cast<int64_t>(sups.size()) - 1;

    // ---- 超节点树 → SeparatorTree（后序建点，叶 = 无子超节点）----
    std::function<int64_t(int64_t)> emit = [&](int64_t sid) -> int64_t {
        const Sup& s = sups[static_cast<size_t>(sid)];
        std::vector<int64_t> kids;
        kids.reserve(s.children.size());
        for (const int64_t c : s.children)
            kids.push_back(emit(c));
        const int64_t id = res.tree.addNode(
            kids.empty() ? NodeRole::Leaf : NodeRole::Separator,
            std::vector<int64_t>(s.verts));
        if (!kids.empty()) {
            res.tree.setChildren(id, std::move(kids));
            for (const int64_t c : res.tree.node(id).children)
                res.tree.setParent(c, id);
        }
        return id;
    };
    emit(rootSup);
    res.tree.relabelRoot();

    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (int64_t id : order) {
        const TreeNode& nd = res.tree.node(id);
        res.perm.insert(res.perm.end(), nd.indices.begin(), nd.indices.end());
    }
    if (!res.tree.validate(n))
        throw std::runtime_error("MetisPartitioner(nestedDissection): 树校验失败");
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        int64_t maxSize = 0, sumLeaf = 0, nLeaf = 0, maxDepth = 0;
        for (int64_t id = 0; id < res.tree.nNodes(); ++id) {
            const TreeNode& nd = res.tree.node(id);
            maxSize = std::max(maxSize,
                               static_cast<int64_t>(nd.indices.size()));
            maxDepth = std::max(
                maxDepth,
                static_cast<int64_t>(res.tree.ancestors(id).size()));
            if (nd.role == NodeRole::Leaf) {
                sumLeaf += static_cast<int64_t>(nd.indices.size());
                ++nLeaf;
            }
        }
        std::fprintf(stderr,
                     "[tree] nd nodes=%lld maxSize=%lld leafAvg=%lld nLeaf=%lld "
                     "maxDepth=%lld\n",
                     static_cast<long long>(res.tree.nNodes()),
                     static_cast<long long>(maxSize),
                     static_cast<long long>(nLeaf ? sumLeaf / nLeaf : 0),
                     static_cast<long long>(nLeaf),
                     static_cast<long long>(maxDepth));
        // 规模直方图（诊断顶层超大分割）
        std::vector<int64_t> sizes;
        for (int64_t id = 0; id < res.tree.nNodes(); ++id)
            sizes.push_back(
                static_cast<int64_t>(res.tree.node(id).indices.size()));
        std::sort(sizes.rbegin(), sizes.rend());
        std::fprintf(stderr, "[tree] top-sizes:");
        for (int64_t i = 0; i < static_cast<int64_t>(std::min<int64_t>(12, sizes.size())); ++i)
            std::fprintf(stderr, " %lld",
                         static_cast<long long>(sizes[static_cast<size_t>(i)]));
        std::fprintf(stderr, "\n");
        // 导出 external 树文件（AMLS_DUMP_TREE_FILE=path）:
        //   "角色 子数 子id... | dof 列表" (每节点一行, id 0..N-1 后序)
        if (const char* tf = std::getenv("AMLS_DUMP_TREE_FILE")) {
            std::ofstream out(tf);
            const int64_t N = res.tree.nNodes();
            for (int64_t id = 0; id < N; ++id) {
                const TreeNode& nd = res.tree.node(id);
                if (nd.role == NodeRole::Leaf)
                    out << "L ";
                else if (nd.role == NodeRole::Root)
                    out << "R ";
                else
                    out << "S ";
                out << nd.children.size();
                for (const int64_t c : nd.children) out << " " << c;
                out << " |";
                for (const int64_t d : nd.indices) out << " " << d;
                out << "\n";
            }
            std::fprintf(stderr, "[tree] 已导出树文件 -> %s\n", tf);
        }
    }
    return res;
}

template <typename T>
typename IPartitioner<T>::Result
MetisPartitioner<T>::supernodePartition(const typename T::SparseMatrix& A,
                                        int64_t maxBlockDofs) {
    using Result = typename IPartitioner<T>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0) return res;
    if (maxBlockDofs < 1) maxBlockDofs = 1;

    // ---- 1. METIS_NodeND 排列 ----
    Csr g = extractGraphFast(A);
    idx_t nvtxs = static_cast<idx_t>(n);
    std::vector<idx_t> mperm(static_cast<size_t>(n)),
        iperm(static_cast<size_t>(n));
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = 42;
    const int r = METIS_NodeND(&nvtxs, g.xadj.data(), g.adjncy.data(),
                               nullptr, opts, mperm.data(), iperm.data());
    if (r != METIS_OK)
        throw std::runtime_error("MetisPartitioner(supernode): NodeND 失败");
    std::vector<int64_t> permND(static_cast<size_t>(n));
    for (int64_t v = 0; v < n; ++v)
        permND[static_cast<size_t>(v)] = mperm[static_cast<size_t>(v)];

    // ---- 2. 符号分析（LDL 列结构 → etree + columnCount）----
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[supernode] step2 analyze\n");
    sym::SymbolicAnalysis sa = sym::analyzeSymmetrized(A, permND);

    // ---- 3. exact 超节点分组 + 受限拆分 + 块森林 ----
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[supernode] step3 build\n");
    sym::BlockForest forest = sym::buildBlockForest(sa, maxBlockDofs);
    if (std::getenv("AMLS_DUMP_TREE") != nullptr)
        sym::dumpBlockForestStats(forest);

    // ---- 4. 块森林 → SeparatorTree ----
    const int64_t nb = static_cast<int64_t>(forest.blocks.size());
    // 森林 → 单根：设计文档 §6.2 首版仅支持结构连通输入；若多根, 显式拒绝。
    if (static_cast<int64_t>(forest.roots.size()) != 1)
        throw std::runtime_error(
            "MetisPartitioner(supernode): 多根森林暂不支持（结构不连通）");
    // 后序块 id 即 res.tree 的节点 id 顺序（buildBlockForest 已保证祖先 id>后代）。
    std::vector<int64_t> nodeOfBlock(static_cast<size_t>(nb), -1);
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        std::vector<int64_t> idx;
        idx.reserve(static_cast<size_t>(blk.end - blk.begin));
        // 块区间 [begin,end) 是重排坐标下的连续段；树节点 indices 必须存
        // 原始坐标（与 nestedDissection 一致），solver 的 remapTreeToReordered
        // 负责用 perm 再映射到重排坐标。若存重排坐标会双重映射导致错乱。
        for (int64_t j = blk.begin; j < blk.end; ++j)
            idx.push_back(permND[static_cast<size_t>(j)]);
        const NodeRole role =
            blk.children.empty() ? NodeRole::Leaf : NodeRole::Separator;
        nodeOfBlock[static_cast<size_t>(b)] =
            res.tree.addNode(role, std::move(idx));
    }
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        if (blk.children.empty()) continue;
        std::vector<int64_t> kids;
        kids.reserve(blk.children.size());
        for (const int64_t c : blk.children)
            kids.push_back(nodeOfBlock[static_cast<size_t>(c)]);
        // 先设父再 move（std::move 后 kids 失效）
        for (const int64_t c : kids)
            res.tree.setParent(c, nodeOfBlock[static_cast<size_t>(b)]);
        res.tree.setChildren(nodeOfBlock[static_cast<size_t>(b)], std::move(kids));
    }
    res.tree.relabelRoot();

    // ---- 5. 后序排列 perm[new]=old（重排坐标 → 原始坐标）----
    // nd.indices 已是原始坐标（step4 存的就是 permND[列]），perm 直接是
    // 后序节点 indices 的顺接；再套一次 permND 会双重映射。
    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (const int64_t id : order) {
        const TreeNode& nd = res.tree.node(id);
        for (const int64_t j : nd.indices)
            res.perm.push_back(j);
    }
    // ---- 4.5 原始 K 全边祖先可比自检（复现 OriginalBlocks 约束）----
    // A 的边 (r,c) 与节点 indices 同为原始坐标，直接用原始 dof 索引 dofBlock。
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        auto isAncestor = [&](int64_t a, int64_t b) -> bool {
            for (int64_t x = b; x != -1; x = res.tree.node(static_cast<size_t>(x)).parent)
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
                const int64_t r = it.row();
                if (r == c) continue;
                const int64_t br = dofBlock[static_cast<size_t>(r)];
                const int64_t bc = dofBlock[static_cast<size_t>(c)];
                if (br == bc) continue;
                if (!isAncestor(br, bc) && !isAncestor(bc, br)) {
                    if (nBad < 5)
                        std::fprintf(stderr,
                                     "[supernode] K边不可比: (dof %lld<->%lld) "
                                     "块 %lld<->%lld\n",
                                     (long long)r, (long long)c,
                                     (long long)br, (long long)bc);
                    ++nBad;
                }
            }
        std::fprintf(stderr, "[supernode] K全边不可比计数=%lld\n",
                     (long long)nBad);
        // 精确诊断 dof 36/37/39/60 的块归属与父链
        auto chainOf = [&](int64_t dof) {
            int64_t b = dofBlock[static_cast<size_t>(dof)];
            std::fprintf(stderr, "[supernode] dof%lld -> blk%lld chain:",
                         (long long)dof, (long long)b);
            for (int64_t x = b; x != -1; x = res.tree.node(static_cast<size_t>(x)).parent)
                std::fprintf(stderr, " %lld", (long long)x);
            std::fprintf(stderr, "\n");
        };
        for (int64_t d : {36, 37, 39, 60}) chainOf(d);
        // 诊断: 输入 A 的 (43,45)/(39,36) 原始边与重排映射
        auto hasEdge = [&](int64_t u, int64_t v) {
            for (int64_t c : {u, v}) {
                if (c < 0 || c >= A.outerSize()) return false;
                for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
                    if (it.row() == (c == u ? v : u)) return true;
            }
            return false;
        };
        std::fprintf(stderr, "[supernode] A边43-45=%d A边39-36=%d\n",
                     (int)hasEdge(43, 45), (int)hasEdge(39, 36));
        // newOfOld: 原始 dof → 重排位置（perm 的逆）
        std::vector<int64_t> newOfOld(static_cast<size_t>(n), -1);
        for (int64_t newp = 0; newp < n; ++newp)
            newOfOld[static_cast<size_t>(res.perm[static_cast<size_t>(newp)])] = newp;
        std::fprintf(stderr, "[supernode] perm[43]=%lld perm[45]=%lld new[43]=%lld new[45]=%lld\n",
                     (long long)res.perm[static_cast<size_t>(43)],
                     (long long)res.perm[static_cast<size_t>(45)],
                     (long long)newOfOld[static_cast<size_t>(res.perm[static_cast<size_t>(43)])],
                     (long long)newOfOld[static_cast<size_t>(res.perm[static_cast<size_t>(45)])]);
    }
    if (!res.tree.validate(n)) {
        if (std::getenv("AMLS_DUMP_TREE")) {
            // 逐项诊断 validate
            const SeparatorTree& t = res.tree;
            int nRoots = 0;
            for (int64_t i = 0; i < t.nNodes(); ++i)
                if (t.node(i).parent == -1) ++nRoots;
            std::fprintf(stderr, "[supernode] validate fail: nRoots=%lld\n",
                         (long long)nRoots);
            std::vector<int64_t> incoming(static_cast<size_t>(t.nNodes()), 0);
            for (int64_t i = 0; i < t.nNodes(); ++i) {
                const auto& nd = t.node(i);
                for (int64_t c : nd.children) {
                    if (c < 0 || c >= t.nNodes() ||
                        t.node(static_cast<size_t>(c)).parent != nd.id)
                        std::fprintf(stderr,
                                     "[supernode] child link bad: node%lld child%lld "
                                     "childParent=%lld\n",
                                     (long long)i, (long long)c,
                                     (long long)t.node(static_cast<size_t>(c)).parent);
                    ++incoming[static_cast<size_t>(c)];
                }
            }
            for (int64_t i = 0; i < t.nNodes(); ++i) {
                const auto& nd = t.node(i);
                std::fprintf(stderr,
                             "[supernode] node%lld role=%d size=%lld parent=%lld "
                             "kids=%lld in=%lld\n",
                             (long long)i, (int)nd.role,
                             (long long)nd.indices.size(), (long long)nd.parent,
                             (long long)nd.children.size(),
                             (long long)incoming[static_cast<size_t>(i)]);
                if (nd.role == NodeRole::Leaf && !nd.children.empty())
                    std::fprintf(stderr, "  -> LEAF has children!\n");
            }
        }
        throw std::runtime_error("MetisPartitioner(supernode): 树校验失败");
    }
    return res;
}

template class MetisPartitioner<DefaultTraits>;

} // namespace amls
