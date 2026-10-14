#include "partition/external_tree.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "partition/partitioner.hpp"

namespace amls {

namespace {

struct NodeSpec {
    NodeRole role;
    std::vector<int64_t> children;
    std::vector<int64_t> dofs;
};

/// 解析一行: "角色 子节点数 子节点id... | DOF列表"
/// 例: "L 0 | 0 1 2"  (叶, 0 子)
///     "R 2 0 1 | 6 7 8"  (根, 2 子: 0,1)
bool parseNode(const std::string& line, NodeSpec& out) {
    const size_t bar = line.find('|');
    if (bar == std::string::npos) return false;
    const std::string head = line.substr(0, bar);
    const std::string tail = line.substr(bar + 1);
    std::istringstream hs(head), ts(tail);
    std::string role;
    if (!(hs >> role)) return false;
    if (role == "L") out.role = NodeRole::Leaf;
    else if (role == "S") out.role = NodeRole::Separator;
    else if (role == "R") out.role = NodeRole::Separator;  // 根角色由 relabelRoot 定
    else return false;
    int64_t nchild = 0;
    if (!(hs >> nchild)) return false;
    out.children.clear();
    int64_t c;
    for (int64_t k = 0; k < nchild; ++k) {
        if (!(hs >> c)) return false;
        out.children.push_back(c);
    }
    // 根(无子,单根叶)⇒ Leaf 角色,与 PKU ND 单节点树语义一致(validate 要求非Leaf≥2子)。
    if (role == "R" && nchild == 0) out.role = NodeRole::Leaf;
    out.dofs.clear();
    int64_t d;
    while (ts >> d) out.dofs.push_back(d);
    return true;
}

}  // namespace

template <typename T>
typename IPartitioner<T>::Result
ExternalTreePartitioner<T>::fromFile(const typename T::SparseMatrix& A,
                                     const std::string& treeFile,
                                     int64_t leafSize) {
    using Result = typename IPartitioner<T>::Result;
    Result res;
    const int64_t n = A.rows();
    if (n == 0) return res;

    // ---- 模式 1: 纯排列文件 (一行 "perm:" 后跟 n 个 old 下标) ----
    {
        std::ifstream f(treeFile);
        std::string first;
        if (f && std::getline(f, first)) {
            std::istringstream ss(first);
            std::string tag;
            ss >> tag;
            if (tag == "perm:") {
                std::vector<int64_t> p;
                int64_t v;
                while (ss >> v) p.push_back(v);
                if (static_cast<int64_t>(p.size()) != n)
                    throw std::runtime_error(
                        "external_tree: perm 长度 != n");
                std::vector<int64_t> seen(static_cast<size_t>(n), -1);
                for (int64_t i = 0; i < n; ++i) {
                    if (p[static_cast<size_t>(i)] < 0 ||
                        p[static_cast<size_t>(i)] >= n ||
                        seen[static_cast<size_t>(p[static_cast<size_t>(i)])] >= 0)
                        throw std::runtime_error(
                            "external_tree: perm 非法(越界或重复)");
                    seen[static_cast<size_t>(p[static_cast<size_t>(i)])] = i;
                }
                // 树由内部 ND 生成, 仅排列被外部覆盖 (注意: ND 树与外部 perm
                // 的 dof 段约定可能冲突, 由调用方保证一致性).
                MetisPartitioner<T> mp;
                Result inner = mp.nestedDissection(A, leafSize);
                res.tree = std::move(inner.tree);
                res.perm = std::move(p);
                return res;
            }
        }
    }

    // ---- 模式 2: 树文件 (每行一个节点) ----
    std::ifstream f(treeFile);
    if (!f)
        throw std::runtime_error("external_tree: 无法打开树文件 " + treeFile);
    std::vector<NodeSpec> nodes;
    std::string line;
    while (std::getline(f, line)) {
        // 去空白/注释
        std::string t = line;
        const size_t c = t.find('#');
        if (c != std::string::npos) t = t.substr(0, c);
        if (t.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        NodeSpec spec;
        if (!parseNode(t, spec))
            throw std::runtime_error("external_tree: 解析失败: " + line);
        nodes.push_back(std::move(spec));
    }
    const int64_t N = static_cast<int64_t>(nodes.size());
    if (N == 0) throw std::runtime_error("external_tree: 空树文件");

    // ---- 校验并建树 ----
    std::vector<int64_t> parent(static_cast<size_t>(N), -1);
    std::vector<int64_t> nChildrenOf(static_cast<size_t>(N), 0);
    std::vector<int64_t> allDofs;
    for (int64_t i = 0; i < N; ++i) {
        const NodeSpec& nd = nodes[static_cast<size_t>(i)];
        // 根(无父)可无子(单根叶, 等价于不剖分); 非根内点必须 >=2 子
        const bool isRoot = (parent[static_cast<size_t>(i)] == -1);
        if (nd.role == NodeRole::Separator && !isRoot && nd.children.size() < 2)
            throw std::runtime_error("external_tree: 内点子数 <2");
        if (!nd.children.empty() && nd.role == NodeRole::Leaf)
            throw std::runtime_error("external_tree: 叶节点不应有子");
        for (const int64_t c : nd.children) {
            if (c < 0 || c >= N)
                throw std::runtime_error("external_tree: 子节点 id 越界");
            if (parent[static_cast<size_t>(c)] != -1)
                throw std::runtime_error("external_tree: 子节点多父");
            parent[static_cast<size_t>(c)] = i;
            ++nChildrenOf[static_cast<size_t>(i)];
        }
        allDofs.insert(allDofs.end(), nd.dofs.begin(), nd.dofs.end());
    }
    // 恰一个根
    int nRoots = 0;
    for (int64_t i = 0; i < N; ++i)
        if (parent[static_cast<size_t>(i)] == -1) ++nRoots;
    if (nRoots != 1)
        throw std::runtime_error("external_tree: 根数 != 1");
    // DOF 完整性: 互不相交、并集 = [0,n)
    std::sort(allDofs.begin(), allDofs.end());
    if (static_cast<int64_t>(allDofs.size()) != n)
        throw std::runtime_error("external_tree: DOF 总数 != n");
    for (int64_t i = 0; i < n; ++i)
        if (allDofs[static_cast<size_t>(i)] != i)
            throw std::runtime_error("external_tree: DOF 集合非法(缺号/重复)");

    // ---- 建 SeparatorTree (id 保持文件顺序) ----
    for (int64_t i = 0; i < N; ++i)
        res.tree.addNode(nodes[static_cast<size_t>(i)].role,
                         nodes[static_cast<size_t>(i)].dofs);
    for (int64_t i = 0; i < N; ++i) {
        const NodeSpec& nd = nodes[static_cast<size_t>(i)];
        if (!nd.children.empty()) {
            res.tree.setChildren(i, nd.children);
            for (const int64_t c : nd.children)
                res.tree.setParent(c, i);
        }
    }
    res.tree.relabelRoot();
    for (const int64_t r : parent)
        if (r == -1) (void)r;  // root 已由 relabelRoot 处理

    // ---- 后序排列 perm[new] = old ----
    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (const int64_t id : order) {
        const TreeNode& nd = res.tree.node(id);
        res.perm.insert(res.perm.end(), nd.indices.begin(), nd.indices.end());
    }
    if (!res.tree.validate(n))
        throw std::runtime_error("external_tree: 树校验失败(validate)");
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        int64_t maxSize = 0, sumLeaf = 0, nLeaf = 0, maxDepth = 0;
        for (int64_t id = 0; id < N; ++id) {
            const TreeNode& nd = res.tree.node(id);
            maxSize = std::max(maxSize,
                               static_cast<int64_t>(nd.indices.size()));
            maxDepth = std::max(
                maxDepth, static_cast<int64_t>(res.tree.ancestors(id).size()));
            if (nd.role == NodeRole::Leaf) {
                sumLeaf += static_cast<int64_t>(nd.indices.size());
                ++nLeaf;
            }
        }
        std::fprintf(stderr,
                     "[tree] ext nodes=%lld maxSize=%lld leafAvg=%lld nLeaf=%lld "
                     "maxDepth=%lld\n",
                     static_cast<long long>(N), static_cast<long long>(maxSize),
                     static_cast<long long>(nLeaf ? sumLeaf / nLeaf : 0),
                     static_cast<long long>(nLeaf),
                     static_cast<long long>(maxDepth));
    }
    return res;
}

template class ExternalTreePartitioner<DefaultTraits>;

}  // namespace amls