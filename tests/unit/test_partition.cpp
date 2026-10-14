// 划分与分割树单元测试（含 perm 约定回归：perm[new] = old，
// 重排后矩阵的节点块必须与树后序段一一对应——曾是端到端错误的根因）。
#include <algorithm>
#include <set>
#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "amls/separator_tree.hpp"
#include "io/case_generator.hpp"
#include "partition/partitioner.hpp"

using namespace amls;
using SM = DefaultTraits::SparseMatrix;

TEST_CASE("METIS 递归二分树校验", "[partition]") {
    auto p = gridGraph2d(20, 20);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.recursiveBisection(p.K + p.M, 3, 60);
    REQUIRE(res.tree.validate(p.K.rows()));
    const int64_t leaves = [&] {
        int64_t c = 0;
        for (int64_t i = 0; i < res.tree.nNodes(); ++i)
            if (res.tree.node(i).role == NodeRole::Leaf) ++c;
        return c;
    }();
    REQUIRE(leaves >= 2);          // 至少完成一次二分
    REQUIRE(leaves <= 8);          // 3 层 ⇒ ≤ 2³ 叶
}

TEST_CASE("perm 约定与节点块连续性", "[partition]") {
    auto p = gridGraph2d(10, 10);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.recursiveBisection(p.K + p.M, 2, 25);
    const int64_t n = p.K.rows();
    // perm 是 [0,n) 的排列
    std::vector<int64_t> s(res.perm);
    std::sort(s.begin(), s.end());
    for (int64_t i = 0; i < n; ++i) REQUIRE(s[static_cast<size_t>(i)] == i);
    // 后序拼接：perm 的连续段恰为各节点索引集（即重排矩阵中节点块连续）
    const auto order = res.tree.postOrder();
    int64_t off = 0;
    for (int64_t id : order) {
        const auto& idx = res.tree.node(id).indices;
        std::set<int64_t> seg(res.perm.begin() + off,
                              res.perm.begin() + off + idx.size());
        std::set<int64_t> expect(idx.begin(), idx.end());
        REQUIRE(seg == expect);
        off += static_cast<int64_t>(idx.size());
    }
}

TEST_CASE("permuteSymmetric 与 perm 约定自洽", "[partition]") {
    // 正确约定:perm[new] = old;重排 B[j,k] = A[perm[j],perm[k]],
    // 即 A 的项 (r,c)（旧坐标）落位到 (pos[r],pos[c]),pos = perm⁻¹。
    auto p = pathGraph(37);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.recursiveBisection(p.K + p.M, 2, 10);
    const int64_t n = p.K.rows();
    std::vector<int64_t> pos(n, -1);
    for (int64_t j = 0; j < n; ++j) pos[res.perm[j]] = j;
    std::vector<Eigen::Triplet<double, int64_t>> trips;
    for (int64_t c = 0; c < n; ++c)
        for (SM::InnerIterator it(p.K, c); it; ++it)
            trips.emplace_back(pos[it.row()], pos[it.col()], it.value());
    SM B(n, n);
    B.setFromTriplets(trips.begin(), trips.end());
    // 节点对角块必须等于原矩阵对应子块（旧坐标索引）
    const auto order = res.tree.postOrder();
    for (int64_t id : order) {
        const auto& idx = res.tree.node(id).indices;
        for (size_t a = 0; a < idx.size(); ++a)
            for (size_t b = 0; b < idx.size(); ++b) {
                const double va = B.coeff(pos[idx[a]], pos[idx[b]]);
                const double vb = p.K.coeff(idx[a], idx[b]);
                REQUIRE(std::abs(va - vb) < 1e-15);
            }
    }
}

TEST_CASE("separator_tree 遍历正确", "[tree]") {
    SeparatorTree t;
    const int64_t l = t.addNode(NodeRole::Leaf, {0, 1});
    const int64_t r = t.addNode(NodeRole::Leaf, {2, 3});
    const int64_t s = t.addNode(NodeRole::Separator, {4});
    t.setChildren(s, l, r);
    t.setParent(l, s);
    t.setParent(r, s);
    t.relabelRoot();
    REQUIRE(t.node(s).role == NodeRole::Root);
    auto po = t.postOrder();
    REQUIRE(po.size() == 3);
    REQUIRE(po[2] == s);                    // 根最后
    REQUIRE(std::count(po.begin(), po.end(), l) == 1);
    auto anc = t.ancestors(l);
    REQUIRE(anc.size() == 1);
    REQUIRE(anc[0] == s);
    auto desc = t.descendants(s);
    REQUIRE(desc.size() == 2);
    REQUIRE(t.validate(5));
}

TEST_CASE("NodeND 单次嵌套剖分：树校验 + 边-祖先性", "[partition][nd]") {
    auto p = gridGraph2d(20, 20);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.nestedDissection(p.K + p.M, 60);
    const int64_t n = p.K.rows();
    REQUIRE(res.tree.validate(n));
    // perm 是 [0,n) 的排列
    std::vector<int64_t> s(res.perm);
    std::sort(s.begin(), s.end());
    for (int64_t i = 0; i < n; ++i)
        REQUIRE(s[static_cast<size_t>(i)] == i);
    // 边-祖先性（AMLS 有效性条件）：任一边 (u,v) 两端所在节点
    // 在树上必须可比（一方是另一方祖先，或同节点）
    std::vector<int64_t> nodeOf(n, -1);
    for (int64_t id = 0; id < res.tree.nNodes(); ++id)
        for (const int64_t v : res.tree.node(id).indices)
            nodeOf[static_cast<size_t>(v)] = id;
    auto comparable = [&](int64_t a, int64_t b) {
        for (int64_t x = a; x != -1; x = res.tree.node(x).parent)
            if (x == b) return true;
        for (int64_t x = b; x != -1; x = res.tree.node(x).parent)
            if (x == a) return true;
        return false;
    };
    SM A = p.K + p.M;
    for (int64_t c = 0; c < n; ++c)
        for (SM::InnerIterator it(A, c); it; ++it)
            if (it.row() != c)
                REQUIRE(comparable(nodeOf[it.row()], nodeOf[c]));
    // 叶规模：分组阈值 60 ⇒ 网格图上叶应处于阈值同量级
    int64_t maxLeaf = 0;
    for (int64_t id = 0; id < res.tree.nNodes(); ++id)
        if (res.tree.node(id).role == NodeRole::Leaf)
            maxLeaf = std::max<int64_t>(
                maxLeaf, static_cast<int64_t>(res.tree.node(id).indices.size()));
    REQUIRE(maxLeaf >= 60);
    REQUIRE(maxLeaf <= 4 * 60);
}

TEST_CASE("NodeND 退化情形：整图小于阈值 → 单叶根", "[partition][nd]") {
    auto p = pathGraph(37);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.nestedDissection(p.K + p.M, 500);
    REQUIRE(res.tree.validate(p.K.rows()));
    REQUIRE(res.tree.nNodes() == 1);
    REQUIRE(res.tree.node(0).role == NodeRole::Leaf);
}

TEST_CASE("NodeND storage variants preserve tree and ancestor edges", "[partition][nd]") {
    auto p = gridGraph2d(9, 9);
    SM full = p.K + p.M;
    SM lower = full.triangularView<Eigen::Lower>();
    SM upper = full.triangularView<Eigen::Upper>();
    SM uncompressed(full.rows(), full.cols());
    uncompressed.reserve(Eigen::VectorXi::Constant(full.cols(), 12));
    for (int64_t c = 0; c < full.cols(); ++c)
        for (SM::InnerIterator it(full, c); it; ++it)
            uncompressed.insert(it.row(), c) = it.value();
    REQUIRE_FALSE(uncompressed.isCompressed());
    MetisPartitioner<DefaultTraits> part;
    auto reference = part.nestedDissection(full, 6);
    for (const SM* input : {&lower, &upper, &uncompressed}) {
        auto result = part.nestedDissection(*input, 6);
        REQUIRE(result.tree.validate(full.rows()));
        REQUIRE(result.perm == reference.perm);
        int64_t off = 0;
        for (int64_t id : result.tree.postOrder()) {
            for (int64_t v : result.tree.node(id).indices)
                REQUIRE(result.perm[static_cast<size_t>(off++)] == v);
        }
    }
}

TEST_CASE("NodeND disconnected graph and isolated vertices", "[partition][nd]") {
    auto grid = gridGraph2d(7, 7);
    const int64_t n = 2 * grid.K.rows() + 3;
    SM A(n, n);
    std::vector<Eigen::Triplet<double, int64_t>> trips;
    for (int64_t shift : {int64_t(0), grid.K.rows()})
        for (int64_t c = 0; c < grid.K.cols(); ++c)
            for (SM::InnerIterator it(grid.K, c); it; ++it)
                trips.emplace_back(shift + it.row(), shift + c, it.value());
    for (int64_t i = n - 3; i < n; ++i) trips.emplace_back(i, i, 1.0);
    A.setFromTriplets(trips.begin(), trips.end());
    MetisPartitioner<DefaultTraits> part;
    auto res = part.nestedDissection(A, 4);
    REQUIRE(res.tree.validate(n));
    std::vector<int64_t> nodeOf(n, -1);
    for (int64_t id = 0; id < res.tree.nNodes(); ++id)
        for (int64_t v : res.tree.node(id).indices) nodeOf[v] = id;
    for (int64_t c = 0; c < n; ++c)
        for (SM::InnerIterator it(A, c); it; ++it) {
            auto left = res.tree.ancestors(nodeOf[it.row()]);
            auto right = res.tree.ancestors(nodeOf[c]);
            REQUIRE((nodeOf[it.row()] == nodeOf[c] ||
                     std::find(left.begin(), left.end(), nodeOf[c]) != left.end() ||
                     std::find(right.begin(), right.end(), nodeOf[it.row()]) != right.end()));
        }
}

TEST_CASE("Separator tree rejects duplicate children and disconnected cycles", "[tree]") {
    SeparatorTree duplicate;
    auto leaf = duplicate.addNode(NodeRole::Leaf, {0});
    auto root = duplicate.addNode(NodeRole::Root, {1});
    duplicate.setChildren(root, leaf, leaf);
    duplicate.setParent(leaf, root);
    REQUIRE_FALSE(duplicate.validate(2));

    SeparatorTree disconnected;
    disconnected.addNode(NodeRole::Leaf, {0});
    auto a = disconnected.addNode(NodeRole::Separator, {1});
    auto b = disconnected.addNode(NodeRole::Separator, {2});
    auto al = disconnected.addNode(NodeRole::Leaf, {3});
    auto bl = disconnected.addNode(NodeRole::Leaf, {4});
    disconnected.setChildren(a, b, al); disconnected.setParent(b, a); disconnected.setParent(al, a);
    disconnected.setChildren(b, a, bl); disconnected.setParent(a, b); disconnected.setParent(bl, b);
    REQUIRE_FALSE(disconnected.validate(5));
}


TEST_CASE("NodeND 大网格回归：深消去树下树校验", "[partition][nd]") {
    auto p = gridGraph2d(80, 80);
    MetisPartitioner<DefaultTraits> part;
    auto res = part.nestedDissection(p.K + p.M, 40);
    const int64_t n = p.K.rows();
    REQUIRE(res.tree.validate(n));
    std::vector<int64_t> nodeOf(n, -1);
    for (int64_t id = 0; id < res.tree.nNodes(); ++id)
        for (const int64_t v : res.tree.node(id).indices)
            nodeOf[static_cast<size_t>(v)] = id;
    auto comparable = [&](int64_t a, int64_t b) {
        for (int64_t x = a; x != -1; x = res.tree.node(x).parent)
            if (x == b) return true;
        for (int64_t x = b; x != -1; x = res.tree.node(x).parent)
            if (x == a) return true;
        return false;
    };
    SM A = p.K + p.M;
    for (int64_t c = 0; c < n; ++c)
        for (SM::InnerIterator it(A, c); it; ++it)
            if (it.row() != c)
                REQUIRE(comparable(nodeOf[it.row()], nodeOf[c]));
}
