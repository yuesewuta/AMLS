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
