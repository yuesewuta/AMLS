// OriginalBlocks 单元测试：对照 Eigen 稠密切片逐块对拍。
// 覆盖：对角块、(后代,祖先) 耦合块（直接 + 镜像两种来源）、空耦合对、
// K/M 独立模式、祖先判定与不可比耦合拒绝。
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Dense>

#include <random>
#include <stdexcept>
#include <vector>

#include "amls/original_blocks.hpp"

using namespace amls;
using Dense = DefaultTraits::DenseMatrix;
using SM = DefaultTraits::SparseMatrix;

namespace {

/// 手搓三层树（后序编号）：
///   node0 叶 {0,1}   node1 叶 {2,3}
///   node2 分割 {4}  （孩子 0,1）
///   node3 叶 {5,6}
///   node4 根分割 {7}（孩子 2,3）
SeparatorTree makeTree() {
    SeparatorTree tree;
    tree.addNode(NodeRole::Leaf, {0, 1});
    tree.addNode(NodeRole::Leaf, {2, 3});
    tree.addNode(NodeRole::Separator, {4});
    tree.addNode(NodeRole::Leaf, {5, 6});
    tree.addNode(NodeRole::Root, {7});
    tree.setChildren(2, 0, 1);
    tree.setChildren(4, 2, 3);
    tree.setParent(0, 2);
    tree.setParent(1, 2);
    tree.setParent(2, 4);
    tree.setParent(3, 4);
    return tree;
}

Dense randomSym(int64_t n, double scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    Dense a = Dense::Zero(n, n);
    for (int64_t c = 0; c < n; ++c)
        for (int64_t r = c; r < n; ++r) {
            const double v = dist(rng);
            a(r, c) = v;
            a(c, r) = v;
        }
    a.diagonal().array() += scale * n;
    return a;
}

SM toSparse(const Dense& d) {
    SM s = d.sparseView();
    s.makeCompressed();
    return s;
}

/// 块对拍：抽取块 == 稠密切片（0.0 容差的逐项相等）。
void expectBlock(const SM& block, const Dense& full, int64_t node,
                 const std::vector<int64_t>& rows, const std::vector<int64_t>& cols) {
    Dense expect = Dense::Zero(rows.size(), cols.size());
    for (size_t p = 0; p < rows.size(); ++p)
        for (size_t q = 0; q < cols.size(); ++q)
            expect(static_cast<int64_t>(p), static_cast<int64_t>(q)) =
                full(rows[p], cols[q]);
    Dense got(block);
    INFO("node pair (" << node << "): shape " << got.rows() << "x" << got.cols());
    REQUIRE(got.rows() == expect.rows());
    REQUIRE(got.cols() == expect.cols());
    REQUIRE(got.isApprox(expect, 0.0));
}

} // namespace

TEST_CASE("OriginalBlocks 对角与耦合块对拍稠密切片", "[original_blocks]") {
    const int64_t n = 8;
    SeparatorTree tree = makeTree();
    // 按树允许的结构生成随机对称矩阵：对角块 + (后代,祖先) 对，无兄弟耦合。
    const std::vector<std::pair<int64_t, int64_t>> pairs = {
        {0, 2}, {1, 2}, {2, 4}, {3, 4}, {0, 4}, {1, 4}};
    const std::vector<std::vector<int64_t>> nodeIdx = {
        {0, 1}, {2, 3}, {4}, {5, 6}, {7}};
    auto buildMatrix = [&](double scale, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-scale, scale);
        Dense a = Dense::Zero(n, n);
        auto fill = [&](int64_t i, int64_t j) {
            const auto& ri = nodeIdx[static_cast<size_t>(i)];
            const auto& rj = nodeIdx[static_cast<size_t>(j)];
            for (int64_t p : ri)
                for (int64_t q : rj) {
                    const double v = dist(rng);
                    a(p, q) = v;
                    a(q, p) = v;
                }
        };
        for (int64_t i = 0; i < 5; ++i) fill(i, i);
        for (const auto& [i, j] : pairs) fill(i, j);
        return a;
    };
    Dense kd = buildMatrix(1.0, 11);
    Dense md = buildMatrix(0.5, 22);
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(kd);
    pencil.M = toSparse(md);

    OriginalBlocks ob(pencil, tree);
    REQUIRE(ob.nNodes() == 5);
    REQUIRE(ob.nCouplingPairs() == 6);  // (0,2)(1,2)(2,4)(3,4)(0,4)(1,4)

    // 对角块
    for (int64_t i = 0; i < 5; ++i) {
        const auto& idx = ob.indices(i);
        expectBlock(ob.diagK(i), kd, i, idx, idx);
        expectBlock(ob.diagM(i), md, i, idx, idx);
    }

    // 耦合块：直接项 + 镜像项（对称矩阵两侧都扫到）必须齐全
    for (const auto& [i, a] : pairs) {
        const auto& ri = ob.indices(i);
        const auto& ra = ob.indices(a);
        expectBlock(ob.subK(i, a), kd, i, ri, ra);
        expectBlock(ob.subM(i, a), md, i, ri, ra);
        // 镜像一致性：subK(i,a)ᵀ == K(a 空间 × i 空间)
        Dense trans(ob.subK(i, a).transpose());
        Dense mirror = Dense::Zero(ra.size(), ri.size());
        for (size_t p = 0; p < ra.size(); ++p)
            for (size_t q = 0; q < ri.size(); ++q)
                mirror(static_cast<int64_t>(p), static_cast<int64_t>(q)) =
                    kd(ra[p], ri[q]);
        REQUIRE(trans.isApprox(mirror, 0.0));
    }

    // 祖先判定：subK(0,1) 兄弟节点非法
    REQUIRE_THROWS_AS(ob.subK(0, 1), std::out_of_range);
    REQUIRE_THROWS_AS(ob.subK(0, 3), std::out_of_range);
}

TEST_CASE("OriginalBlocks 空耦合对与 K/M 独立模式", "[original_blocks]") {
    const int64_t n = 8;
    SeparatorTree tree = makeTree();
    // K：只有 (node0, node2) 耦合（dof 0-4），其余耦合为零
    Dense kd = Dense::Zero(n, n);
    kd(0, 4) = kd(4, 0) = 1.5;
    kd(1, 4) = kd(4, 1) = -0.5;
    kd.diagonal().setOnes();
    // M：额外含 (node1, node4) 耦合（dof 3-7），模式与 K 不同
    Dense md = kd;
    md(3, 7) = md(7, 3) = 0.25;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(kd);
    pencil.M = toSparse(md);

    OriginalBlocks ob(pencil, tree);
    // (0,2) 两路都有
    REQUIRE(ob.hasCouplingK(0, 2));
    REQUIRE(ob.hasCouplingM(0, 2));
    // (1,2)：K 空、M 空（md(3,4)=0）—— 布局仍存在但形状正确
    REQUIRE(!ob.hasCouplingK(1, 2));
    REQUIRE(!ob.hasCouplingM(1, 2));
    REQUIRE(ob.subK(1, 2).rows() == 2);
    REQUIRE(ob.subK(1, 2).cols() == 1);
    REQUIRE(ob.subK(1, 2).nonZeros() == 0);
    // (1,4)：M 有、K 无（模式独立）
    REQUIRE(!ob.hasCouplingK(1, 4));
    REQUIRE(ob.hasCouplingM(1, 4));
    Dense mblk(ob.subM(1, 4));
    REQUIRE(mblk(1, 0) == Catch::Approx(0.25));  // dof 3 在 {2,3} 中局部行 1
    // 对角块逐项
    Dense kdDiag(ob.diagK(1));
    REQUIRE(kdDiag(0, 0) == Catch::Approx(1.0));
    REQUIRE(kdDiag(0, 1) == Catch::Approx(0.0));
}

TEST_CASE("OriginalBlocks 不可比节点耦合拒绝", "[original_blocks]") {
    const int64_t n = 8;
    SeparatorTree tree = makeTree();
    Dense kd = Dense::Identity(n, n);
    kd(0, 5) = kd(5, 0) = 0.7;  // node0 与 node3 直接耦合 → 非法
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(kd);
    pencil.M = Dense::Identity(n, n).sparseView();

    REQUIRE_THROWS_AS(OriginalBlocks(pencil, tree), std::invalid_argument);
}

TEST_CASE("OriginalBlocks 结构非法输入拒绝", "[original_blocks]") {
    SeparatorTree tree = makeTree();
    // 索引不覆盖全空间
    SeparatorTree bad = tree;
    Pencil<DefaultTraits> pencil;
    pencil.K = SM(9, 9);
    pencil.M = SM(9, 9);
    REQUIRE_THROWS_AS(OriginalBlocks(pencil, bad), std::invalid_argument);
    // 形状不一致
    pencil.K = SM(8, 8);
    pencil.M = SM(8, 7);
    REQUIRE_THROWS_AS(OriginalBlocks(pencil, tree), std::invalid_argument);
}
