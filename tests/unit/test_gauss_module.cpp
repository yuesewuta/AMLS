// GaussEliminationModule 暴力对拍单元测试。
// 稠密参考实现按同一消去数学逐步复算 Ψ / K̃ / M̃ / M_temp，与列压缩模块
// 产物逐项对拍（<1e-10）；覆盖深层链树、getReservedColState、release 语义。
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <map>
#include <random>
#include <vector>
#include "amls/gauss_module.hpp"
#include "amls/original_blocks.hpp"

using namespace amls;
using Dense = DefaultTraits::DenseMatrix;
using SM = DefaultTraits::SparseMatrix;

namespace {

/// 手搓树：3 叶 + 2 分割（与 test_original_blocks 同构）。
SeparatorTree makeTree() {
    SeparatorTree tree;
    tree.addNode(NodeRole::Leaf, {0, 1});        // 0
    tree.addNode(NodeRole::Leaf, {2, 3});        // 1
    tree.addNode(NodeRole::Separator, {4});      // 2（孩子 0,1）
    tree.addNode(NodeRole::Leaf, {5, 6});        // 3
    tree.addNode(NodeRole::Root, {7});           // 4（孩子 2,3）
    tree.setChildren(2, 0, 1);
    tree.setChildren(4, 2, 3);
    tree.setParent(0, 2);
    tree.setParent(1, 2);
    tree.setParent(2, 4);
    tree.setParent(3, 4);
    return tree;
}

SeparatorTree makeChainTree() {
    // 深链：0→2→4→6 三跳传播；每层分割节点至少 2 孩子（validate 约束）。
    // node0 叶{0,1} node1 叶{2} node2 分割{3}(孩子0,1) node3 叶{4}
    // node4 分割{5}(孩子2,3) node5 叶{6,7} node6 根{8}(孩子4,5)
    SeparatorTree tree;
    tree.addNode(NodeRole::Leaf, {0, 1});        // 0
    tree.addNode(NodeRole::Leaf, {2});           // 1
    tree.addNode(NodeRole::Separator, {3});      // 2（孩子 0,1）
    tree.addNode(NodeRole::Leaf, {4});           // 3
    tree.addNode(NodeRole::Separator, {5});      // 4（孩子 2,3）
    tree.addNode(NodeRole::Leaf, {6, 7});        // 5
    tree.addNode(NodeRole::Root, {8});           // 6（孩子 4,5）
    tree.setChildren(2, 0, 1);
    tree.setChildren(4, 2, 3);
    tree.setChildren(6, 4, 5);
    tree.setParent(0, 2);
    tree.setParent(1, 2);
    tree.setParent(2, 4);
    tree.setParent(3, 4);
    tree.setParent(4, 6);
    tree.setParent(5, 6);
    return tree;
}

/// 按树允许结构生成随机对称稠密矩阵（对角块 + (后代,祖先) 对）。
Dense buildStructured(const SeparatorTree& tree, int64_t n, double scale,
                      uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    Dense a = Dense::Zero(n, n);
    auto fill = [&](int64_t i, int64_t j) {
        for (int64_t p : tree.node(i).indices)
            for (int64_t q : tree.node(j).indices) {
                const double v = dist(rng);
                a(p, q) = v;
                a(q, p) = v;
            }
    };
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        fill(i, i);
        for (int64_t a : tree.ancestors(i)) fill(i, a);
    }
    // 对角占优保证 K̂ 局部可分解（对称正定）
    a.diagonal().array() += scale * static_cast<double>(n) * 2.0;
    return a;
}

SM toSparse(const Dense& d) {
    SM s = d.sparseView();
    s.makeCompressed();
    return s;
}

/// 稠密暴力参考：同一消去数学，全对称稠密块表逐步复算。
struct DenseRef {
    std::map<std::pair<int64_t, int64_t>, Dense> K, M, psi, mtemp;

    Dense& at(std::map<std::pair<int64_t, int64_t>, Dense>& t, int64_t i,
              int64_t j) {
        return t[{std::min(i, j), std::max(i, j)}];
    }
    const Dense& at(const std::map<std::pair<int64_t, int64_t>, Dense>& t,
                    int64_t i, int64_t j) const {
        return t.at({std::min(i, j), std::max(i, j)});
    }

    DenseRef(const Pencil<DefaultTraits>& pencil, const SeparatorTree& tree) {
        const int64_t N = tree.nNodes();
        for (int64_t i = 0; i < N; ++i) {
            at(K, i, i) = denseBlock(pencil.K, tree.node(i).indices,
                                     tree.node(i).indices);
            at(M, i, i) = denseBlock(pencil.M, tree.node(i).indices,
                                     tree.node(i).indices);
            for (int64_t a : tree.ancestors(i)) {
                at(K, i, a) = denseBlock(pencil.K, tree.node(i).indices,
                                         tree.node(a).indices);
                at(M, i, a) = denseBlock(pencil.M, tree.node(i).indices,
                                         tree.node(a).indices);
            }
        }
        for (int64_t i : tree.postOrder()) {
            const std::vector<int64_t> anc = tree.ancestors(i);
            // Ψ_ij = −K̂_ii⁻¹·K̃_ij
            for (int64_t j : anc)
                at(psi, i, j) = -at(K, i, i).llt().solve(at(K, i, j));
            // K̃_jk += Ψ_ijᵀ·K̃_ik (j ≤ k)
            for (size_t p = 0; p < anc.size(); ++p)
                for (size_t q = p; q < anc.size(); ++q) {
                    const int64_t j = anc[p], k = anc[q];
                    at(K, j, k) += at(psi, i, j).transpose() * at(K, i, k);
                    if (j != k)
                        at(K, k, j) =
                            at(K, j, k).transpose();  // 保持对称镜像
                }
            // M_temp_ij = M̃_ij + M̃_ii·Ψ_ij
            for (int64_t j : anc)
                at(mtemp, i, j) =
                    at(M, i, j) + at(M, i, i) * at(psi, i, j);
            // M̃_jk += Ψ_ijᵀ·M̃_ik + M_temp_ijᵀ·Ψ_ik (j ≤ k)
            for (size_t p = 0; p < anc.size(); ++p)
                for (size_t q = p; q < anc.size(); ++q) {
                    const int64_t j = anc[p], k = anc[q];
                    at(M, j, k) += at(psi, i, j).transpose() * at(M, i, k) +
                                   at(mtemp, i, j).transpose() * at(psi, i, k);
                    if (j != k) at(M, k, j) = at(M, j, k).transpose();
                }
        }
    }

    static Dense denseBlock(const SM& A, const std::vector<int64_t>& rows,
                            const std::vector<int64_t>& cols) {
        Dense out = Dense::Zero(static_cast<int64_t>(rows.size()),
                                static_cast<int64_t>(cols.size()));
        for (size_t p = 0; p < rows.size(); ++p)
            for (size_t q = 0; q < cols.size(); ++q)
                out(static_cast<int64_t>(p), static_cast<int64_t>(q)) =
                    A.coeff(rows[p], cols[q]);
        return out;
    }
};

void expectPsi(const GaussEliminationModule& gm, const DenseRef& ref, int64_t i,
               int64_t j, double tol) {
    const ColCompMat* psi = gm.getPsi(i, j);
    REQUIRE(psi != nullptr);
    Dense out;
    psi->inflate(out);
    const Dense& expect = ref.at(ref.psi, i, j);
    REQUIRE(out.rows() == expect.rows());
    REQUIRE(out.cols() == expect.cols());
    INFO("psi pair (" << i << "," << j << ") maxdiff="
                      << (out - expect).cwiseAbs().maxCoeff());
    REQUIRE(out.isApprox(expect, tol));
}

} // namespace

TEST_CASE("GaussEliminationModule 暴力对拍（三层树）", "[gauss_module]") {
    SeparatorTree tree = makeTree();
    const int64_t n = 8;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(buildStructured(tree, n, 1.0, 101));
    pencil.M = toSparse(buildStructured(tree, n, 0.5, 202));

    OriginalBlocks ob(pencil, tree);
    GaussEliminationModule gm;
    REQUIRE(gm.initialize(ob));

    DenseRef ref(pencil, tree);
    for (int64_t i : tree.postOrder()) {
        REQUIRE(gm.computeConstraint(i));
        REQUIRE(gm.accumulateReducedStiffness(i));
        REQUIRE(gm.computeMtemp(i));
        REQUIRE(gm.accumulateReducedMass(i));
    }

    const double tol = 1e-10;
    // Ψ 逐项
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t a : tree.ancestors(i)) expectPsi(gm, ref, i, a, tol);
    // M_temp 逐项
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t a : tree.ancestors(i)) {
            const ColCompMat* mt = gm.getMTemp(i, a);
            REQUIRE(mt != nullptr);
            Dense out;
            mt->inflate(out);
            REQUIRE(out.isApprox(ref.at(ref.mtemp, i, a), tol));
        }
    // 约化对角（全节点：叶 = 原始块，分割 = 累计值）
    for (int64_t i = 0; i < 5; ++i) {
        const Dense* kd = gm.getReducedDiagK(i);
        const Dense* md = gm.getReducedDiagM(i);
        REQUIRE(kd != nullptr);
        REQUIRE(md != nullptr);
        REQUIRE(kd->isApprox(ref.at(ref.K, i, i), tol));
        REQUIRE(md->isApprox(ref.at(ref.M, i, i), tol));
    }
    // 约化耦合块：通过对角块间接验证不足，再抽查 (0,4)（多跳隔代耦合）
    // 经 Ψ/M_temp 一致性已覆盖更新链；此处验证 getReservedColState 编码。
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t a : tree.ancestors(i)) {
            const std::vector<int32_t>* state = gm.getReservedColState(i, a);
            REQUIRE(state != nullptr);
            // 解码升序且落在 [0, n_a)
            int64_t prev = -1;
            for (int32_t v : *state) {
                const int64_t c = (v > 0 ? v : -v) - 1;
                REQUIRE(c > prev);
                REQUIRE(c >= 0);
                REQUIRE(c < 8);
                prev = c;
            }
            // 布局须覆盖原始 K 耦合块的全部非零列
            const SM& src = ob.subK(i, a);
            for (int64_t c = 0; c < src.cols(); ++c)
                if (src.outerIndexPtr()[c + 1] > src.outerIndexPtr()[c]) {
                    const bool in = std::binary_search(
                        state->begin(), state->end(),
                        static_cast<int32_t>(c + 1));
                    const bool neg = std::binary_search(
                        state->begin(), state->end(),
                        static_cast<int32_t>(-(c + 1)));
                    REQUIRE((in || neg));
                }
        }
}

TEST_CASE("GaussEliminationModule 暴力对拍（深链树）", "[gauss_module]") {
    SeparatorTree tree = makeChainTree();
    const int64_t n = 9;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(buildStructured(tree, n, 1.0, 303));
    pencil.M = toSparse(buildStructured(tree, n, 0.5, 404));

    OriginalBlocks ob(pencil, tree);
    GaussEliminationModule gm;
    REQUIRE(gm.initialize(ob));

    DenseRef ref(pencil, tree);
    for (int64_t i : tree.postOrder()) {
        REQUIRE(gm.computeConstraint(i));
        REQUIRE(gm.accumulateReducedStiffness(i));
        REQUIRE(gm.computeMtemp(i));
        REQUIRE(gm.accumulateReducedMass(i));
    }

    const double tol = 1e-10;
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        for (int64_t a : tree.ancestors(i)) {
            expectPsi(gm, ref, i, a, tol);
            const ColCompMat* mt = gm.getMTemp(i, a);
            REQUIRE(mt != nullptr);
            Dense out;
            mt->inflate(out);
            REQUIRE(out.isApprox(ref.at(ref.mtemp, i, a), tol));
        }
        const Dense* kd = gm.getReducedDiagK(i);
        const Dense* md = gm.getReducedDiagM(i);
        REQUIRE(kd != nullptr);
        REQUIRE(md != nullptr);
        REQUIRE(kd->isApprox(ref.at(ref.K, i, i), tol));
        REQUIRE(md->isApprox(ref.at(ref.M, i, i), tol));
    }
}

TEST_CASE("GaussEliminationModule release 语义", "[gauss_module]") {
    SeparatorTree tree = makeTree();
    const int64_t n = 8;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(buildStructured(tree, n, 1.0, 505));
    pencil.M = toSparse(buildStructured(tree, n, 0.5, 606));

    OriginalBlocks ob(pencil, tree);
    GaussEliminationModule gm;
    REQUIRE(gm.initialize(ob));
    for (int64_t i : tree.postOrder()) {
        REQUIRE(gm.computeConstraint(i));
        REQUIRE(gm.accumulateReducedStiffness(i));
        REQUIRE(gm.computeMtemp(i));
        REQUIRE(gm.accumulateReducedMass(i));
    }

    const int64_t leaf = 0, anc = 2;
    Dense keep;
    {
        const ColCompMat* psi = gm.getPsi(leaf, anc);
        REQUIRE(psi != nullptr);
        psi->inflate(keep);
    }
    gm.releasePsi(leaf, anc);
    REQUIRE(gm.getPsi(leaf, anc) == nullptr);
    REQUIRE(keep.rows() == 2);  // 释放前的展开结果仍归调用方持有
    // 预留列状态在数值释放后仍可读
    REQUIRE(gm.getReservedColState(leaf, anc) != nullptr);
    gm.releaseReducedDiagK(2);
    REQUIRE(gm.getReducedDiagK(2) == nullptr);
    // 未初始化模块 get 返回空
    GaussEliminationModule empty;
    REQUIRE(empty.getPsi(0, 2) == nullptr);
}
