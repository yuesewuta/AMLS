// AMLS 核心（截断策略 + 缩聚结构性质 + 全模态精确性）单元测试
#include <cmath>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "amls/eigenpair_solver.hpp"
#include "amls/gaussian_elimination.hpp"
#include "amls/separator_tree.hpp"
#include "amls/solver.hpp"
#include "amls/spectral_truncation.hpp"
#include "amls/subspace_condensation.hpp"
#include "io/case_generator.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "partition/partitioner.hpp"

using namespace amls;
using DM = DefaultTraits::DenseMatrix;
using Vec = DefaultTraits::Vector;
using SM = DefaultTraits::SparseMatrix;

static Pencil<DefaultTraits> reorder(const PencilPair& p,
                                     const std::vector<int64_t>& perm) {
    const int64_t n = p.K.rows();
    std::vector<int64_t> inv(n);
    for (int64_t j = 0; j < n; ++j) inv[perm[j]] = j;
    Pencil<DefaultTraits> out;
    for (const SM* src : {&p.K, &p.M}) {
        std::vector<Eigen::Triplet<double, int64_t>> trips;
        for (int64_t c = 0; c < n; ++c)
            for (SM::InnerIterator it(*src, c); it; ++it)
                trips.emplace_back(inv[it.row()], inv[it.col()], it.value());
        SM B(n, n);
        B.setFromTriplets(trips.begin(), trips.end());
        if (src == &p.K) out.K = B; else out.M = B;
    }
    return out;
}

TEST_CASE("截断策略:固定数目与前 k 阶提示", "[truncation]") {
    Vec w(10);
    for (int i = 0; i < 10; ++i) w[i] = i + 1;
    FixedCountTruncation<DefaultTraits> tr(3);
    REQUIRE(tr.select(w, 10) == 3);
    REQUIRE(tr.select(w, 2) == 2);            // 不超过 maxKeep
    FixedCountTruncation<DefaultTraits> all(0);
    REQUIRE(all.select(w, 10) == 10);
    typename DefaultTraits::Index il, iu;
    REQUIRE(tr.indexHint(il, iu));
    REQUIRE(il == 1);
    REQUIRE(iu == 3);
    REQUIRE_FALSE(all.indexHint(il, iu));
}

TEST_CASE("截断策略:阈值与区间提示", "[truncation]") {
    Vec w(10);
    for (int i = 0; i < 10; ++i) w[i] = i + 1;
    ThresholdTruncation<DefaultTraits> tr(5.5);
    REQUIRE(tr.select(w, 10) == 5);
    double lo, hi;
    REQUIRE(tr.intervalHint(lo, hi));
    REQUIRE(hi == Catch::Approx(5.5));
    ThresholdTruncation<DefaultTraits> none(-1.0);
    REQUIRE(none.select(w, 10) == 0);
}

TEST_CASE("缩聚结构性质 P3/P4/P8", "[condensation]") {
    auto pair = gridGraph2d(8, 8);
    MetisPartitioner<DefaultTraits> part;
    auto pres = part.recursiveBisection(pair.K + pair.M, 1, 15);
    auto reord = reorder(pair, pres.perm);
    GaussianElimination ge(reord, pres.tree);
    ge.run();
    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    SubspaceCondensation cond(ge, eig);
    FixedCountTruncation<DefaultTraits> tr(0);   // 全保留
    auto r = cond.run(tr, 1);

    // P3: K̃ 只存对角（mu）,长度=k_i,升序,等于局部特征值
    const auto order = pres.tree.postOrder();
    for (int64_t id : order) {
        const int64_t k = r.k[id];
        REQUIRE(r.mu[id].size() == k);
        for (int64_t j = 1; j < k; ++j)
            REQUIRE(r.mu[id][j] >= r.mu[id][j - 1]);
        REQUIRE(r.Z[id].rows() ==
                static_cast<int64_t>(pres.tree.node(id).indices.size()));
        REQUIRE(r.Z[id].cols() == k);
    }
    // P4: M̃ 非对角块的行节点是列节点的祖先
    for (const auto& b : r.mprojBlocks) {
        auto anc = pres.tree.ancestors(b.colNode);
        REQUIRE(std::find(anc.begin(), anc.end(), b.rowNode) != anc.end());
    }
    // P8: 全保留 ⇒ M̃ 对角块 = I（装配后检查）
    DM Mp = DM::Identity(r.p, r.p);
    for (const auto& b : r.mprojBlocks) {
        const int64_t r0 = r.projOffset[b.rowNode], c0 = r.projOffset[b.colNode];
        Mp.block(r0, c0, b.block.rows(), b.block.cols()) = b.block;
        Mp.block(c0, r0, b.block.cols(), b.block.rows()) = b.block.transpose();
    }
    for (int64_t id : order) {
        const int64_t o = r.projOffset[id], k = r.k[id];
        REQUIRE((Mp.block(o, o, k, k) - DM::Identity(k, k)).norm() < 1e-10);
    }
}

TEST_CASE("缩聚全模态:投影谱 == 闭式谱", "[condensation]") {
    // 装配 + 投影求解在重排 pencil 上精确再现全谱（对拍金标准）
    auto pair = gridGraph2d(10, 10);
    MetisPartitioner<DefaultTraits> part;
    auto pres = part.recursiveBisection(pair.K + pair.M, 2, 20);
    auto reord = reorder(pair, pres.perm);
    GaussianElimination ge(reord, pres.tree);
    ge.run();
    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    SubspaceCondensation cond(ge, eig);
    FixedCountTruncation<DefaultTraits> tr(0);
    auto r = cond.run(tr, 1);
    REQUIRE(r.p == pair.K.rows());        // 全保留 ⇒ p = n

    DM Mp = DM::Identity(r.p, r.p);
    for (const auto& b : r.mprojBlocks) {
        const int64_t r0 = r.projOffset[b.rowNode], c0 = r.projOffset[b.colNode];
        Mp.block(r0, c0, b.block.rows(), b.block.cols()) = b.block;
        Mp.block(c0, r0, b.block.cols(), b.block.rows()) = b.block.transpose();
    }
    DM Kp = DM::Zero(r.p, r.p);
    for (int64_t i = 0; i < pres.tree.nNodes(); ++i)
        for (int64_t j = 0; j < r.k[i]; ++j)
            Kp(r.projOffset[i] + j, r.projOffset[i] + j) = r.mu[i][j];

    LapackDenseEigensolver<DefaultTraits> lap;
    DM Kc = Kp, Mc = Mp, V;
    Vec w;
    lap.solveAll(Kc, Mc, w, V);
    auto ref = gridGraph2dSpectrum(10, 10);
    for (int64_t j = 0; j < 10; ++j)
        REQUIRE(std::abs(w[j] - ref[j]) <= 1e-10 * std::max(1.0, std::abs(ref[j])));

    // 回代残差（前 3 阶）
    DM Q = V.leftCols(3);
    DM z = cond.backtransform(r, Q);
    for (int64_t j = 0; j < 3; ++j) {
        DM rz = DM(reord.K) * z.col(j) - w[j] * (DM(reord.M) * z.col(j));
        REQUIRE(rz.norm() < 1e-10 * (DM(reord.K) * z.col(j)).norm());
    }
}
