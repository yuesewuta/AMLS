// AMLS 核心（截断策略 + 缩聚结构性质 + 全模态精确性 + 多跳 μ̃ 传播 +
// 统一 DAG 端到端）单元测试
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

/// 装配降阶 pencil（M̃：对角 I + 树块；K̃：保留特征值对角）并稠密求解。
static Vec solveReducedSpectrum(const SubspaceCondensation::Result& r,
                                const SeparatorTree& tree) {
    DM reducedMass = DM::Identity(r.reducedSize, r.reducedSize);
    for (const auto& b : r.reducedMassBlocks) {
        const int64_t r0 = r.reducedOffset[b.rowNode],
                      c0 = r.reducedOffset[b.colNode];
        reducedMass.block(r0, c0, b.block.rows(), b.block.cols()) = b.block;
        reducedMass.block(c0, r0, b.block.cols(), b.block.rows()) =
            b.block.transpose();
    }
    DM reducedK = DM::Zero(r.reducedSize, r.reducedSize);
    for (int64_t i = 0; i < tree.nNodes(); ++i)
        for (int64_t j = 0; j < r.keptModes[i]; ++j)
            reducedK(r.reducedOffset[i] + j, r.reducedOffset[i] + j) =
                r.eigenvalues[i][j];
    LapackDenseEigensolver<DefaultTraits> lap;
    DM Kc = reducedK, Mc = reducedMass, projectedVectors;
    Vec projectedEigenvalues;
    lap.solveAll(Kc, Mc, projectedEigenvalues, projectedVectors);
    return projectedEigenvalues;
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

    // P3: K̃ 只存对角（eigenvalues）,长度=k_i,升序,等于局部特征值
    const auto order = pres.tree.postOrder();
    for (int64_t id : order) {
        const int64_t k = r.keptModes[id];
        REQUIRE(r.eigenvalues[id].size() == k);
        for (int64_t j = 1; j < k; ++j)
            REQUIRE(r.eigenvalues[id][j] >= r.eigenvalues[id][j - 1]);
        REQUIRE(r.eigenvectors[id].rows() ==
                static_cast<int64_t>(pres.tree.node(id).indices.size()));
        REQUIRE(r.eigenvectors[id].cols() == k);
    }
    // P4: M̃ 非对角块的行节点是列节点的祖先
    for (const auto& b : r.reducedMassBlocks) {
        auto anc = pres.tree.ancestors(b.colNode);
        REQUIRE(std::find(anc.begin(), anc.end(), b.rowNode) != anc.end());
    }
    // P8: 全保留 ⇒ M̃ 对角块 = I（装配后检查）
    DM reducedMass = DM::Identity(r.reducedSize, r.reducedSize);
    for (const auto& b : r.reducedMassBlocks) {
        const int64_t r0 = r.reducedOffset[b.rowNode],
                      c0 = r.reducedOffset[b.colNode];
        reducedMass.block(r0, c0, b.block.rows(), b.block.cols()) = b.block;
        reducedMass.block(c0, r0, b.block.cols(), b.block.rows()) =
            b.block.transpose();
    }
    for (int64_t id : order) {
        const int64_t o = r.reducedOffset[id], k = r.keptModes[id];
        REQUIRE((reducedMass.block(o, o, k, k) - DM::Identity(k, k)).norm() <
                1e-10);
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
    REQUIRE(r.reducedSize == pair.K.rows());   // 全保留 ⇒ p = n

    Vec projectedEigenvalues = solveReducedSpectrum(r, pres.tree);
    auto ref = gridGraph2dSpectrum(10, 10);
    for (int64_t j = 0; j < 10; ++j)
        REQUIRE(std::abs(projectedEigenvalues[j] - ref[j]) <=
                1e-10 * std::max(1.0, std::abs(ref[j])));

    // 回代残差（前 3 阶）：投影坐标 → 重排坐标（纯 GEMM 回代）
    DM reducedMass = DM::Identity(r.reducedSize, r.reducedSize);
    for (const auto& b : r.reducedMassBlocks) {
        const int64_t r0 = r.reducedOffset[b.rowNode],
                      c0 = r.reducedOffset[b.colNode];
        reducedMass.block(r0, c0, b.block.rows(), b.block.cols()) = b.block;
        reducedMass.block(c0, r0, b.block.cols(), b.block.rows()) =
            b.block.transpose();
    }
    DM reducedK = DM::Zero(r.reducedSize, r.reducedSize);
    for (int64_t i = 0; i < pres.tree.nNodes(); ++i)
        for (int64_t j = 0; j < r.keptModes[i]; ++j)
            reducedK(r.reducedOffset[i] + j, r.reducedOffset[i] + j) =
                r.eigenvalues[i][j];
    LapackDenseEigensolver<DefaultTraits> lap;
    DM Kc = reducedK, Mc = reducedMass, projectedVectors;
    Vec wAll;
    lap.solveAll(Kc, Mc, wAll, projectedVectors);
    DM recovered = cond.backtransform(r, projectedVectors.leftCols(3));
    for (int64_t j = 0; j < 3; ++j) {
        DM rz = DM(reord.K) * recovered.col(j) -
                wAll[j] * (DM(reord.M) * recovered.col(j));
        REQUIRE(rz.norm() < 1e-10 * (DM(reord.K) * recovered.col(j)).norm());
    }
}

TEST_CASE("缩聚全模态:三层树多跳 μ̃ 传播", "[condensation]") {
    // 3 层树 ⇒ 存在 c→i→a 两跳以上的传播链（μ̃(c,a) 需经中间节点 fold
    // 逐段累加定稿）；全保留下投影谱仍须精确再现闭式谱。
    auto pair = gridGraph2d(12, 12);
    MetisPartitioner<DefaultTraits> part;
    auto pres = part.recursiveBisection(pair.K + pair.M, 3, 20);
    REQUIRE(pres.tree.nNodes() >= 8);   // 确实为 ≥3 层
    auto reord = reorder(pair, pres.perm);
    GaussianElimination ge(reord, pres.tree);
    ge.run();
    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    SubspaceCondensation cond(ge, eig);
    FixedCountTruncation<DefaultTraits> tr(0);
    auto r = cond.run(tr, 1);
    REQUIRE(r.reducedSize == pair.K.rows());

    Vec projectedEigenvalues = solveReducedSpectrum(r, pres.tree);
    auto ref = gridGraph2dSpectrum(12, 12);
    for (int64_t j = 0; j < 12; ++j)
        REQUIRE(std::abs(projectedEigenvalues[j] - ref[j]) <=
                1e-10 * std::max(1.0, std::abs(ref[j])));
}

TEST_CASE("求解器统一DAG:全保留 ⇒ 精确谱与回代残差", "[solver]") {
    // numThreads=4 走统一 DAG 路径（GE N1–N5 + 缩聚 N7–N10 同一 runner）；
    // fixedKeep 超过任意节点规模 ⇒ 全保留 ⇒ 谱精确、回代残差机器精度。
    auto pair = gridGraph2d(10, 10);
    AmlsOptions opt;
    opt.nLevels = 3;
    opt.leafSize = 20;
    opt.fixedKeep = 100;
    opt.nev = 10;
    opt.numThreads = 4;
    opt.verbosity = 0;
    opt.recoverVectors = true;
    AmlsSolver solver;
    auto sol = solver.solve({pair.K, pair.M}, opt);
    auto ref = gridGraph2dSpectrum(10, 10);
    for (int64_t j = 0; j < 10; ++j)
        REQUIRE(std::abs(sol.eigenvalues[j] - ref[j]) <=
                1e-8 * std::max(1.0, std::abs(ref[j])));
    for (int64_t j = 0; j < 3; ++j) {
        DM rz = DM(pair.K) * sol.eigenvectors.col(j) -
                sol.eigenvalues[j] * (DM(pair.M) * sol.eigenvectors.col(j));
        REQUIRE(rz.norm() <
                1e-8 * (DM(pair.K) * sol.eigenvectors.col(j)).norm());
    }
}
