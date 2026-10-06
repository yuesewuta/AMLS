// 逐块对拍：u / K̄ / M̄ / Mtemp / M̃ 树块与暴力公式逐项比较（两层树，
// 被消去节点全为叶子 ⇒ 暴力公式可直接用原始块）。
#include <cmath>
#include <iostream>
#include <map>

#include <catch2/catch_test_macros.hpp>

#include "amls/eigenpair_solver.hpp"
#include "amls/gaussian_elimination.hpp"
#include "amls/separator_tree.hpp"
#include "amls/spectral_truncation.hpp"
#include "amls/subspace_condensation.hpp"
#include "io/case_generator.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "partition/partitioner.hpp"

using namespace amls;
using DM = DefaultTraits::DenseMatrix;
using SM = DefaultTraits::SparseMatrix;

static Pencil<DefaultTraits> reorder2(const PencilPair& p,
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

/// 列压缩块展开为稠密（full 列）。
static DM expand(const amls::ColComp& blk) {
    DM out = DM::Zero(blk.rows(), blk.full);
    if (blk.ident)
        return blk.data;
    for (int64_t p = 0; p < static_cast<int64_t>(blk.cols.size()); ++p)
        out.col(blk.cols[static_cast<size_t>(p)]) = blk.data.col(p);
    return out;
}

TEST_CASE("对拍:u/K̄/M̄/Mtemp/M̃ 树块", "[bruteforce]") {
    auto pair = gridGraph2d(4, 4);
    MetisPartitioner<DefaultTraits> part;
    auto pres = part.recursiveBisection(pair.K + pair.M, 1, 6);
    const SeparatorTree& tree = pres.tree;
    auto reord = reorder2(pair, pres.perm);
    const DM K(reord.K), M(reord.M);
    GaussianElimination ge(reord, tree);
    ge.run();

    const int64_t root = tree.root();
    const int64_t oroot = ge.offset(root), nroot = ge.size(root);

    DM KbarBrute = K.block(oroot, oroot, nroot, nroot);
    DM MbarBrute = M.block(oroot, oroot, nroot, nroot);

    for (const int64_t i : tree.postOrder()) {
        const std::vector<int64_t> anc = tree.ancestors(i);
        if (anc.empty())
            continue;
        REQUIRE(ge.isLeaf(i));   // 两层树：被消去节点全为叶
        const int64_t oi = ge.offset(i), ni = ge.size(i);
        const DM Kii = K.block(oi, oi, ni, ni);
        const DM Mii = M.block(oi, oi, ni, ni);
        for (const int64_t a : anc) {
            const int64_t oa = ge.offset(a), na = ge.size(a);
            const DM Kia = K.block(oi, oa, ni, na);
            const DM Mia = M.block(oi, oa, ni, na);
            const DM uBrute = -Kii.ldlt().solve(Kia);
            // ---- u(i,a) ----
            const DM uGot = expand(ge.constraintModes(i, a));
            const double uErr = (uGot - uBrute).norm();
            std::cout << "[bf] u(" << i << "," << a << ") err=" << uErr
                      << "\n";
            REQUIRE(uErr < 1e-12);
            // ---- Mtemp(i,a) = M_{i,a} + M_ii·u ----
            const DM mtBrute = Mia + Mii * uBrute;
            const DM mtGot = expand(ge.temporaryMass(i, a));
            const double mtErr = (mtGot - mtBrute).norm();
            std::cout << "[bf] Mtemp(" << i << "," << a << ") err=" << mtErr
                      << "\n";
            REQUIRE(mtErr < 1e-12);
            // ---- 累加暴力 K̄/M̄ ----
            const DM Kai = K.block(oa, oi, na, ni);
            const DM Mai = M.block(oa, oi, na, ni);
            KbarBrute += Kai * uBrute;
            MbarBrute += Mai * uBrute + uBrute.transpose() * mtBrute;
        }
    }
    // ---- K̄(root) / M̄(root) ----
    const double kErr = (ge.kDiag(root).data - KbarBrute).norm();
    const double mErr = (ge.mDiag(root).data - MbarBrute).norm();
    std::cout << "[bf] K̄(root) err=" << kErr << "  M̄(root) err=" << mErr
              << "\n";
    REQUIRE(kErr < 1e-12);
    REQUIRE(mErr < 1e-12);

    // ---- M̃ 树块（全保留）：M̃_{r,c} = Z_rᵀ·(Mtemp(c,r))ᵀ·Z_c ----
    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    SubspaceCondensation cond(ge, eig);
    FixedCountTruncation<DefaultTraits> tr(0);
    auto r = cond.run(tr, 1);
    REQUIRE(r.reducedSize == K.rows());
    int64_t nBlocks = 0;
    for (const auto& b : r.reducedMassBlocks) {
        const int64_t a = b.rowNode, c = b.colNode;
        const DM mtT =
            expand(ge.temporaryMass(c, a)).transpose();   // (n_a × n_c)
        const DM brute = r.eigenvectors[static_cast<size_t>(a)].transpose() *
                         mtT * r.eigenvectors[static_cast<size_t>(c)];
        const double bErr = (b.block - brute).norm();
        std::cout << "[bf] M̃(" << a << "," << c << ") err=" << bErr << "\n";
        REQUIRE(bErr < 1e-10);
        ++nBlocks;
    }
    std::cout << "[bf] M̃ 树块数=" << nBlocks << "\n";
}

TEST_CASE("对拍:三层树中间分割点 + 多跳 μ̃", "[bruteforce]") {
    auto pair = gridGraph2d(6, 6);
    MetisPartitioner<DefaultTraits> part;
    auto pres = part.recursiveBisection(pair.K + pair.M, 2, 5);
    const SeparatorTree& tree = pres.tree;
    REQUIRE(tree.nNodes() >= 7);   // 4 叶 + 2 中间分割点 + 根
    auto reord = reorder2(pair, pres.perm);
    const int64_t n = reord.K.rows();
    const DM K(reord.K), M(reord.M);
    GaussianElimination ge(reord, tree);
    ge.run();

    // ---- 稠密工作矩阵暴力消去（与算法同序：后序、全祖先对更新）----
    DM Kw = K, Mw = M;
    std::map<std::pair<int64_t, int64_t>, DM> uBrute, mtBrute;
    for (const int64_t i : tree.postOrder()) {
        if (i == tree.root())
            continue;
        const int64_t oi = ge.offset(i), ni = ge.size(i);
        const std::vector<int64_t> anc = tree.ancestors(i);
        for (const int64_t a : anc) {
            const int64_t oa = ge.offset(a), na = ge.size(a);
            const DM Kii = Kw.block(oi, oi, ni, ni);
            const DM u = -Kii.ldlt().solve(DM(Kw.block(oi, oa, ni, na)));
            uBrute[{i, a}] = u;
            const DM uGot = expand(ge.constraintModes(i, a));
            const double uErr = (uGot - u).norm();
            std::cout << "[bf3] u(" << i << "," << a << ") err=" << uErr
                      << "\n";
            REQUIRE(uErr < 1e-10);
            const DM mt = DM(Mw.block(oi, oa, ni, na)) +
                          DM(Mw.block(oi, oi, ni, ni)) * u;
            mtBrute[{i, a}] = mt;
            const DM mtGot = expand(ge.temporaryMass(i, a));
            const double mtErr = (mtGot - mt).norm();
            std::cout << "[bf3] Mtemp(" << i << "," << a << ") err=" << mtErr
                      << "\n";
            REQUIRE(mtErr < 1e-10);
        }
        for (const int64_t a : anc) {
            const int64_t oa = ge.offset(a), na = ge.size(a);
            for (const int64_t lo : anc) {
                const int64_t olo = ge.offset(lo), nlo = ge.size(lo);
                Kw.block(oa, olo, na, nlo) +=
                    DM(Kw.block(oa, oi, na, ni)) * uBrute[{i, lo}];
            }
        }
        for (const int64_t a : anc) {
            const int64_t oa = ge.offset(a), na = ge.size(a);
            for (const int64_t lo : anc) {
                const int64_t olo = ge.offset(lo), nlo = ge.size(lo);
                Mw.block(oa, olo, na, nlo) +=
                    DM(Mw.block(oa, oi, na, ni)) * uBrute[{i, lo}] +
                    uBrute[{i, a}].transpose() * mtBrute[{i, lo}];
            }
        }
    }
    // ---- 全分割点对角块 ----
    for (const int64_t i : tree.postOrder()) {
        if (!ge.hasDiag(i))
            continue;
        const int64_t oi = ge.offset(i), ni = ge.size(i);
        const double kErr =
            (ge.kDiag(i).data - DM(Kw.block(oi, oi, ni, ni))).norm();
        const double mErr =
            (ge.mDiag(i).data - DM(Mw.block(oi, oi, ni, ni))).norm();
        std::cout << "[bf3] diag(" << i << ") K err=" << kErr
                  << " M err=" << mErr << "\n";
        REQUIRE(kErr < 1e-10);
        REQUIRE(mErr < 1e-10);
    }

    // ---- 多跳 μ̃ 与 M̃ 树块 ----
    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    SubspaceCondensation cond(ge, eig);
    FixedCountTruncation<DefaultTraits> tr(0);
    auto r = cond.run(tr, 1);
    REQUIRE(r.reducedSize == n);
    std::map<std::pair<int64_t, int64_t>, DM> muBrute;   // (c,a)
    int64_t nBlocks = 0;
    for (const int64_t i : tree.postOrder()) {
        const std::vector<int64_t> anc = tree.ancestors(i);
        if (i != tree.root())
            for (const int64_t a : anc)
                muBrute[{i, a}] =
                    r.eigenvectors[static_cast<size_t>(i)].transpose() *
                    mtBrute[{i, a}];
        for (const int64_t c : tree.descendants(i)) {
            const DM block =
                (muBrute[{c, i}] *
                 r.eigenvectors[static_cast<size_t>(i)]).transpose();
            bool found = false;
            for (const auto& b : r.reducedMassBlocks) {
                if (b.rowNode == i && b.colNode == c) {
                    const double bErr = (b.block - block).norm();
                    std::cout << "[bf3] M̃(" << i << "," << c
                              << ") err=" << bErr << "\n";
                    REQUIRE(bErr < 1e-8);
                    found = true;
                    ++nBlocks;
                }
            }
            REQUIRE(found);
        }
        if (i != tree.root())
            for (const int64_t c : tree.descendants(i))
                for (const int64_t a : anc)
                    muBrute[{c, a}] += muBrute[{c, i}] * uBrute[{i, a}];
    }
    std::cout << "[bf3] M̃ 树块数=" << nBlocks << "\n";
}
