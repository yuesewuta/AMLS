// 稠密特征解与稀疏直接解单元测试
#include <cmath>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "io/case_generator.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "linalg/sparse_direct.hpp"

using namespace amls;
using DM = DefaultTraits::DenseMatrix;
using Vec = DefaultTraits::Vector;

TEST_CASE("dsygvd 全谱精度 < 1e-12", "[lapack]") {
    auto p = pathGraph(60);
    LapackDenseEigensolver<DefaultTraits> lap;
    DM K = DM(p.K), M = DM(p.M), V;
    Vec w;
    lap.solveAll(K, M, w, V);
    auto ref = pathGraphSpectrum(60);
    for (int64_t j = 0; j < 60; ++j)
        REQUIRE(std::abs(w[j] - ref[j]) <= 1e-12 * std::max(1.0, std::abs(ref[j])));
    // 特征向量 M-正交: Vᵀ M V = I
    DM g = V.transpose() * DM(p.M) * V;
    REQUIRE((g - DM::Identity(g.rows(), g.cols())).norm() < 1e-10);
}

TEST_CASE("dsygvx 前 k 阶与全谱一致", "[lapack]") {
    auto p = gridGraph2d(5, 5);
    LapackDenseEigensolver<DefaultTraits> lap;
    DM K = DM(p.K), M = DM(p.M), V;
    Vec w;
    lap.solveIndex(K, M, 1, 7, w, V);
    REQUIRE(w.size() == 7);
    auto ref = gridGraph2dSpectrum(5, 5);
    for (int64_t j = 0; j < 7; ++j)
        REQUIRE(std::abs(w[j] - ref[j]) <= 1e-12 * std::max(1.0, std::abs(ref[j])));
    REQUIRE(V.cols() == 7);
}

TEST_CASE("dsygvx 区间解与全谱一致", "[lapack]") {
    auto p = pathGraph(80);
    LapackDenseEigensolver<DefaultTraits> lap;
    DM K = DM(p.K), M = DM(p.M), V;
    Vec w;
    lap.solveInterval(K, M, -1.0, 0.5, w, V);
    auto ref = pathGraphSpectrum(80);
    int64_t cnt = 0;
    while (cnt < 80 && ref[cnt] <= 0.5) ++cnt;
    REQUIRE(w.size() == cnt);
    for (int64_t j = 0; j < cnt; ++j)
        REQUIRE(std::abs(w[j] - ref[j]) <= 1e-12 * std::max(1.0, std::abs(ref[j])));
}

TEST_CASE("稀疏直接解 SPD 多右端精确", "[sparse_direct]") {
    auto p = gridGraph2d(8, 8);
    EigenSimplicialLDLTSolver<DefaultTraits> solver;
    solver.factorize(p.K);
    const int64_t n = p.K.rows();
    DM B = DM::Random(n, 5);
    DM X;
    solver.solveWithFactor(B, X);
    DM R = DM(p.K) * X - B;
    REQUIRE(R.norm() < 1e-10 * B.norm());
}

TEST_CASE("稀疏直接解拒绝非正定矩阵", "[sparse_direct]") {
    // 不定矩阵（对角含负元）应抛异常（P10）
    auto p = pathGraph(10);
    DefaultTraits::SparseMatrix A = p.K;
    for (int c = 0; c < A.cols(); ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
            if (it.row() == c) it.valueRef() = -1.0;
    EigenSimplicialLDLTSolver<DefaultTraits> solver;
    REQUIRE_THROWS_AS(solver.factorize(A), std::runtime_error);
}

TEST_CASE("稠密直接解与稀疏解一致", "[sparse_direct]") {
    auto p = pathGraph(30);
    DenseLDLTSolver<DefaultTraits> dsolver;
    dsolver.factorizeDense(DM(p.K));
    EigenSimplicialLDLTSolver<DefaultTraits> ssolver;
    ssolver.factorize(p.K);
    DM B = DM::Random(30, 3), X1, X2;
    dsolver.solveWithFactor(B, X1);
    ssolver.solveWithFactor(B, X2);
    REQUIRE((X1 - X2).norm() < 1e-12 * X2.norm());
}
