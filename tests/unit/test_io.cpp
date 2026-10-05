// IO 与闭式谱用例生成器单元测试
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstdio>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "linalg/dense_eigensolver.hpp"

using namespace amls;
using SM = DefaultTraits::SparseMatrix;

TEST_CASE("matrix_market 往返一致", "[io]") {
    auto pencil = gridGraph2d(5, 7);
    const std::string path = "test_mm_roundtrip.tmp.mtx";  // CWD 相对（ctest 在 build/ 下运行）
    writeMatrixMarket<DefaultTraits>(path, pencil.K);
    SM K2 = readMatrixMarket<DefaultTraits>(path);
    std::remove(path.c_str());
    REQUIRE(K2.rows() == pencil.K.rows());
    REQUIRE(K2.cols() == pencil.K.cols());
    SM d = SM(K2) - SM(pencil.K);
    REQUIRE(d.norm() < 1e-14);
}

TEST_CASE("pathGraph 尺寸与谱闭式验证", "[case]") {
    const int64_t n = 50;
    auto p = pathGraph(n);
    REQUIRE(p.K.rows() == n);
    REQUIRE(p.M.rows() == n);
    // M = I
    SM d = SM(p.M) - SM(SM(p.M.rows(), p.M.cols()));
    for (int64_t c = 0; c < p.M.cols(); ++c)
        for (SM::InnerIterator it(p.M, c); it; ++it)
            REQUIRE(it.row() == c);  // 仅对角
    auto w = pathGraphSpectrum(n);
    REQUIRE(w.size() == n);
    // 与 dsygvd 金标准比对
    LapackDenseEigensolver<DefaultTraits> lap;
    auto Kd = DefaultTraits::DenseMatrix(p.K);
    auto Md = DefaultTraits::DenseMatrix(p.M);
    DefaultTraits::Vector w2;
    DefaultTraits::DenseMatrix V;
    lap.solveAll(Kd, Md, w2, V);
    for (int64_t j = 0; j < n; ++j)
        REQUIRE(std::abs(w2[j] - w[j]) <= 1e-12 * std::max(1.0, std::abs(w[j])));
    // 闭式值抽查：λ_1 = 2−2cos(π/(n+1))
    const double l1 = 2.0 - 2.0 * std::cos(M_PI / (n + 1));
    REQUIRE(w[0] == Catch::Approx(l1).epsilon(1e-14));
}

TEST_CASE("pathGraph 一致质量矩阵闭式谱", "[case]") {
    const int64_t n = 40;
    auto w = pathGraphSpectrum(n, true);
    const double th = M_PI / (n + 1);
    const double l1 = 6.0 * (1.0 - std::cos(th)) / (2.0 + std::cos(th));
    REQUIRE(w[0] == Catch::Approx(l1).epsilon(1e-12));
}

TEST_CASE("gridGraph2d 闭式谱与稠密金标准一致", "[case]") {
    auto p = gridGraph2d(6, 6);
    const int64_t n = 36;
    REQUIRE(p.K.rows() == n);
    auto w = gridGraph2dSpectrum(6, 6);
    REQUIRE(w.size() == n);
    LapackDenseEigensolver<DefaultTraits> lap;
    auto Kd = DefaultTraits::DenseMatrix(p.K);
    auto Md = DefaultTraits::DenseMatrix(p.M);
    DefaultTraits::Vector w2;
    DefaultTraits::DenseMatrix V;
    lap.solveAll(Kd, Md, w2, V);
    for (int64_t j = 0; j < n; ++j)
        REQUIRE(std::abs(w2[j] - w[j]) <= 1e-12 * std::max(1.0, std::abs(w[j])));
}
