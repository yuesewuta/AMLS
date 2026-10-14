// IO 与闭式谱用例生成器单元测试
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "linalg/dense_eigensolver.hpp"

using namespace amls;
using SM = DefaultTraits::SparseMatrix;

namespace {

class MatrixMarketFile {
public:
    explicit MatrixMarketFile(const std::string& text = "") {
        static unsigned serial = 0;
        path = "test_mm_parser_" + std::to_string(++serial) + ".tmp.mtx";
        std::ofstream out(path, std::ios::binary);
        out << text;
        REQUIRE(out.good());
    }
    ~MatrixMarketFile() { std::remove(path.c_str()); }
    MatrixMarketFile(const MatrixMarketFile&) = delete;
    MatrixMarketFile& operator=(const MatrixMarketFile&) = delete;

    std::string path;
};

} // namespace

TEST_CASE("matrix_market 数值格式与 general 重复累加", "[io][parser]") {
    MatrixMarketFile file(
        "%%MatrixMarket matrix coordinate real general\r\n"
        "  % comment\r\n\t\r\n"
        " +2\t+3 +6 \r\n"
        " +1 +2 +1.25e+2\r\n"
        " 1 2 -2.5E1\r\n"
        "\t% interleaved comment\r\n"
        " 2 3 +.5e-1\r\n"
        " 2 1 -0.0\r\n"
        " 1 1 1.\r\n"
        " 2 2 6.25E-2"); // final line without newline
    const SM A = readMatrixMarket<DefaultTraits>(file.path);
    REQUIRE(A.rows() == 2);
    REQUIRE(A.cols() == 3);
    REQUIRE(A.nonZeros() == 5);
    REQUIRE(A.coeff(0, 1) == 100.0);
    REQUIRE(A.coeff(1, 2) == Catch::Approx(0.05));
    REQUIRE(A.coeff(1, 0) == 0.0);
    REQUIRE(std::signbit(A.coeff(1, 0)));
    REQUIRE(A.coeff(0, 0) == 1.0);
    REQUIRE(A.coeff(1, 1) == 0.0625);
    REQUIRE(A.coeff(0, 2) == 0.0);
}

TEST_CASE("matrix_market symmetric 展开与重复累加", "[io][parser]") {
    MatrixMarketFile file(
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "3 3 5\n"
        "1 1 2\n"
        "1 1 3\n"
        "2 1 4\n"
        "2 1 -1\n"
        "2 3 5\n"); // either triangle expands; diagonal does not double
    const SM A = readMatrixMarket<DefaultTraits>(file.path);
    REQUIRE(A.nonZeros() == 5);
    REQUIRE(A.coeff(0, 0) == 5.0);
    REQUIRE(A.coeff(0, 1) == 3.0);
    REQUIRE(A.coeff(1, 0) == 3.0);
    REQUIRE(A.coeff(1, 2) == 5.0);
    REQUIRE(A.coeff(2, 1) == 5.0);
}

TEST_CASE("matrix_market pattern integer double 兼容", "[io][parser]") {
    SECTION("pattern symmetric duplicates") {
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate pattern symmetric\n"
            "2 2 3\n+1 +1\n2 1\n2 1\n");
        const SM A = readMatrixMarket<DefaultTraits>(file.path);
        REQUIRE(A.nonZeros() == 3);
        REQUIRE(A.coeff(0, 0) == 1.0);
        REQUIRE(A.coeff(0, 1) == 2.0);
        REQUIRE(A.coeff(1, 0) == 2.0);
    }
    SECTION("integer retains double parsing, including scientific notation") {
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate integer general\n"
            "2 2 3\n1 1 +42\n1 1 -2\n2 2 +2.5E1\n");
        const SM A = readMatrixMarket<DefaultTraits>(file.path);
        REQUIRE(A.coeff(0, 0) == 40.0);
        REQUIRE(A.coeff(1, 1) == 25.0);
    }
    SECTION("double alias and finite extremes") {
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate double general\n"
            "2 2 2\n1 1 1.7976931348623157e308\n"
            "2 2 4.9406564584124654e-324\n");
        const SM A = readMatrixMarket<DefaultTraits>(file.path);
        REQUIRE(A.coeff(0, 0) == (std::numeric_limits<double>::max)());
        REQUIRE(A.coeff(1, 1) == std::numeric_limits<double>::denorm_min());
    }
}

TEST_CASE("matrix_market 空矩阵和零记录", "[io][parser]") {
    for (const auto& dimensions : {"0 0 0", "0 3 0", "2 0 0", "2 3 0"}) {
        CAPTURE(dimensions);
        MatrixMarketFile file(
            std::string("%%MatrixMarket matrix coordinate real general\n") +
            dimensions + "\n \t\n % trailing comment\n");
        const SM A = readMatrixMarket<DefaultTraits>(file.path);
        REQUIRE(A.nonZeros() == 0);
    }
}

TEST_CASE("matrix_market 拒绝非法尺寸与 banner", "[io][parser]") {
    const std::string banner = "%%MatrixMarket matrix coordinate real general\n";
    const std::vector<std::string> invalid = {
        "", "%%MatrixMarket matrix coordinate real\n1 1 0\n",
        "%%MatrixMarket matrix coordinate real general extra\n1 1 0\n",
        "%%MatrixMarket vector coordinate real general\n1 1 0\n",
        "%%MatrixMarket matrix array real general\n1 1 0\n",
        "%%MatrixMarket matrix coordinate complex general\n1 1 0\n",
        "%%MatrixMarket matrix coordinate real skew-symmetric\n1 1 0\n",
        banner, banner + "% comment only\n", banner + "1\n",
        banner + "1 1\n", banner + "-1 1 0\n", banner + "1 -1 0\n",
        banner + "1 1 -1\n", banner + "1 1 0 extra\n",
        banner + "1x 1 0\n", banner + "1 1.0 0\n",
        banner + "1 1 1e0\n", banner + "1 1 +-1\n",
        banner + "1 1 +\n", banner + "1 1 ++1\n",
        banner + "9223372036854775808 1 0\n",
        banner + "9223372036854775807 1 0\n",
        banner + "1 9223372036854775807 0\n",
        banner + "1 1 9223372036854775807\n",
        banner + "0 1 1\n", banner + "1 0 1\n",
        "%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n",
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "1 1 4611686018427387904\n"
    };
    for (const auto& text : invalid) {
        CAPTURE(text);
        MatrixMarketFile file(text);
        REQUIRE_THROWS_AS(readMatrixMarket<DefaultTraits>(file.path), std::runtime_error);
    }
}

TEST_CASE("matrix_market 拒绝非法记录与索引", "[io][parser]") {
    const std::vector<std::string> invalid = {
        "", "1\n", "1 1\n", "1 1 +\n", "1 1 ++1\n", "1 1 +-1\n",
        "1 1 + 1\n", "1 1 junk\n", "1 1 1x\n", "1 1 1e\n",
        "1 1 1e+\n", "1 1 1.2.3\n", "1 1 0x1p0\n",
        "1 1 1 2\n", "1 1 1 % inline\n", "1 1 nan\n",
        "1 1 NaN(payload)\n", "1 1 inf\n", "1 1 -infinity\n",
        "1 1 +INF\n", "1 1 1e309\n", "1 1 1e-9999\n",
        "0 1 1\n", "1 0 1\n", "-1 1 1\n", "1 -1 1\n",
        "3 1 1\n", "1 3 1\n", "1.0 1 1\n", "1 1x 1\n",
        "1-1 1\n", "+-1 1 1\n", "1 ++1 1\n",
        "-9223372036854775808 1 1\n", "1 -9223372036854775808 1\n",
        "9223372036854775808 1 1\n", "1 9223372036854775808 1\n",
        "1 1 1\n2 2 2\n"
    };
    for (const auto& body : invalid) {
        CAPTURE(body);
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate real general\n2 2 1\n" + body);
        REQUIRE_THROWS_AS(readMatrixMarket<DefaultTraits>(file.path), std::runtime_error);
    }
    SECTION("too few entries despite comments") {
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate real general\n2 2 2\n"
            "1 1 1\n% only one entry\n\n");
        REQUIRE_THROWS_AS(readMatrixMarket<DefaultTraits>(file.path), std::runtime_error);
    }
    SECTION("entry with declared zero nnz") {
        MatrixMarketFile file(
            "%%MatrixMarket matrix coordinate real general\n2 2 0\n1 1 1\n");
        REQUIRE_THROWS_AS(readMatrixMarket<DefaultTraits>(file.path), std::runtime_error);
    }
    SECTION("pattern rejects values and out-of-bounds indices") {
        for (const auto& body : {"1 1 1\n", "1\n", "-1 1\n", "1 3\n"}) {
            CAPTURE(body);
            MatrixMarketFile file(
                std::string("%%MatrixMarket matrix coordinate pattern symmetric\n") +
                "2 2 1\n" + body);
            REQUIRE_THROWS_AS(readMatrixMarket<DefaultTraits>(file.path), std::runtime_error);
        }
    }
}

TEST_CASE("matrix_market 写出 header 按下三角计数", "[io][parser]") {
    const std::vector<Eigen::Triplet<double, int64_t>> trips = {
        {0, 0, 2}, {1, 0, -3}, {0, 1, -3}, {1, 1, 4},
        {2, 1, 0}, {1, 2, 0}, {2, 2, 5}
    };
    SM A(3, 3);
    A.setFromTriplets(trips.begin(), trips.end());
    MatrixMarketFile file;
    writeMatrixMarket<DefaultTraits>(file.path, A);
    std::ifstream in(file.path);
    std::string banner;
    std::getline(in, banner);
    int64_t rows = 0, cols = 0, nnz = 0;
    REQUIRE(static_cast<bool>(in >> rows >> cols >> nnz));
    REQUIRE(rows == 3);
    REQUIRE(cols == 3);
    REQUIRE(nnz == 5); // includes explicitly stored zero, excludes upper triangle
    int64_t records = 0, i = 0, j = 0;
    double v = 0;
    while (in >> i >> j >> v) {
        REQUIRE(i >= j);
        ++records;
    }
    REQUIRE(records == nnz);
    const SM roundtrip = readMatrixMarket<DefaultTraits>(file.path);
    REQUIRE(roundtrip.nonZeros() == A.nonZeros());
    REQUIRE((roundtrip - A).norm() == 0.0);
}

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
