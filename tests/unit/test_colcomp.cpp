// ColCompMat 生命周期/布局 与 col_ops 算子的单元测试。
// 验收基准：全部算子与稠密参考实现对拍；生命周期与预留列行为逐项断言。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include <random>
#include <vector>

#include "linalg/col_ops.hpp"
#include "linalg/colcomp_mat.hpp"

using namespace amls;
using Dense = DefaultTraits::DenseMatrix;
using Sparse = DefaultTraits::SparseMatrix;

namespace {

/// 随机稠密矩阵（固定种子）。
Dense randomDense(int64_t rows, int64_t cols, double scale = 1.0) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-scale, scale);
    Dense a(rows, cols);
    for (int64_t c = 0; c < cols; ++c)
        for (int64_t r = 0; r < rows; ++r) a(r, c) = dist(rng);
    return a;
}

/// 稀疏化：把稠密矩阵按给定列集 + 行内稀疏模式转稀疏（满对称由调用方保证）。
Sparse toSparse(const Dense& d) {
    Sparse s = d.sparseView();
    s.makeCompressed();
    return s;
}

/// 把 ColCompMat 展开成稠密并与参考逐项比较（预留列 = 0）。
void expectDenseEqual(const ColCompMat& m, const Dense& ref, double tol) {
    REQUIRE(m.numericReady());
    Dense out;
    m.inflate(out);
    REQUIRE(out.rows() == ref.rows());
    REQUIRE(out.cols() == ref.cols());
    REQUIRE(out.isApprox(ref, tol));
}

} // namespace

TEST_CASE("ColCompMat lifecycle and layout", "[colcomp_mat]") {
    ColCompMat m(5, 8);
    REQUIRE(!m.layoutReady());
    REQUIRE(m.setLayout({0, 3, 7}));
    REQUIRE(m.layoutReady());
    REQUIRE(!m.numericReady());
    REQUIRE(m.validCols() == 3);
    REQUIRE(m.storedCols() == std::vector<int64_t>{0, 3, 7});
    // 已布局后重入：同列集允许，异列集拒绝
    REQUIRE(m.setLayout({0, 3, 7}));
    REQUIRE(!m.setLayout({1, 2}));
    // 未数值化不能展开
    { Dense tmp; REQUIRE_THROWS_AS(m.inflate(tmp), std::logic_error); }
    // initializeZero 后可展开（全零）
    REQUIRE(m.initializeZero());
    REQUIRE(m.activeCols() == 0);
    expectDenseEqual(m, Dense::Zero(5, 8), 0.0);
    // 占位激活
    m.markColNonZero(3);
    REQUIRE(m.colActive(1));
    REQUIRE(m.activeCols() == 1);
    m.markColNonZero(3);  // 幂等
    REQUIRE(m.activeCols() == 1);
    m.markColNonZero(2);  // 不在布局内：忽略
    REQUIRE(m.activeCols() == 1);
    // 释放后布局保留、数值清空、禁止重建
    m.releaseNumeric();
    REQUIRE(m.numericReleased());
    REQUIRE(m.validCols() == 3);
    REQUIRE(m.dense().size() == 0);
    REQUIRE(!m.initializeZero());
    m.releaseNumeric();  // 幂等
    REQUIRE(m.numericReleased());
}

TEST_CASE("ColCompMat storageCol mapping", "[colcomp_mat]") {
    ColCompMat full(4, 4);
    full.setLayout({0, 1, 2, 3});
    REQUIRE(full.initializeZero());
    // 满列布局恒等 O(1)
    for (int64_t c = 0; c < 4; ++c) REQUIRE(full.storageCol(c) == c);
    REQUIRE(full.storageCol(9) == -1);

    ColCompMat part(4, 6);
    part.setLayout({1, 3, 5});
    REQUIRE(part.storageCol(1) == 0);
    REQUIRE(part.storageCol(3) == 1);
    REQUIRE(part.storageCol(5) == 2);
    REQUIRE(part.storageCol(0) == -1);
    REQUIRE(part.storageCol(4) == -1);
}

TEST_CASE("ColCompMat initializeFromSparse fills only laid-out columns",
          "[colcomp_mat]") {
    Dense ref = Dense::Zero(4, 5);
    ref(0, 1) = 2.0;
    ref(2, 1) = -1.0;
    ref(3, 3) = 0.5;
    ref(1, 4) = 7.0;  // 布局外列
    const Sparse src = toSparse(ref);

    ColCompMat m(4, 5);
    REQUIRE(m.setLayout({1, 3}));  // 刻意不含 4
    REQUIRE(m.initializeFromSparse(src));
    REQUIRE(m.activeCols() == 2);
    Dense expect = Dense::Zero(4, 5);
    expect(0, 1) = 2.0;
    expect(2, 1) = -1.0;
    expect(3, 3) = 0.5;
    expectDenseEqual(m, expect, 1e-15);
    // Released 后拒绝重建
    m.releaseNumeric();
    REQUIRE(!m.initializeFromSparse(src));
}

TEST_CASE("ColCompMat initializeConstraintFromRhs keeps active columns",
          "[colcomp_mat]") {
    Dense src = Dense::Zero(3, 6);
    src(0, 2) = 1.0;
    src(1, 2) = 2.0;
    src(2, 5) = 3.0;
    ColCompMat rhs(3, 6);
    rhs.setLayout({0, 2, 4, 5});
    rhs.initializeFromSparse(toSparse(src));
    // rhs 只有列 2、5 有非零 → active
    ColCompMat x;
    REQUIRE(x.initializeConstraintFromRhs(rhs));
    REQUIRE(x.totalCols() == 6);
    REQUIRE(x.rows() == 3);
    REQUIRE(x.storedCols() == std::vector<int64_t>{2, 5});
    expectDenseEqual(x, src, 1e-15);
}

TEST_CASE("ColCompMat initializeMtempFromInputs unions active columns",
          "[colcomp_mat]") {
    // mass：布局 {0,2,3}，active {0,3}；psi：布局 {1,2,5}，active {2,5}
    Dense md = Dense::Zero(4, 6);
    md(1, 0) = 1.5;
    md(2, 3) = -2.0;
    ColCompMat mass(4, 6);
    mass.setLayout({0, 2, 3});
    REQUIRE(mass.initializeFromSparse(toSparse(md)));

    Dense pd = Dense::Zero(4, 6);
    pd(0, 2) = 0.25;
    pd(3, 5) = 4.0;
    ColCompMat psi(4, 6);
    psi.setLayout({1, 2, 5});
    REQUIRE(psi.initializeFromSparse(toSparse(pd)));

    ColCompMat mt;
    REQUIRE(mt.initializeMtempFromInputs(mass, psi));
    // 布局 = 两侧有效列并集；数值仅取 mass 对应列，psi 侧为占位列
    // （后续 opsColAcc(M̃_ii, Ψ) 写入时才激活）
    REQUIRE(mt.storedCols() == std::vector<int64_t>{0, 2, 3, 5});
    REQUIRE(mt.activeCols() == 2);
    Dense expect = Dense::Zero(4, 6);
    expect.col(0) = md.col(0);
    expect.col(3) = md.col(3);
    expectDenseEqual(mt, expect, 1e-15);
}

TEST_CASE("ColCompMat initializeMuFromInputs merges reserved state",
          "[colcomp_mat]") {
    // mtemp：布局 {1,3,6}，active {3}（1、6 为预留占位）
    Dense md = Dense::Zero(2, 7);
    md(1, 3) = 5.0;
    ColCompMat mtemp(2, 7);
    mtemp.setLayout({1, 3, 6});
    REQUIRE(mtemp.initializeFromSparse(toSparse(md)));
    REQUIRE(mtemp.activeCols() == 1);

    // 预留状态（±(列+1) 编码）：额外带列 4
    const std::vector<int32_t> reserved{-5};
    ColCompMat mu;
    REQUIRE(mu.initializeMuFromInputs(3, mtemp, &reserved));
    REQUIRE(mu.rows() == 3);
    REQUIRE(mu.totalCols() == 7);
    // 列集 = mtemp 有效列(3) ∪ 预留解码(4) = {3,4}，全部为占位（负）
    REQUIRE(mu.storedCols() == std::vector<int64_t>{3, 4});
    REQUIRE(mu.activeCols() == 0);
    REQUIRE(mu.numericReady());
    // 解码校验：乱序预留被拒绝
    ColCompMat bad;
    const std::vector<int32_t> disorder{-7, -5};
    REQUIRE(!bad.initializeMuFromInputs(3, mtemp, &disorder));
}

TEST_CASE("opsColSolve matches dense dsytrs solve", "[col_ops]") {
    const int64_t n = 12;
    Dense kd = randomDense(n, n, 1.0);
    kd = kd.transpose() * kd + 0.5 * Dense::Identity(n, n);  // SPD（dsytrf 容忍不定，此处取正定便于校验）
    Dense lu = kd;
    std::vector<int32_t> ipiv;
    REQUIRE(sytrfFactor(lu, ipiv) == 0);

    // 右端 D：布局 {1,4,8,11}，全部有效
    Dense dd = randomDense(n, 12, 1.0);
    ColCompMat D(n, 12);
    D.setLayout({1, 4, 8, 11});
    REQUIRE(D.initializeFromSparse(toSparse(dd)));

    ColCompMat X;
    REQUIRE(opsColSolve(n, lu, ipiv, D, X));
    REQUIRE(X.storedCols() == std::vector<int64_t>{1, 4, 8, 11});
    Dense full = kd.llt().solve(dd);
    Dense expect = Dense::Zero(n, 12);  // X 仅保留有效列, 其余列膨胀为 0
    for (int64_t c : {1, 4, 8, 11}) expect.col(c) = full.col(c);
    expectDenseEqual(X, expect, 1e-10);
}

TEST_CASE("opsColSolve keeps only active right-hand columns", "[col_ops]") {
    const int64_t n = 6;
    Dense kd = randomDense(n, n, 1.0);
    kd = kd.transpose() * kd + Dense::Identity(n, n);
    Dense lu = kd;
    std::vector<int32_t> ipiv;
    REQUIRE(sytrfFactor(lu, ipiv) == 0);

    Dense dd = Dense::Zero(n, n);
    dd.col(0).setRandom();
    dd.col(5).setRandom();  // 布局 {0,2,5}，active {0,5}，2 为占位
    ColCompMat D(n, n);
    D.setLayout({0, 2, 5});
    REQUIRE(D.initializeFromSparse(toSparse(dd)));

    ColCompMat X;
    REQUIRE(opsColSolve(n, lu, ipiv, D, X));
    REQUIRE(X.storedCols() == std::vector<int64_t>{0, 5});
    Dense expect = kd.llt().solve(dd);
    expectDenseEqual(X, expect, 1e-11);
}

namespace {

/// opsColAcc 对拍骨架：随机 B/C 列压缩块，gemv/gemm 两路与稠密参考比。
void checkAccAgainstDense(const ColCompMat& B, const ColCompMat& C,
                          const Dense& Bd, bool transB) {
    Dense cd;
    C.inflate(cd);
    Dense expect = transB ? Dense(Bd.transpose() * cd) : Dense(Bd * cd);
    ColCompMat fullA(B.totalCols() > 0 && transB ? B.totalCols() : Bd.rows(),
                     C.totalCols());
    // A 布局：覆盖 C 有效列即可（含占位超集）
    std::vector<int64_t> aCols;
    for (int64_t s = 0; s < C.validCols(); ++s)
        aCols.push_back(C.origColAt(s));
    if (aCols.empty()) aCols.push_back(0);
    fullA.setLayout(aCols);
    REQUIRE(fullA.initializeZero());

    ColCompMat gemvA = fullA, gemmA = fullA;
    REQUIRE(opsColAccGemv(B, transB, C, gemvA));
    REQUIRE(opsColAccGemm(B, transB, C, gemmA));
    expectDenseEqual(gemvA, expect, 1e-12);
    expectDenseEqual(gemmA, expect, 1e-12);

    // 稠密左端两路
    ColCompMat dGemvA = fullA, dGemmA = fullA;
    REQUIRE(opsColAccGemv(Bd, transB, C, dGemvA));
    REQUIRE(opsColAccGemm(Bd, transB, C, dGemmA));
    expectDenseEqual(dGemvA, expect, 1e-12);
    expectDenseEqual(dGemmA, expect, 1e-12);
}

} // namespace

TEST_CASE("opsColAcc gemv and gemm match dense product (non-trans)",
          "[col_ops]") {
    // B：4 行 × 5 总列（B·C 要求 B.totalCols == C.rows），布局 {0,2,3}，
    // 列 2 为占位
    Dense bd = Dense::Zero(4, 5);
    bd.col(0).setRandom();
    bd.col(3).setRandom();
    ColCompMat B(4, 5);
    B.setLayout({0, 2, 3});
    REQUIRE(B.initializeFromSparse(toSparse(bd)));
    // C：5 行 × 6 总列，布局 {1,2,4}，列 2 占位
    Dense cd = Dense::Zero(5, 6);
    cd.col(1).setRandom();
    cd.col(4).setRandom();
    ColCompMat C(5, 6);
    C.setLayout({1, 2, 4});
    REQUIRE(C.initializeFromSparse(toSparse(cd)));
    checkAccAgainstDense(B, C, bd, false);
}

TEST_CASE("opsColAcc gemv and gemm match dense product (trans)",
          "[col_ops]") {
    // B：7 行 × 5 总列（转置后乘积行 = B.totalCols = 5）
    Dense bd = Dense::Zero(7, 5);
    bd.col(1).setRandom();
    bd.col(4).setRandom();
    ColCompMat B(7, 5);
    B.setLayout({0, 1, 4});
    REQUIRE(B.initializeFromSparse(toSparse(bd)));
    Dense cd = Dense::Zero(7, 3);
    cd.col(0).setRandom();
    cd.col(2).setRandom();
    ColCompMat C(7, 3);
    C.setLayout({0, 2});
    REQUIRE(C.initializeFromSparse(toSparse(cd)));
    checkAccAgainstDense(B, C, bd, true);
}

TEST_CASE("opsColAcc activates placeholder target columns on write",
          "[col_ops]") {
    Dense bd = Dense::Zero(3, 3);
    bd.col(0).setRandom();
    ColCompMat B(3, 3);
    B.setLayout({0, 1, 2});
    REQUIRE(B.initializeFromSparse(toSparse(bd)));
    Dense cd = Dense::Zero(3, 2);
    cd.col(1).setRandom();
    ColCompMat C(3, 2);
    C.setLayout({0, 1});
    REQUIRE(C.initializeFromSparse(toSparse(cd)));
    // A 布局含占位列 1（C 的 active 列），写入后应激活
    ColCompMat A(3, 2);
    A.setLayout({1});
    REQUIRE(A.initializeZero());
    REQUIRE(A.activeCols() == 0);
    REQUIRE(opsColAccGemm(B, false, C, A));
    REQUIRE(A.activeCols() == 1);
    Dense expect = bd * Dense(cd);
    expectDenseEqual(A, expect, 1e-13);
}

TEST_CASE("opsColAcc rejects uncovered active columns", "[col_ops]") {
    Dense bd = Dense::Zero(2, 2);
    bd.col(0).setRandom();
    ColCompMat B(2, 2);
    B.setLayout({0, 1});
    REQUIRE(B.initializeFromSparse(toSparse(bd)));
    Dense cd = Dense::Zero(2, 3);
    cd.col(2).setRandom();
    ColCompMat C(2, 3);
    C.setLayout({2});
    REQUIRE(C.initializeFromSparse(toSparse(cd)));
    ColCompMat A(2, 3);
    A.setLayout({0});  // 未覆盖 C 的有效列 2
    REQUIRE(A.initializeZero());
    REQUIRE(!opsColAccGemm(B, false, C, A));
    REQUIRE(!opsColAccGemv(B, false, C, A));
}

TEST_CASE("opsColDenseProd matches dense product with placeholder rows",
          "[col_ops]") {
    Dense ad = Dense::Zero(3, 5);
    ad.col(0).setRandom();
    ad.col(3).setRandom();  // A：3×5，active {0,3}
    ColCompMat A(3, 5);
    A.setLayout({0, 1, 3, 4});  // 含占位 1、4
    REQUIRE(A.initializeFromSparse(toSparse(ad)));
    Dense b = randomDense(5, 4, 1.0);
    Dense out;
    REQUIRE(opsColDenseProd(A, b, out));
    REQUIRE(out.isApprox(ad * b, 1e-12));
    // 全列激活走零拷贝路径
    Dense adFull = randomDense(3, 5, 1.0);
    ColCompMat AFull(3, 5);
    AFull.setLayout({0, 1, 2, 3, 4});
    REQUIRE(AFull.initializeFromSparse(toSparse(adFull)));
    REQUIRE(opsColDenseProd(AFull, b, out));
    REQUIRE(out.isApprox(adFull * b, 1e-12));
}
