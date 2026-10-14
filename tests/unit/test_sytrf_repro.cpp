// dsytrf/dsytrs 绑定最小复现：三对角链矩阵逐规模残差扫描。
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>

#include "linalg/col_ops.hpp"

using DM = Eigen::MatrixXd;

namespace {
DM chainK(int64_t n) {
    DM K = DM::Zero(n, n);
    for (int64_t i = 0; i < n; ++i) {
        K(i, i) = 2.0;
        if (i + 1 < n) {
            K(i, i + 1) = -1.0;
            K(i + 1, i) = -1.0;
        }
    }
    return K;
}
} // namespace

TEST_CASE("sytrf/sytrs 绑定逐规模残差", "[sytrf]") {
    for (int64_t n : {10, 50, 100, 200, 500, 999, 1000}) {
        DM K = chainK(n);
        Eigen::LLT<DM> llt(K);
        DM rhs = DM::Random(n, 1);
        DM xRef = llt.solve(rhs);
        DM lu = K;
        std::vector<int32_t> ipiv;
        const int32_t fInfo = amls::sytrfFactor(lu, ipiv);
        DM x = rhs;
        const int32_t sInfo = amls::sytrsSolve(static_cast<int32_t>(n), lu,
                                               ipiv, x.data(), 1);
        const double resid = (K * x - rhs).norm();
        const double xDiff = (x - xRef).norm();
        INFO("n=" << n << " fInfo=" << fInfo << " sInfo=" << sInfo
                  << " 残差=" << resid << " 与LLT差=" << xDiff);
        CHECK(resid <= 1e-10 * rhs.norm());
    }
}
