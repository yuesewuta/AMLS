#include "reference/error_metrics.hpp"

#include <cmath>

namespace aams {

std::vector<double> ErrorMetrics::eigenvalueRelErr(
    const DefaultTraits::Vector& theta, const DefaultTraits::Vector& lambdaRef) {
    const int64_t m = std::min<int64_t>(theta.size(), lambdaRef.size());
    std::vector<double> e(static_cast<size_t>(m));
    for (int64_t j = 0; j < m; ++j) {
        const double denom = std::abs(lambdaRef[j]);
        e[static_cast<size_t>(j)] =
            denom > 0 ? std::abs(theta[j] - lambdaRef[j]) / denom
                      : std::abs(theta[j] - lambdaRef[j]);
    }
    return e;
}

double ErrorMetrics::maxEigenvalueRelErr(const DefaultTraits::Vector& theta,
                                         const DefaultTraits::Vector& lambdaRef) {
    const auto e = eigenvalueRelErr(theta, lambdaRef);
    double mx = 0.0;
    for (double v : e)
        mx = std::max(mx, v);
    return mx;
}

double ErrorMetrics::residual(const DefaultTraits::SparseMatrix& K,
                              const DefaultTraits::SparseMatrix& M, double theta,
                              const DefaultTraits::Vector& z) {
    const DefaultTraits::Vector Kz = K * z;
    const DefaultTraits::Vector Mz = M * z;
    const double num = (Kz - theta * Mz).norm();
    const double den = Kz.norm() + std::abs(theta) * Mz.norm();
    return den > 0 ? num / den : 0.0;
}

std::vector<double> ErrorMetrics::frequencyRelErr(
    const DefaultTraits::Vector& theta, const DefaultTraits::Vector& lambdaRef) {
    const int64_t m = std::min<int64_t>(theta.size(), lambdaRef.size());
    std::vector<double> e(static_cast<size_t>(m));
    for (int64_t j = 0; j < m; ++j) {
        const double sr = std::sqrt(std::max(lambdaRef[j], 0.0));
        e[static_cast<size_t>(j)] =
            sr > 0 ? std::abs(std::sqrt(std::max(theta[j], 0.0)) - sr) / sr
                   : 0.0;
    }
    return e;
}

} // namespace aams
