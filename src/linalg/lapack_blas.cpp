// GNU ld --wrap entry points for the BLAS3 calls in static NETLIB LAPACK.
// Keep the Fortran LP64 ABI, including gfortran's trailing CHARACTER lengths.
#include "linalg/blas.hpp"

#include <cstddef>
#include <vector>

namespace {
// LAPACK uses narrow panels (typically 32 columns), so the application-level
// 0.5 GF threshold would miss almost all calls. Below 8 MF, querying the DAG's
// mutex and starting an OpenMP team costs more than the potential benefit.
constexpr double kMinFlops = 8.0e6;

amls::blas::ScopedBudget lapackBudget(double flops) {
    if (flops < kMinFlops || amls::blas::tls_budgetActive ||
        std::getenv("AMLS_N7_SERIAL"))
        return amls::blas::ScopedBudget(1);
    // The size test above replaces maybeBudget's application-level threshold.
    // ScopedBudget queries the *current* idle capacity for each BLAS3 call.
    return amls::blas::maybeBudget(1.0);
}

double positive(int n) { return static_cast<double>(std::max(0, n)); }
double sideSize(const char* side, const int* m, const int* n) {
    return positive((*side == 'L' || *side == 'l') ? *m : *n);
}
} // namespace

extern "C" {
void __real_dgemm_(const char*, const char*, const int*, const int*, const int*,
                   const double*, const double*, const int*, const double*,
                   const int*, const double*, double*, const int*,
                   std::size_t, std::size_t);
void __real_dsyrk_(const char*, const char*, const int*, const int*,
                   const double*, const double*, const int*, const double*,
                   double*, const int*, std::size_t, std::size_t);
void __real_dsyr2k_(const char*, const char*, const int*, const int*,
                    const double*, const double*, const int*, const double*,
                    const int*, const double*, double*, const int*,
                    std::size_t, std::size_t);
void __real_dtrsm_(const char*, const char*, const char*, const char*,
                   const int*, const int*, const double*, const double*,
                   const int*, double*, const int*, std::size_t, std::size_t,
                   std::size_t, std::size_t);
void __real_dtrmm_(const char*, const char*, const char*, const char*,
                   const int*, const int*, const double*, const double*,
                   const int*, double*, const int*, std::size_t, std::size_t,
                   std::size_t, std::size_t);
void __real_dsymm_(const char*, const char*, const int*, const int*,
                   const double*, const double*, const int*, const double*,
                   const int*, const double*, double*, const int*,
                   std::size_t, std::size_t);

void __wrap_dgemm_(const char* ta, const char* tb, const int* m, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc,
                   std::size_t la, std::size_t lb) {
    auto budget = lapackBudget(2.0 * positive(*m) * positive(*n) * positive(*k));
    __real_dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, la, lb);
}

// OpenBLAS's DGEMMTR (added for LAPACK 3.12 support) silently produces wrong
// results in the DLASYF call pattern (n>64 factorizations; see
// tests/unit/test_sytrf_repro.cpp). Replace it with a full DGEMM into a
// temporary, then apply alpha/beta to the requested triangle only — the other
// triangle of C holds live factor data and must stay untouched.
void __wrap_dgemmtr_(const char* uplo, const char* ta, const char* tb,
                     const int* n, const int* k, const double* alpha,
                     const double* a, const int* lda, const double* b,
                     const int* ldb, const double* beta, double* c,
                     const int* ldc, std::size_t, std::size_t, std::size_t,
                     std::size_t) {
    const int nn = *n, kk = *k;
    if (nn <= 0 || kk <= 0) return;
    auto budget = lapackBudget(2.0 * static_cast<double>(nn) * nn * kk);
    std::vector<double> tmp(static_cast<size_t>(nn) * static_cast<size_t>(nn));
    const double zero = 0.0;
    __real_dgemm_(ta, tb, n, n, k, alpha, a, lda, b, ldb, &zero, tmp.data(), &nn,
                  0, 0);
    const bool lower = (*uplo == 'L' || *uplo == 'l');
    for (int j = 0; j < nn; ++j) {
        for (int i = 0; i < nn; ++i) {
            if (lower ? (i < j) : (i > j)) continue;
            double& cj = c[static_cast<size_t>(j) * (*ldc) + i];
            cj = (*beta) * cj + tmp[static_cast<size_t>(j) * nn + i];
        }
    }
}

void __wrap_dsyrk_(const char* uplo, const char* trans, const int* n,
                   const int* k, const double* alpha, const double* a,
                   const int* lda, const double* beta, double* c,
                   const int* ldc, std::size_t lu, std::size_t lt) {
    auto budget = lapackBudget(positive(*n) * (positive(*n) + 1) * positive(*k));
    __real_dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc, lu, lt);
}

void __wrap_dsyr2k_(const char* uplo, const char* trans, const int* n,
                    const int* k, const double* alpha, const double* a,
                    const int* lda, const double* b, const int* ldb,
                    const double* beta, double* c, const int* ldc,
                    std::size_t lu, std::size_t lt) {
    auto budget = lapackBudget(2.0 * positive(*n) * (positive(*n) + 1) * positive(*k));
    __real_dsyr2k_(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, lu, lt);
}

void __wrap_dtrsm_(const char* side, const char* uplo, const char* trans,
                   const char* diag, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   double* b, const int* ldb, std::size_t ls, std::size_t lu,
                   std::size_t lt, std::size_t ld) {
    auto budget = lapackBudget(positive(*m) * positive(*n) * sideSize(side, m, n));
    __real_dtrsm_(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb,
                  ls, lu, lt, ld);
}

void __wrap_dtrmm_(const char* side, const char* uplo, const char* trans,
                   const char* diag, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   double* b, const int* ldb, std::size_t ls, std::size_t lu,
                   std::size_t lt, std::size_t ld) {
    auto budget = lapackBudget(positive(*m) * positive(*n) * sideSize(side, m, n));
    __real_dtrmm_(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb,
                  ls, lu, lt, ld);
}

void __wrap_dsymm_(const char* side, const char* uplo, const int* m,
                   const int* n, const double* alpha, const double* a,
                   const int* lda, const double* b, const int* ldb,
                   const double* beta, double* c, const int* ldc,
                   std::size_t ls, std::size_t lu) {
    auto budget = lapackBudget(2.0 * positive(*m) * positive(*n) * sideSize(side, m, n));
    __real_dsymm_(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, ls, lu);
}
} // extern "C"
