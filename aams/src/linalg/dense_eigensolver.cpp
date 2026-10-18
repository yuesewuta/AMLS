#include "linalg/dense_eigensolver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "linalg/lapack.h"

namespace aams {

// dsygvx 分段剖析（AAMS_PROFILE）：workspace 查询 / 实解。
struct GvxProf {
    std::atomic<uint64_t> qNs{0}, rNs{0}, qCalls{0}, rCalls{0};
    static GvxProf& inst() {
        static GvxProf p;
        return p;
    }
    ~GvxProf() {
        if (std::getenv("AAMS_PROFILE") == nullptr) return;
        std::fprintf(stderr,
                     "[prof] dsygvx: query=%.3fs (%llu) real=%.3fs (%llu)\n",
                     qNs.load() * 1e-9, (unsigned long long)qCalls.load(),
                     rNs.load() * 1e-9, (unsigned long long)rCalls.load());
    }
};

template <typename T>
void LapackDenseEigensolver<T>::solveAll(DenseMatrix& K, DenseMatrix& M,
                                         Vector& w, DenseMatrix& V) {
    const int32_t n = static_cast<int32_t>(K.rows());
    if (n == 0) {
        w.resize(0);
        V.resize(0, 0);
        return;
    }
    const int32_t itype = 1;
    const char jobz = 'V', uplo = 'L';
    w.resize(n);

    int32_t lwork = -1, liwork = -1, info = 0;
    double wkopt = 0;
    int32_t iwkopt = 0;
    dsygvd_(&itype, &jobz, &uplo, &n, K.data(), &n, M.data(), &n, w.data(),
            &wkopt, &lwork, &iwkopt, &liwork, &info);
    if (info != 0)
        throw std::runtime_error("dsygvd workspace 查询失败");
    lwork  = static_cast<int32_t>(wkopt);
    liwork = iwkopt;
    std::vector<double> work(static_cast<size_t>(lwork));
    std::vector<int32_t> iwork(static_cast<size_t>(liwork));

    dsygvd_(&itype, &jobz, &uplo, &n, K.data(), &n, M.data(), &n, w.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        throw std::runtime_error("dsygvd 失败, info=" + std::to_string(info));

    V = std::move(K);  // 特征向量覆盖 A
}

template <typename T>
void LapackDenseEigensolver<T>::solveInterval(DenseMatrix& K, DenseMatrix& M,
                                              Scalar lo, Scalar hi, Vector& w,
                                              DenseMatrix& V) {
    const int32_t n = static_cast<int32_t>(K.rows());
    if (n == 0 || !(hi > lo)) {
        w.resize(0);
        V.resize(n, 0);
        return;
    }
    const int32_t itype = 1;
    const char jobz = 'V', uplo = 'L';
    const double vl = lo, vu = hi;
    // dsygvx 区间语义: 用值区间 [vl,vu] 时须设 IL=1, IU=N
    // (同时给定 IL/IU 与 VL/VU 属未定义行为, 曾致 info=92 B 分解失败)。
    const int32_t il = 1, iu = n;
    const double abstol = 0.0;
    int32_t m = 0, info = 0;

    w.resize(n);
    DenseMatrix Z(n, n);

    int32_t lwork = -1;
    double wkopt = 0;
    std::vector<int32_t> iwork(5 * static_cast<size_t>(n));
    std::vector<int32_t> ifail(static_cast<size_t>(n));
    {
        const auto tq0 = std::chrono::steady_clock::now();
        dsygvx_(&itype, &jobz, const_cast<char*>("V"), &uplo, &n, K.data(), &n,
                M.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, w.data(), Z.data(),
                &n, &wkopt, &lwork, iwork.data(), ifail.data(), &info);
        auto& gp = GvxProf::inst();
        gp.qNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<
                                              std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tq0).count()),
                         std::memory_order_relaxed);
        gp.qCalls.fetch_add(1, std::memory_order_relaxed);
    }
    if (info != 0)
        throw std::runtime_error("dsygvx workspace 查询失败");
    lwork = static_cast<int32_t>(wkopt);
    std::vector<double> work(static_cast<size_t>(lwork));

    {
        const auto tr0 = std::chrono::steady_clock::now();
        dsygvx_(&itype, &jobz, const_cast<char*>("V"), &uplo, &n, K.data(), &n,
                M.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, w.data(), Z.data(),
                &n, work.data(), &lwork, iwork.data(), ifail.data(), &info);
        auto& gp = GvxProf::inst();
        gp.rNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<
                                              std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tr0).count()),
                         std::memory_order_relaxed);
        gp.rCalls.fetch_add(1, std::memory_order_relaxed);
    }
    if (info != 0)
        throw std::runtime_error("dsygvx 失败, info=" + std::to_string(info));
    if (m < 0 || m > n)
        throw std::runtime_error("dsygvx 返回非法计数 m");

    w = w.head(m).eval();
    V = Z.leftCols(m).eval();
}

template <typename T>
void LapackDenseEigensolver<T>::solveIndex(DenseMatrix& K, DenseMatrix& M,
                                           int64_t il, int64_t iu, Vector& w,
                                           DenseMatrix& V) {
    const int32_t n = static_cast<int32_t>(K.rows());
    if (n == 0 || iu < il) {
        w.resize(0);
        V.resize(n, 0);
        return;
    }
    const int32_t itype = 1;
    const char jobz = 'V', uplo = 'L';
    const double vl = 0.0, vu = 0.0;
    const int32_t il32 = static_cast<int32_t>(il);
    const int32_t iu32 =
        static_cast<int32_t>(std::min<int64_t>(iu, n));
    const double abstol = 0.0;
    int32_t m = 0, info = 0;

    w.resize(n);
    DenseMatrix Z(n, n);

    int32_t lwork = -1;
    double wkopt = 0;
    std::vector<int32_t> iwork(5 * static_cast<size_t>(n));
    std::vector<int32_t> ifail(static_cast<size_t>(n));
    {
        const auto tq0 = std::chrono::steady_clock::now();
        dsygvx_(&itype, &jobz, const_cast<char*>("I"), &uplo, &n, K.data(), &n,
                M.data(), &n, &vl, &vu, &il32, &iu32, &abstol, &m, w.data(),
                Z.data(), &n, &wkopt, &lwork, iwork.data(), ifail.data(), &info);
        auto& gp = GvxProf::inst();
        gp.qNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<
                                              std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tq0).count()),
                         std::memory_order_relaxed);
        gp.qCalls.fetch_add(1, std::memory_order_relaxed);
    }
    if (info != 0)
        throw std::runtime_error("dsygvx workspace 查询失败");
    lwork = static_cast<int32_t>(wkopt);
    std::vector<double> work(static_cast<size_t>(lwork));

    {
        const auto tr0 = std::chrono::steady_clock::now();
        dsygvx_(&itype, &jobz, const_cast<char*>("I"), &uplo, &n, K.data(), &n,
                M.data(), &n, &vl, &vu, &il32, &iu32, &abstol, &m, w.data(),
                Z.data(), &n, work.data(), &lwork, iwork.data(), ifail.data(),
                &info);
        auto& gp = GvxProf::inst();
        gp.rNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<
                                              std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tr0).count()),
                         std::memory_order_relaxed);
        gp.rCalls.fetch_add(1, std::memory_order_relaxed);
    }
    if (info != 0)
        throw std::runtime_error("dsygvx 失败, info=" + std::to_string(info));
    if (m < 0 || m > n)
        throw std::runtime_error("dsygvx 返回非法计数 m");

    w = w.head(m).eval();
    V = Z.leftCols(m).eval();
}

template class LapackDenseEigensolver<DefaultTraits>;

} // namespace aams
