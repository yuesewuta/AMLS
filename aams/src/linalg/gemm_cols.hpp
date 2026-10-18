#pragma once
// 列分组并行 dgemm 工具（自 colcomp.hpp 抽出，新旧块算子层共用）。
//
// 设计要点：
//   * 各线程写不相交列段，无竞争、无锁、不依赖 BLAS 内部线程
//     （规避 libgomp 嵌套并行区风险）。
//   * 与 BLAS 预算共用容量（ScopedReservation）：借不到线程就串行。
//   * 算子层并行度 opsTlsThreads() 为 thread_local：大乘法入口按执行器
//     当前空闲容量刷新，不在 worker 里的调用保持串行。
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <openblas/cblas.h>

#include "core/types.hpp"
#include "linalg/blas.hpp"

namespace aams {

/// 算子层并行度（thread_local）：大乘法入口按执行器当前空闲容量刷新，
/// 不在 worker 里的调用保持串行。
inline int& opsTlsThreads() {
    static thread_local int value = 1;
    return value;
}

namespace colcomp_detail {

// 在线程组析构前保留捕获对象；调用方须在预留容量之后声明线程组。
class ThreadGroup {
public:
    explicit ThreadGroup(size_t count) { threads_.reserve(count); }
    ~ThreadGroup() { join(); }
    ThreadGroup(const ThreadGroup&) = delete;
    ThreadGroup& operator=(const ThreadGroup&) = delete;

    template <class F>
    void start(F&& fn) {
        threads_.emplace_back([this, fn = std::forward<F>(fn)]() mutable {
            run(fn);
        });
    }
    template <class F>
    void run(F&& fn) noexcept {
        try {
            std::forward<F>(fn)();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!firstEx_) firstEx_ = std::current_exception();
        }
    }
    void wait() {
        join();
        if (firstEx_) std::rethrow_exception(firstEx_);
    }

private:
    void join() {
        for (auto& th : threads_)
            if (th.joinable()) th.join();
    }
    std::mutex mutex_;
    std::exception_ptr firstEx_;
    std::vector<std::thread> threads_;
};

} // namespace colcomp_detail

/// 剖析计数（AAMS_PROFILE=1 时在 REDUCTION 收尾打印）。
struct GemmColsStats {
    static inline std::atomic<int64_t> calls{0};
    static inline std::atomic<int64_t> smallFlops{0};
    static inline std::atomic<int64_t> fewCols{0};
    static inline std::atomic<int64_t> noGrant{0};
    static inline std::atomic<int64_t> parallel{0};
    static inline std::atomic<int64_t> helperThreads{0};
    static inline void report() {
        std::fprintf(stderr,
                     "[gemmcols] calls=%lld smallFlops=%lld fewCols=%lld "
                     "noGrant=%lld parallel=%lld helperThreads=%lld\n",
                     (long long)calls.load(std::memory_order_relaxed),
                     (long long)smallFlops.load(std::memory_order_relaxed),
                     (long long)fewCols.load(std::memory_order_relaxed),
                     (long long)noGrant.load(std::memory_order_relaxed),
                     (long long)parallel.load(std::memory_order_relaxed),
                     (long long)helperThreads.load(std::memory_order_relaxed));
    }
};

/// 原始指针列分组并行 dgemm（tb=CblasNoTrans 专用）：
/// C(:, jseg) = alpha·op(A)·op(B)(:, jseg) + beta·C(:,jseg)。
inline void dgemmColsRaw(CBLAS_TRANSPOSE ta, int m, int n, int k,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb, double beta, double* C,
                         int ldc, int nthreads) {
    if (n <= 0 || m <= 0) return;
    GemmColsStats::calls.fetch_add(1, std::memory_order_relaxed);
    static const bool forceSerial = std::getenv("AMLS_DGEMM_SERIAL") != nullptr;
    if (forceSerial) nthreads = 1;
    // 小乘法直接串行：ThreadGroup 每次 spawn/join/pin 的固定开销 ms 级，
    // 而关键链上大量小调用（如 1×3×66）的 flops 只有 µs 级，分段并行
    // 净亏千倍（cant_20 时间线实测：链节点 ~29 次批量 × ~7ms 线程开销）。
    if (2.0 * static_cast<double>(m) * n * k < 3e7) {
        GemmColsStats::smallFlops.fetch_add(1, std::memory_order_relaxed);
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, n, k, alpha, A, lda,
                    B, ldb, beta, C, ldc);
        return;
    }
    const int capped = std::min<int>(nthreads, (n + 31) / 32);
    if (capped <= 1) {
        GemmColsStats::fewCols.fetch_add(1, std::memory_order_relaxed);
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, n, k, alpha, A, lda,
                    B, ldb, beta, C, ldc);
        return;
    }
    nthreads = capped;
    // 手工列分块与 BLAS 预算共用容量：借不到（队列忙/互斥被占）就串行。
    blas::ScopedReservation reserve(nthreads);
    nthreads = reserve.threads;
    if (nthreads <= 1) {
        GemmColsStats::noGrant.fetch_add(1, std::memory_order_relaxed);
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, n, k, alpha, A, lda,
                    B, ldb, beta, C, ldc);
        return;
    }
    GemmColsStats::parallel.fetch_add(1, std::memory_order_relaxed);
    GemmColsStats::helperThreads.fetch_add(nthreads - 1,
                                           std::memory_order_relaxed);
    auto runSeg = [&](int j0, int j1) {
        if (j1 <= j0) return;
        const int jn = j1 - j0;
        blas::pinThisThread();   // 临时线程：OpenBLAS(OMP 版) ICV 钉 1，
                                 // 否则每线程内 dgemm 再开满核 → 过度订阅
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, jn, k, alpha, A, lda,
                    B + static_cast<int64_t>(j0) * ldb, ldb, beta,
                    C + static_cast<int64_t>(j0) * ldc, ldc);
    };
    // 声明在预算之后：先 join 辅助线程，再归还预留容量。
    colcomp_detail::ThreadGroup helper(static_cast<size_t>(nthreads - 1));
    const int64_t base = n / nthreads, extra = n % nthreads;
    int64_t j = 0;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t jn = base + (t < extra ? 1 : 0);
        const int64_t j0 = j, j1 = j + jn;
        j = j1;
        if (t == nthreads - 1)
            runSeg(j0, j1);
        else
            helper.start([runSeg, j0, j1] { runSeg(j0, j1); });
    }
    helper.wait();
}

/// 列分组并行 dgemm：C(:, jseg) = alpha·op(A)·op(B)(:, jseg) + beta·C(:,jseg)。
/// tb=Trans 时 op(B) 的列段 = B 的行段（列主序下行不连续）→ 线程内先
/// 物化成连续段再算；对调用方完全等价于单次 dgemm。
inline void dgemmCols(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const DefaultTraits::DenseMatrix& A,
                      const DefaultTraits::DenseMatrix& B, double beta,
                      DefaultTraits::DenseMatrix& C, int nthreads) {
    using Dense = DefaultTraits::DenseMatrix;
    const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
    const int k = static_cast<int>((ta == CblasNoTrans) ? A.cols() : A.rows());
    const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
    if (n <= 0 || m <= 0) return;
    if (tb == CblasNoTrans) {
        dgemmColsRaw(ta, m, n, k, alpha, A.data(),
                     static_cast<int>(A.rows()), B.data(),
                     static_cast<int>(B.rows()), beta, C.data(),
                     static_cast<int>(C.rows()), nthreads);
        return;
    }
    static const bool forceSerial = std::getenv("AMLS_DGEMM_SERIAL") != nullptr;
    if (forceSerial) nthreads = 1;
    // 同 dgemmColsRaw：小乘法串行，免 ThreadGroup 固定开销。
    if (2.0 * static_cast<double>(m) * n * k < 3e7) {
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A.data(),
                    static_cast<int>(A.rows()), B.data(),
                    static_cast<int>(B.rows()), beta, C.data(),
                    static_cast<int>(C.rows()));
        return;
    }
    nthreads = std::min<int>(nthreads, static_cast<int>((n + 31) / 32));
    blas::ScopedReservation reserve(nthreads);
    nthreads = reserve.threads;
    if (nthreads <= 1) {
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A.data(),
                    static_cast<int>(A.rows()), B.data(),
                    static_cast<int>(B.rows()), beta, C.data(),
                    static_cast<int>(C.rows()));
        return;
    }
    const int ldb = static_cast<int>(B.rows());
    const int ldc = static_cast<int>(C.rows());
    auto runSeg = [&](int j0, int j1) {
        if (j1 <= j0) return;
        const int jn = j1 - j0;
        blas::pinThisThread();
        // op(B)=Bᵀ，op(B)(r, j0+c) = B(j0+c, r)：物化 bSeg(r,c) = B(j0+c, r)。
        Dense bSeg(k, jn);
        for (int c = 0; c < jn; ++c)
            for (int r = 0; r < k; ++r)
                bSeg(r, c) = B(static_cast<int64_t>(j0) + c,
                               static_cast<int64_t>(r));
        double* cPtr = C.data() + static_cast<int64_t>(j0) * ldc;
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, jn, k, alpha, A.data(),
                    static_cast<int>(A.rows()), bSeg.data(), k, beta, cPtr,
                    ldc);
    };
    colcomp_detail::ThreadGroup helper(static_cast<size_t>(nthreads - 1));
    const int64_t base = n / nthreads, extra = n % nthreads;
    int64_t j = 0;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t jn = base + (t < extra ? 1 : 0);
        const int64_t j0 = j, j1 = j + jn;
        j = j1;
        if (t == nthreads - 1)
            runSeg(j0, j1);
        else
            helper.start([runSeg, j0, j1] { runSeg(j0, j1); });
    }
    helper.wait();
    (void)sizeof(Dense);
}

} // namespace aams
