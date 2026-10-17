#pragma once
// OpenBLAS CBLAS 封装（列主序 = Eigen 默认布局）。
// 转置通过 CblasTrans/CblasNoTrans 标志“隐式”完成，不物化转置副本。
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <utility>

#include <openblas/cblas.h>

#include "core/types.hpp"

namespace aams {
namespace blas {

/// C = alpha·op(A)·op(B) + beta·C（op 由转置标志指定）。
/// A/B 用 Eigen::Ref 接受稠密矩阵的连续行块（列主序、满行高），
/// 调用前 C 必须已按 beta 语义初始化（beta=0 时无需预清零，CBLAS 不读 C）。
inline void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                 const Eigen::Ref<const DefaultTraits::DenseMatrix>& A,
                 const Eigen::Ref<const DefaultTraits::DenseMatrix>& B,
                 double beta, DefaultTraits::DenseMatrix& C) {
    const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
    const int k = static_cast<int>((ta == CblasNoTrans) ? A.cols() : A.rows());
    const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
    cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A.data(),
                static_cast<int>(A.rows()), B.data(),
                static_cast<int>(B.rows()), beta, C.data(),
                static_cast<int>(C.rows()));
}

/// C = op(A)·op(B)（C 由本函数按结果尺寸创建/重置）。
inline void gemm0(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                  const Eigen::Ref<const DefaultTraits::DenseMatrix>& A,
                  const Eigen::Ref<const DefaultTraits::DenseMatrix>& B,
                  DefaultTraits::DenseMatrix& C) {
    const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
    const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
    C.resize(m, n);
    gemm(ta, tb, 1.0, A, B, 0.0, C);
}

/// C -= op(A)·op(B)（C 预清零后累加用 beta=1 链；此处为独立封装）。
inline void subMul(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                   const Eigen::Ref<const DefaultTraits::DenseMatrix>& A,
                   const Eigen::Ref<const DefaultTraits::DenseMatrix>& B,
                   DefaultTraits::DenseMatrix& C) {
    gemm(ta, tb, -1.0, A, B, 1.0, C);
}

/// C += op(A)·op(B)。
inline void addMul(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                   const Eigen::Ref<const DefaultTraits::DenseMatrix>& A,
                   const Eigen::Ref<const DefaultTraits::DenseMatrix>& B,
                   DefaultTraits::DenseMatrix& C) {
    gemm(ta, tb, 1.0, A, B, 1.0, C);
}

/// OpenBLAS 线程数控制（并行阶段防过度订阅）。
/// MSYS2 的 libopenblas 链 libgomp（OpenMP 版）：其内部 gemm 并行由
/// GOMP 线程池驱动，openblas_set_num_threads 无法限制，必须同时
/// omp_set_num_threads（否则 12 worker × BLAS 内部线程互踩，实测
/// 节点池 wall 不降、CPU 膨胀近一个数量级）。
/// 工程未以 -fopenmp 编译（_OPENMP 不存在），直接声明 libgomp 的 C 接口
/// （libgomp-1.dll 随 OpenBLAS 常驻进程）。
extern "C" void omp_set_num_threads(int);
extern "C" int omp_get_max_threads();
inline void setNumThreads(int t) {
    openblas_set_num_threads(t);
    omp_set_num_threads(t);
}

/// 将**调用线程**的 BLAS 并行度钉为单线程。nthreads-var 是每线程 ICV：
/// 主线程 setNumThreads(1) 不影响 std::thread 工作线程（它们持初值 =
/// 核数，BLAS 在其上仍全开核 → 过度订阅）。工作线程入口必须各调一次。
inline void pinThisThread() { omp_set_num_threads(1); }

// BLAS 内部线程与手工列拆分共用一份非阻塞预算。回调仅在 DAG worker
// 上安装；acquire 在调度器锁内授予总线程数并登记额外线程，release 归还。
extern thread_local bool tls_starved;
extern thread_local bool tls_budgetActive;
extern thread_local std::function<int()> tls_availableThreads;
extern thread_local std::function<int(int)> tls_acquireThreads;
extern thread_local std::function<void(int)> tls_releaseThreads;
extern std::atomic_flag g_mtBudgetLock;

inline bool adaptationEnabled() {
    return std::getenv("AMLS_NO_ADAPT") == nullptr;
}

inline int availableThreads() {
    if (!adaptationEnabled()) return 1;
    return tls_availableThreads ? tls_availableThreads() : 1;
}

// 仅预留容量，不修改 OpenMP ICV。必须在取得预算的同一 worker 上析构；
// 持有者只能等待自己的辅助线程，不能等待尚在 DAG 队列中的任务。
struct ScopedReservation {
    bool active = false;
    int threads = 1;
    static inline std::atomic<int64_t> nBudgetGranted{0};
    static inline std::atomic<int64_t> nBudgetContended{0};
    static inline std::atomic<int64_t> nHungry{0};

    explicit ScopedReservation(int k) {
        if (std::getenv("AMLS_NO_MT") || !adaptationEnabled() ||
            tls_budgetActive || k <= 1 || !tls_acquireThreads ||
            !tls_releaseThreads)
            return;
        // 先复制归还回调，避免登记容量后 std::function 分配失败漏还。
        release_ = tls_releaseThreads;
        nHungry.fetch_add(1, std::memory_order_relaxed);
        if (g_mtBudgetLock.test_and_set(std::memory_order_acquire)) {
            nBudgetContended.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // 锁顺序只有预算 try_lock → DAG qm；qm 持有者从不获取预算锁。
        try {
            threads = tls_acquireThreads(k);
        } catch (...) {
            g_mtBudgetLock.clear(std::memory_order_release);
            throw;
        }
        if (threads <= 1) {
            threads = 1;
            g_mtBudgetLock.clear(std::memory_order_release);
            return;
        }
        tls_budgetActive = true;
        active = true;
        nBudgetGranted.fetch_add(1, std::memory_order_relaxed);
    }
    ~ScopedReservation() {
        if (active) {
            release_(threads - 1);
            tls_budgetActive = false;
            g_mtBudgetLock.clear(std::memory_order_release);
        }
    }
    ScopedReservation(const ScopedReservation&) = delete;
    ScopedReservation& operator=(const ScopedReservation&) = delete;
    ScopedReservation(ScopedReservation&& o) noexcept
        : active(o.active), threads(o.threads), release_(std::move(o.release_)) {
        o.active = false;
        o.threads = 1;
    }
    ScopedReservation& operator=(ScopedReservation&&) = delete;

private:
    std::function<void(int)> release_;
};

struct ScopedBudget {
    bool active = false;
    int saved = 1;
    static inline auto& nBudgetGranted = ScopedReservation::nBudgetGranted;
    static inline auto& nBudgetContended = ScopedReservation::nBudgetContended;
    static inline auto& nHungry = ScopedReservation::nHungry;
    explicit ScopedBudget(int k) : reservation_(k) {
        if (!reservation_.active) return;
        saved = omp_get_max_threads();
        omp_set_num_threads(reservation_.threads);
        active = true;
    }
    ~ScopedBudget() {
        // 先恢复 ICV，再由 reservation_ 析构归还容量与全局互斥预算。
        if (active) omp_set_num_threads(saved);
    }
    ScopedBudget(const ScopedBudget&) = delete;
    ScopedBudget& operator=(const ScopedBudget&) = delete;
    ScopedBudget(ScopedBudget&& o) noexcept
        : active(o.active), saved(o.saved), reservation_(std::move(o.reservation_)) {
        o.active = false;
    }
    ScopedBudget& operator=(ScopedBudget&&) = delete;

private:
    ScopedReservation reservation_;
};

inline ScopedBudget maybeBudget(double gflops, int k = 8) {
    return ScopedBudget(gflops > 0.5 ? k : 1);
}

} // namespace blas
} // namespace aams
