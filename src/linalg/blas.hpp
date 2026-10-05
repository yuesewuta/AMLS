#pragma once
// OpenBLAS CBLAS 封装（列主序 = Eigen 默认布局）。
// 转置通过 CblasTrans/CblasNoTrans 标志“隐式”完成，不物化转置副本。
#include <atomic>

#include <openblas/cblas.h>

#include "core/types.hpp"

namespace amls {
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
inline void setNumThreads(int t) {
    openblas_set_num_threads(t);
    omp_set_num_threads(t);
}

/// 将**调用线程**的 BLAS 并行度钉为单线程。nthreads-var 是每线程 ICV：
/// 主线程 setNumThreads(1) 不影响 std::thread 工作线程（它们持初值 =
/// 核数，BLAS 在其上仍全开核 → 过度订阅）。工作线程入口必须各调一次。
inline void pinThisThread() { omp_set_num_threads(1); }

// ---- 大任务内部临时放开 BLAS 多线程（尾部单任务占满多核）----
// 观测（250k trace）：收尾 ~15.4s（占 wall 10%）就绪队列空、单任务 3–5.5s
// 独占 1 核。方案 = 就绪队列空闸（starved）+ 按规模阈值 + 全局同时只允许
// 一个 MT 任务（atomic_flag，防 12 个大任务各拉 8 线程打爆核）。
extern thread_local bool tls_starved;      // DagRunner 弹出任务时更新
extern std::atomic_flag g_mtBudgetLock;    // MT 预算互斥（try_lock 非阻塞）

struct ScopedBudget {
    bool active = false;
    int saved = 1;
    explicit ScopedBudget(int k) {
        if (!tls_starved || k <= 1)
            return;
        if (g_mtBudgetLock.test_and_set(std::memory_order_acquire))
            return;   // 已有别的任务持有 MT 预算，本次保持单线程
        saved = 1;
        omp_set_num_threads(k);
        active = true;
    }
    ~ScopedBudget() {
        if (active) {
            omp_set_num_threads(saved);
            g_mtBudgetLock.clear(std::memory_order_release);
        }
    }
    ScopedBudget(const ScopedBudget&) = delete;
    ScopedBudget& operator=(const ScopedBudget&) = delete;
    ScopedBudget(ScopedBudget&& o) noexcept
        : active(o.active), saved(o.saved) { o.active = false; }
    ScopedBudget& operator=(ScopedBudget&& o) noexcept {
        if (this != &o) {
            if (active) { omp_set_num_threads(saved); g_mtBudgetLock.clear(std::memory_order_release); }
            active = o.active; saved = o.saved; o.active = false;
        }
        return *this;
    }
};

/// 规模阈值（GFLOP）：约 2 GFLOP（单核 ~30ms）以上的 GEMM 才值得开多线程。
inline ScopedBudget maybeBudget(double gflops, int k = 8) {
    return ScopedBudget(gflops > 2.0 ? k : 1);
}

} // namespace blas
} // namespace amls
