#pragma once
// DAG 调度器：N 个常驻 worker + 共享就绪队列（mutex/CV）。
// 任务 = fn + 前驱计数 + 后继表；前驱完成时递减后继计数，归零即入队。
// 用于 GE/缩聚统一子任务 DAG（依赖关系拆解表 N1–N10 的任务拆分）：
//   节点就绪 = 直接孩子 N4/N5 完成（updatesDone 哨兵 → N1(parent) 边显式化），
//   节点内 N1(u 回代) → N3(Mtemp) → N4(K 更新) / N5(M 更新) → 哨兵 →
//   N7(局部谱) → N8(μ̃ 种子) → fold(N9 折叠 / N10 传播)。
// 全图可静态预建（结构不依赖运行时），只入队叶子；内点由完成事件推进。
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "linalg/blas.hpp"
#include "linalg/colcomp.hpp"

namespace amls {

struct DagTask {
    template <class F>
    explicit DagTask(F&& f) : fn(std::forward<F>(f)) {}
    DagTask(const DagTask&) = delete;
    DagTask& operator=(const DagTask&) = delete;

    std::function<void()> fn;
    std::atomic<int> remaining{0};   // 未完成前驱数；0 = 可入队
    std::vector<DagTask*> next;      // 后继（前驱完成时逐个递减）
    int64_t prio = 0;                // 就绪队列优先级：小者先出队（默认 FIFO 行为由 seq 保证）
    int64_t seq = 0;                 // 同 prio 的入队序号（稳定 FIFO）
};

class DagRunner {
public:
    explicit DagRunner(int numThreads) {
        if (numThreads <= 0)
            numThreads =
                static_cast<int>(std::thread::hardware_concurrency());
        numThreads = numThreads < 1 ? 1 : numThreads;
        workers_.reserve(static_cast<size_t>(numThreads));
        for (int t = 0; t < numThreads; ++t)
            workers_.emplace_back([this] { workerLoop_(); });
    }
    ~DagRunner() {
        {
            std::lock_guard<std::mutex> lk(qm_);
            stop_ = true;
        }
        qc_.notify_all();
        for (auto& th : workers_)
            th.join();
    }
    DagRunner(const DagRunner&) = delete;
    DagRunner& operator=(const DagRunner&) = delete;

    /// 建任务（不入队）。所有依赖边连好后调用 ready()。
    template <class F>
    DagTask& emplace(F&& f) {
        tasks_.emplace_back(std::forward<F>(f));
        tasks_.back().seq = nextSeq_++;
        return tasks_.back();
    }
    /// 依赖边：pred 完成后 succ 才可运行。
    void depend(DagTask& succ, DagTask& pred) {
        succ.remaining.fetch_add(1, std::memory_order_relaxed);
        pred.next.push_back(&succ);
    }
    /// 图建完后对根任务调用；remaining==0 的立即入队，其余等前驱推进。
    void ready(DagTask& t) {
        if (t.remaining.load(std::memory_order_acquire) == 0)
            pushLocked_(&t);
    }

    // ---- 负载自适应：任务内并行度档位（只升不降）----
    // 判定（在节点 DONE 任务里调用）：就绪队列空 且 活跃 worker ≤2
    // （多数线程无事可做 = 负载不均衡）。连续 3 个节点判定成立 → 升一档。
    // 档位映射任务内并行线程数 {1,2,4,8}。近根段就绪队列常年枯竭 +
    // 右耦合大任务串行，正是该机制的触发区；一旦升档，后续 N4/N5 任务的
    // 主 dgemm 按目标列分组并行（见 colcomp.hpp dgemmCols）。
    static constexpr int kMaxLoadLevel = 3;
    void noteNodeDone() {
        static const bool disabled = std::getenv("AMLS_NO_ADAPT") != nullptr;
        if (disabled) return;
        bool starved;
        {
            std::lock_guard<std::mutex> lk(qm_);
            starved = ready_.empty() && running_ <= 2;
        }
        if (!starved) {
            starveStreak_.store(0, std::memory_order_relaxed);
            return;
        }
        if (starveStreak_.fetch_add(1, std::memory_order_relaxed) + 1 < 3)
            return;
        starveStreak_.store(0, std::memory_order_relaxed);
        int lv = loadLevel_.load(std::memory_order_relaxed);
        while (lv < kMaxLoadLevel) {
            if (loadLevel_.compare_exchange_weak(
                    lv, lv + 1, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                nEscalations_.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    /// 当前任务内并行度（DAG worker 任务入口设置到 opsTlsThreads）。
    int taskParallelism() const {
        static const int kTbl[] = {1, 2, 4, 8};
        return kTbl[std::min(loadLevel_.load(std::memory_order_relaxed),
                             kMaxLoadLevel)];
    }
    /// 等待全部任务完成（worker 保留，可继续建任务/入队）。
    /// 任一任务抛异常：记录首个、跳过其余，此处重抛——否则异常在线程
    /// 里直接 terminate 且无诊断输出（大规模算例排查时踩过）。
    void waitIdle() {
        {
            std::unique_lock<std::mutex> lk(qm_);
            qc_.wait(lk, [&] { return pending_ == 0; });
        }
        if (std::getenv("AMLS_DEBUG_ADAPT") != nullptr) {
            std::fprintf(stderr,
                         "[adapt] 升档次数=%lld 最终档位=%d "
                         "MT预算: 饥饿通过=%lld 获取=%lld 互斥挡=%lld\n",
                         static_cast<long long>(nEscalations_.load()),
                         finalLoadLevel(),
                         static_cast<long long>(
                             blas::ScopedBudget::nHungry.load()),
                         static_cast<long long>(
                             blas::ScopedBudget::nBudgetGranted.load()),
                         static_cast<long long>(
                             blas::ScopedBudget::nBudgetContended.load()));
        }
        if (firstEx_)
            std::rethrow_exception(firstEx_);
    }

private:
    struct ReadyCmp {
        bool operator()(const DagTask* a, const DagTask* b) const {
            if (a->prio != b->prio)
                return a->prio > b->prio;    // 小 prio 先出队
            return a->seq > b->seq;          // 同 prio FIFO
        }
    };
    void pushLocked_(DagTask* t) {
        {
            std::lock_guard<std::mutex> lk(qm_);
            ++pending_;
            ready_.push(t);
        }
        qc_.notify_one();
    }
    void workerLoop_() {
        blas::pinThisThread();   // 每线程 ICV：本线程的 BLAS 并行度钉 1
        for (;;) {
            DagTask* t;
            {
                std::unique_lock<std::mutex> lk(qm_);
                qc_.wait(lk, [&] { return stop_ || !ready_.empty(); });
                if (stop_ && ready_.empty())
                    return;
                t = ready_.top();
                ready_.pop();
                // 空闸：弹出后队列空 且 在跑任务 ≤2（叶子波期间队列常瞬间
                // 空但 12 worker 全忙——那种时刻放开 MT 只会互相踩踏）。
                ++running_;
                blas::tls_starved = ready_.empty() && running_ <= 2;
            }
            if (!failed_.load(std::memory_order_relaxed)) {
                try {
                    opsTlsThreads() = taskParallelism();
                    t->fn();
                    opsTlsThreads() = 1;
                } catch (...) {
                    opsTlsThreads() = 1;
                    if (!failed_.exchange(true))
                        firstEx_ = std::current_exception();
                    // 失败后续任务跳过，但依赖推进继续，否则死等
                }
            }
            for (DagTask* s : t->next) {
                if (s->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    pushLocked_(s);
            }
            {
                std::lock_guard<std::mutex> lk(qm_);
                --running_;
                if (--pending_ == 0)
                    qc_.notify_all();
            }
        }
    }

    std::list<DagTask> tasks_;   // list：插入不移动既有元素，指针稳定
    std::priority_queue<DagTask*, std::vector<DagTask*>, ReadyCmp> ready_;
    std::mutex qm_;
    std::condition_variable qc_;
    std::exception_ptr firstEx_;
    std::atomic<bool> failed_{false};
    int64_t pending_ = 0;
    int64_t running_ = 0;   // 正在执行的任务数（空闸判断用，qm_ 保护）
    int64_t nextSeq_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
    std::atomic<int> starveStreak_{0};   // 连续饥饿节点计数
    std::atomic<int> loadLevel_{0};      // 并行度档位（只升不降，0..3）
public:
    /// 诊断（AMLS_DEBUG_ADAPT）：升档触发次数与最终档位。
    std::atomic<int64_t> nEscalations_{0};
    int finalLoadLevel() const {
        return loadLevel_.load(std::memory_order_relaxed);
    }
private:
};

} // namespace amls
