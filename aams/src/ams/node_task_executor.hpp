#pragma once
// 节点任务执行器 —— 移植自 third_party/amlsPKU utils/amls_task_executor +
// node_task_state_store（NODE 路径：双线程池节点序调度 + 节点内 11 任务
// 静态依赖位表 DAG）。
//
// 调度语义（与 PKU 一致）：
//   - 节点级（nodePool，round(T/1.5) 线程）：叶子先入队；节点全部 11 任务
//     完成后按 unfinishedChildren 计数级联入队父节点。节点级串行保证
//     update/left 块终态、CompressLeft 读 M_temp 终态等跨节点依赖。
//   - 节点内（branchPool，T 线程）：11 任务统一入池，按 kDagDepMask 依赖
//     位表级联入队，无阶段屏障：
//       SolveLocalEigenPair / GaussConstraint 无前置；
//       GaussMTemp / GaussKAccum / CompressUpdate ← GaussConstraint；
//       GaussMAccum ← GaussMTemp；
//       CompressLeft ← SolveLocalEigenPair + GaussMTemp；
//       CompressRight ← SolveLocalEigenPair；
//       GaussReleaseMtemp ← CompressLeft + GaussMAccum；
//       GaussReleaseReducedDiag ← SolveLocalEigenPair + GaussMAccum；
//       CompressRelease ← CompressRight + CompressUpdate。
//   - NodeTaskStateStore 每任务独立原子状态（Unknown/Running/Done/Failed）+
//     全局 firstFailure 短路；任务幂等（AlreadyDone 不重跑）。
//   - 局部特征求解：叶子读 OriginalBlocks 原始对角块转稠密，非叶读约化
//     对角块副本；统一稠密区间 dsygvx（-∞, λ_hi]，λ_hi=(2π·γ·maxFreq)²，
//     空区间得 0 模态节点（allowZero 由 saver 承担）。
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "ams/eigen_pair_saver.hpp"
#include "ams/gauss_module.hpp"
#include "ams/original_blocks.hpp"
#include "ams/separator_tree.hpp"
#include "ams/temp_block_manager.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "linalg/blas.hpp"
#include "linalg/dense_eigensolver.hpp"

namespace aams {

/// 固定 worker 线程池（优先级队列；任务签名 void(int threadId)）。
/// enqueue 的 priority 越小越先执行（0 = 最高）：REDUCTION 关键链靠近
/// 根的节点若按 FIFO 排在叶洪流之后，关键路径会被排队延迟主导
/// （cant_20 时间线实测：链上小节点任务共 ~4ms 却贡献 ~140ms）。
class ThreadPool {
public:
    explicit ThreadPool(int numThreads) {
        for (int t = 0; t < numThreads; ++t)
            workers_.emplace_back([this, t]() {
                blas::pinThisThread();  // BLAS 并行度钉 1（BLAS 全局单线程，
                                        // 见 solver 编排；钉线程防库内逃逸）
                for (;;) {
                    std::function<void(int)> task;
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        cv_.wait(lock, [this]() {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(
                            const_cast<PrioritizedTask&>(tasks_.top()).task);
                        tasks_.pop();
                    }
                    task(t);
                }
            });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void enqueue(std::function<void(int)> f, int priority = 0) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            tasks_.push({std::move(f), priority, ++seq_});
        }
        cv_.notify_one();
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    struct PrioritizedTask {
        std::function<void(int)> task;
        int priority = 0;
        std::uint64_t seq = 0;
    };
    struct EarlierFirst {
        bool operator()(const PrioritizedTask& a,
                        const PrioritizedTask& b) const {
            if (a.priority != b.priority) return a.priority > b.priority;
            return a.seq > b.seq;
        }
    };
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::priority_queue<PrioritizedTask, std::vector<PrioritizedTask>,
                        EarlierFirst>
        tasks_;
    std::uint64_t seq_ = 0;
    bool stop_ = false;
};

/// 11 任务/节点的静态依赖位表 DAG。
enum class NodeTaskKind : std::uint8_t {
    SolveLocalEigenPair = 0,
    GaussConstraint,
    GaussKAccum,
    GaussMTemp,
    GaussMAccum,
    GaussReleaseMtemp,
    GaussReleaseReducedDiag,
    CompressLeft,
    CompressUpdate,
    CompressRight,
    ReleaseTempBlock,
    Count,
};

/// 按连续 nodeID 的任务状态存储：每任务独立原子状态 + 全局首失败短路。
class NodeTaskStateStore {
public:
    enum class Enter : std::uint8_t {
        Entered,
        AlreadyDone,
        Busy,
        Failed,
        Invalid,
    };

    explicit NodeTaskStateStore(int64_t nodeCount) {
        slots_.reserve(static_cast<size_t>(nodeCount));
        for (int64_t i = 0; i < nodeCount; ++i)
            slots_.emplace_back(std::make_unique<NodeState>());
    }

    Enter tryEnter(int64_t nodeID, NodeTaskKind task) {
        NodeState* s = find_(nodeID);
        if (s == nullptr) return Enter::Invalid;
        int32_t expected = kUnknown;
        if (s->tasks[static_cast<size_t>(task)].compare_exchange_strong(
                expected, kRunning, std::memory_order_acq_rel,
                std::memory_order_acquire))
            return Enter::Entered;
        if (expected == kDone) return Enter::AlreadyDone;
        if (expected == kRunning) return Enter::Busy;
        if (expected == kFailed) return Enter::Failed;
        return Enter::Invalid;
    }

    /// 任务收尾：CAS Running→Done/Failed；失败发布全局首失败标志。
    bool finish(int64_t nodeID, NodeTaskKind task, bool ok) {
        NodeState* s = find_(nodeID);
        if (s == nullptr) return false;
        int32_t expected = kRunning;
        const int32_t target = ok ? kDone : kFailed;
        if (!s->tasks[static_cast<size_t>(task)].compare_exchange_strong(
                expected, target, std::memory_order_acq_rel,
                std::memory_order_acquire))
            return false;
        if (!ok) firstFailure_.store(1, std::memory_order_release);
        return true;
    }

    bool isDone(int64_t nodeID, NodeTaskKind task) const {
        const NodeState* s = find_(nodeID);
        return s != nullptr &&
               s->tasks[static_cast<size_t>(task)].load(
                   std::memory_order_acquire) == kDone;
    }

    bool failed() const {
        return firstFailure_.load(std::memory_order_acquire) != 0;
    }

private:
    static constexpr int32_t kUnknown = 0;
    static constexpr int32_t kRunning = 1;
    static constexpr int32_t kDone = 2;
    static constexpr int32_t kFailed = 3;

    struct NodeState {
        std::array<std::atomic<int32_t>,
                   static_cast<size_t>(NodeTaskKind::Count)>
            tasks{};
    };
    const NodeState* find_(int64_t nodeID) const {
        if (nodeID < 0 || nodeID >= static_cast<int64_t>(slots_.size()))
            return nullptr;
        return slots_[static_cast<size_t>(nodeID)].get();
    }
    NodeState* find_(int64_t nodeID) {
        if (nodeID < 0 || nodeID >= static_cast<int64_t>(slots_.size()))
            return nullptr;
        return slots_[static_cast<size_t>(nodeID)].get();
    }

    std::vector<std::unique_ptr<NodeState>> slots_;
    std::atomic<int32_t> firstFailure_{0};
};

class NodeTaskExecutor {
public:
    /// 局部特征求解配置（统一稠密区间路径）。
    struct LocalEigenConfig {
        IDenseEigensolver<DefaultTraits>* dense = nullptr;  // 区间求解内核
        const OriginalBlocks* blocks = nullptr;             // 叶子原始子块
        double lambdaHi = 0.0;  // 局部截断上界 (2π·γ·maxFreq)²；区间 (-∞, λ_hi]
        // 逐节点截断上界（ApplyClassicCutoffs/ApplyEnhancedCutoffs 的分发
        // 语义，pme_core_pseudocode.txt:44632+：主截断=ω1²，根层=ωlast²，
        // 增强标记=ωenh²；下标=节点 id）。非空时优先于 lambdaHi。
        const std::vector<double>* lambdaHiPerNode = nullptr;
        // 区间空时回退保留的最低阶数（PKU allowZeroEigenPair=false 的默认
        // 行为）：分隔节点约化对角谱整体高于 λ_hi 时至少保留 1 阶，否则其
        // 质量耦合链（μ 经分隔节点传递）断裂，子空间出现成对伪模态。
        int64_t minExtractOrder = 1;
    };

    NodeTaskExecutor(const SeparatorTree& tree, GaussEliminationModule& gauss,
                     TempBlockManager& temp, EigenPairSaver& saver,
                     NodeTaskStateStore& states,
                     const LocalEigenConfig& eigenCfg)
        : tree_(tree),
          gauss_(gauss),
          temp_(temp),
          saver_(saver),
          states_(states),
          cfg_(eigenCfg) {
        if (eigenCfg.dense == nullptr || eigenCfg.blocks == nullptr)
            throw std::invalid_argument(
                "NodeTaskExecutor 需要 dense 求解器与 OriginalBlocks");
        taskNanos_ = std::vector<std::atomic<std::uint64_t>>(
            static_cast<std::size_t>(NodeTaskKind::Count));
        taskCounts_ = std::vector<std::atomic<std::int64_t>>(
            static_cast<std::size_t>(NodeTaskKind::Count));
        timelineOn_ = std::getenv("AAMS_TASK_TIMELINE") != nullptr;
        if (timelineOn_) {
            taskStartNs_ = std::vector<std::atomic<std::uint64_t>>(
                static_cast<std::size_t>(tree_.nNodes()) *
                static_cast<std::size_t>(NodeTaskKind::Count));
            taskEndNs_ = std::vector<std::atomic<std::uint64_t>>(
                static_cast<std::size_t>(tree_.nNodes()) *
                static_cast<std::size_t>(NodeTaskKind::Count));
        }
    }

    /// 节点序执行全部节点（叶子先入队、父节点级联）；失败返回 false
    ///（原因见日志）。结束时不做根释放（调用方兜底，幂等）。
    bool runNodeOrder(ThreadPool& nodePool, ThreadPool& branchPool);

    int64_t completedNodes() const {
        return completedNodes_.load(std::memory_order_relaxed);
    }

    /// 性能剖析（AAMS_PROFILE=1）：11 类 DAG 任务的累计执行时间（秒）
    /// 与调用次数，下标即 NodeTaskKind。
    std::vector<double> profileTaskSeconds() const;
    std::vector<int64_t> profileTaskCounts() const;

    /// 时间线转储（AAMS_TASK_TIMELINE=1 时记录）：CSV "node,taskIdx,
    /// startNs,endNs"，相对 runNodeOrder 起点，用于关键路径离线分析。
    void dumpTimeline(const std::string& path) const;

private:
    // 11 任务入口（幂等；失败经 states_.firstFailure 短路）。
    bool runSolveLocalEigenPair(int64_t nodeID);
    bool runGaussConstraint(int64_t nodeID);
    bool runGaussK(int64_t nodeID);
    bool runGaussMtemp(int64_t nodeID);
    bool runGaussM(int64_t nodeID);
    bool runGaussReleaseMtemp(int64_t nodeID);
    bool runGaussReleaseReducedDiag(int64_t nodeID);
    bool runCompressLeft(int64_t nodeID);
    bool runCompressUpdate(int64_t nodeID);
    bool runCompressRight(int64_t nodeID);
    bool runCompressRelease(int64_t nodeID);

    bool runNodeParallelDag(int64_t nodeID, ThreadPool& branchPool);
    void execDagTask(const std::shared_ptr<void>& ctx, std::size_t taskIdx);
    void execNodeTask(const std::shared_ptr<void>& ctx, int64_t nodeID);

    /// 节点深度（根=0，叶子最深）缓存；小值=靠根=高调度优先级。
    void buildDepths_();
    int depthAt_(int64_t nodeID) const {
        return depth_[static_cast<size_t>(nodeID)];
    }
    std::vector<int> depth_;

    const SeparatorTree& tree_;
    GaussEliminationModule& gauss_;
    TempBlockManager& temp_;
    EigenPairSaver& saver_;
    NodeTaskStateStore& states_;
    LocalEigenConfig cfg_;
    std::atomic<int64_t> completedNodes_{0};
    std::vector<std::atomic<std::uint64_t>> taskNanos_;
    std::vector<std::atomic<std::int64_t>> taskCounts_;
    // AAMS_TASK_TIMELINE=1：每(节点,任务) start/end（ns，相对 t0Ns_）
    bool timelineOn_ = false;
    std::uint64_t t0Ns_ = 0;
    std::vector<std::atomic<std::uint64_t>> taskStartNs_;
    std::vector<std::atomic<std::uint64_t>> taskEndNs_;
};

} // namespace aams
