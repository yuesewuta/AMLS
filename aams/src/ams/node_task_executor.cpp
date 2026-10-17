#include "ams/node_task_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <exception>

#include "linalg/gemm_cols.hpp"

namespace aams {

namespace {

using DenseMatrix = DefaultTraits::DenseMatrix;
using Vector = DefaultTraits::Vector;

constexpr std::size_t kDagTaskCount =
    static_cast<std::size_t>(NodeTaskKind::Count);

constexpr std::size_t idx(NodeTaskKind k) {
    return static_cast<std::size_t>(k);
}

// 入参依赖位表（编译期）：与 PKU kDagDepMask 同构，下标即 NodeTaskKind。
static constexpr std::array<std::uint16_t, kDagTaskCount> kDagDepMask = [] {
    std::array<std::uint16_t, kDagTaskCount> m{};
    auto bit = [](NodeTaskKind k) -> std::uint16_t {
        return static_cast<std::uint16_t>(1) << static_cast<int>(k);
    };
    m[idx(NodeTaskKind::SolveLocalEigenPair)] = 0;
    m[idx(NodeTaskKind::GaussConstraint)] = 0;
    m[idx(NodeTaskKind::GaussKAccum)] = bit(NodeTaskKind::GaussConstraint);
    m[idx(NodeTaskKind::GaussMTemp)] = bit(NodeTaskKind::GaussConstraint);
    m[idx(NodeTaskKind::GaussMAccum)] = bit(NodeTaskKind::GaussMTemp);
    m[idx(NodeTaskKind::CompressLeft)] =
        bit(NodeTaskKind::SolveLocalEigenPair) | bit(NodeTaskKind::GaussMTemp);
    m[idx(NodeTaskKind::CompressUpdate)] = bit(NodeTaskKind::GaussConstraint);
    m[idx(NodeTaskKind::CompressRight)] = bit(NodeTaskKind::SolveLocalEigenPair);
    m[idx(NodeTaskKind::GaussReleaseMtemp)] =
        bit(NodeTaskKind::CompressLeft) | bit(NodeTaskKind::GaussMAccum);
    m[idx(NodeTaskKind::GaussReleaseReducedDiag)] =
        bit(NodeTaskKind::SolveLocalEigenPair) | bit(NodeTaskKind::GaussMAccum);
    m[idx(NodeTaskKind::ReleaseTempBlock)] =
        bit(NodeTaskKind::CompressRight) | bit(NodeTaskKind::CompressUpdate);
    return m;
}();

// 后继位表 = 依赖位表转置（编译期）。
static constexpr std::array<std::uint16_t, kDagTaskCount> kDagSuccMask = [] {
    std::array<std::uint16_t, kDagTaskCount> s{};
    for (std::size_t t = 0; t < kDagTaskCount; ++t)
        for (std::size_t d = 0; d < kDagTaskCount; ++d)
            if ((kDagDepMask[t] >> d) & 1u)
                s[d] = static_cast<std::uint16_t>(s[d] | (1u << t));
    return s;
}();

// 节点内 DAG 上下文（任务间共享，shared_ptr 保活）。
struct DagContext {
    int64_t nodeID = -1;
    ThreadPool* pool = nullptr;
    int prio = 0;   // 调度优先级（深度越小越靠根越优先）
    std::array<std::atomic<int32_t>, kDagTaskCount> pendingDeps{};
    std::atomic<int32_t> inFlight{0};
    std::atomic<int32_t> settled{0};
    std::atomic<int32_t> failure{0};
    std::mutex mtx;
    std::condition_variable cv;
};

// 节点序上下文（全部节点共享）。
struct OrderContext {
    explicit OrderContext(int64_t count)
        : unfinishedChildren(static_cast<size_t>(count)) {}
    ThreadPool* nodePool = nullptr;  // 节点任务（execNodeTask）队列
    ThreadPool* pool = nullptr;      // 节点内 11 任务 DAG 队列（branchPool）
    std::vector<std::atomic<int32_t>> unfinishedChildren;
    std::atomic<int32_t> inFlight{0};
    std::atomic<int32_t> settled{0};
    std::atomic<int32_t> failure{0};
    std::mutex mtx;
    std::condition_variable cv;
};

// 任务幂等进入：Entered → 执行；AlreadyDone → 跳过（返回成功）；其余 → 失败。
bool beginTask(NodeTaskStateStore& states, int64_t nodeID, NodeTaskKind task,
               bool& shouldRun) {
    shouldRun = false;
    switch (states.tryEnter(nodeID, task)) {
        case NodeTaskStateStore::Enter::Entered:
            shouldRun = true;
            return true;
        case NodeTaskStateStore::Enter::AlreadyDone:
            return true;
        case NodeTaskStateStore::Enter::Busy:
            AMLS_LOG_ERROR("节点 {} 任务 {} 状态异常：Busy（调度保证不发生）。",
                           nodeID, static_cast<int>(task));
            return false;
        case NodeTaskStateStore::Enter::Failed:
            return false;  // 全局首失败短路
        default:
            AMLS_LOG_ERROR("节点 {} 任务 {} 进入失败：Invalid。", nodeID,
                           static_cast<int>(task));
            return false;
    }
}

bool finishTask(NodeTaskStateStore& states, int64_t nodeID, NodeTaskKind task,
                bool ok) {
    if (!states.finish(nodeID, task, ok))
        AMLS_LOG_ERROR("节点 {} 任务 {} 收尾状态异常。", nodeID,
                       static_cast<int>(task));
    return ok;
}

} // namespace

// ---- 11 任务入口 ----

bool NodeTaskExecutor::runSolveLocalEigenPair(int64_t nodeID) {
    const auto& nd = tree_.node(nodeID);
    // 非叶：全部孩子的 KAccum/MAccum 必须完成（约化对角定稿）。
    for (int64_t child : nd.children) {
        if (!states_.isDone(child, NodeTaskKind::GaussKAccum) ||
            !states_.isDone(child, NodeTaskKind::GaussMAccum)) {
            AMLS_LOG_ERROR("局部特征求解节点 {} 前置不就绪（孩子 {}）。", nodeID,
                           child);
            return false;
        }
    }
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::SolveLocalEigenPair,
                   shouldRun))
        return false;
    if (!shouldRun) return true;

    bool ok = false;
    DenseMatrix K, M;
    // 就地重装载局部块。dsygvx 会破坏 A/B（B 变 Cholesky 因子、A 变标准化
    // 矩阵）：空区间回退 solveIndex 前必须重取原始块，否则对三角因子再做
    // 分解会得到 info=N+i 的假失败。
    auto loadBlocks = [&]() -> bool {
        if (nd.children.empty()) {
            // 叶子：原始对角子块转稠密（局部行号 = 升序索引位置）。
            K = DenseMatrix(cfg_.blocks->diagK(nodeID));
            M = DenseMatrix(cfg_.blocks->diagM(nodeID));
        } else {
            const DenseMatrix* dk = gauss_.getReducedDiagK(nodeID);
            const DenseMatrix* dm = gauss_.getReducedDiagM(nodeID);
            if (dk == nullptr || dm == nullptr) {
                AMLS_LOG_ERROR("局部特征求解节点 {} 约化对角块不可读。", nodeID);
                return false;
            }
            K = *dk;  // 求解覆盖入参，取副本
            M = *dm;
        }
        return K.rows() > 0 && K.rows() == K.cols() && M.rows() == K.rows();
    };
    try {
        ok = loadBlocks();
        if (ok && std::getenv("AMLS_DUMP_REDUCED_NODES") != nullptr) {
            // 诊断: 转储指定节点的约化对角块（AMLS_DUMP_REDUCED_NODES="656,2532"）
            const std::string want = std::getenv("AMLS_DUMP_REDUCED_NODES");
            if (want.find(std::to_string(nodeID)) != std::string::npos) {
                const std::string tag = std::to_string(nodeID) + "_n" +
                                        std::to_string(K.rows());
                std::ofstream fk("reduced_" + tag + "_K.bin",
                                 std::ios::binary);
                fk.write(reinterpret_cast<const char*>(K.data()),
                         static_cast<std::streamsize>(K.size() * sizeof(double)));
                std::ofstream fm("reduced_" + tag + "_M.bin",
                                 std::ios::binary);
                fm.write(reinterpret_cast<const char*>(M.data()),
                         static_cast<std::streamsize>(M.size() * sizeof(double)));
                std::fprintf(stderr, "[dump] node=%lld n=%lld written K/M bins\n",
                             static_cast<long long>(nodeID),
                             static_cast<long long>(K.rows()));
            }
        }
        if (!ok)
            AMLS_LOG_ERROR("局部特征求解节点 {} 子块形状非法。", nodeID);
        if (ok) {
            Vector w;
            DenseMatrix V;
            const double lambdaHi =
                cfg_.lambdaHiPerNode != nullptr &&
                        static_cast<size_t>(nodeID) <
                            cfg_.lambdaHiPerNode->size()
                    ? (*cfg_.lambdaHiPerNode)[static_cast<size_t>(nodeID)]
                    : cfg_.lambdaHi;
            cfg_.dense->solveInterval(K, M, -1e300, lambdaHi, w, V);
            if (w.size() == 0 && cfg_.minExtractOrder > 0) {
                // 区间空回退：保留最低 minExtractOrder 阶（对齐 PKU
                // allowZeroEigenPair=false 语义）。solveInterval 已就地
                // 破坏 K/M，先重装载再解。
                loadBlocks();
                const int64_t keep =
                    std::min<int64_t>(cfg_.minExtractOrder, K.rows());
                cfg_.dense->solveIndex(K, M, 1, keep, w, V);
            }
            if (std::getenv("AMLS_DEBUG_KEPT"))
                std::fprintf(stderr,
                             "[kept] node=%lld leaf=%d n=%lld kept=%lld "
                             "lmax=%.6g\n",
                             static_cast<long long>(nodeID),
                             nd.children.empty() ? 1 : 0,
                             static_cast<long long>(K.rows()),
                             static_cast<long long>(w.size()),
                             w.size() ? w[w.size() - 1] : -1.0);
            std::vector<double> vals(w.data(), w.data() + w.size());
            ok = saver_.setEigenPair(nodeID, std::move(vals), std::move(V));
            if (!ok)
                AMLS_LOG_ERROR("局部特征求解节点 {} 特征对提交失败。", nodeID);
        }
    } catch (const std::exception& e) {
        {
            int64_t badK = 0, badM = 0;
            for (int64_t i = 0; i < K.size(); ++i) {
                const double kv = K.data()[i], mv = M.data()[i];
                if (!std::isfinite(kv)) ++badK;
                if (!std::isfinite(mv)) ++badM;
            }
            AMLS_LOG_ERROR(
                "局部特征求解节点 {} 异常：{} (n={}, leaf={}, nonfinite K={}, "
                "M={})",
                nodeID, e.what(), K.rows(),
                nd.children.empty() ? 1 : 0, badK, badM);
        }
        ok = false;
    }
    return finishTask(states_, nodeID, NodeTaskKind::SolveLocalEigenPair, ok);
}

bool NodeTaskExecutor::runGaussConstraint(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussConstraint, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = gauss_.computeConstraint(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::GaussConstraint, ok);
}

bool NodeTaskExecutor::runGaussK(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussKAccum, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = gauss_.accumulateReducedStiffness(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::GaussKAccum, ok);
}

bool NodeTaskExecutor::runGaussMtemp(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussMTemp, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = gauss_.computeMtemp(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::GaussMTemp, ok);
}

bool NodeTaskExecutor::runGaussM(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussMAccum, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = gauss_.accumulateReducedMass(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::GaussMAccum, ok);
}

bool NodeTaskExecutor::runGaussReleaseMtemp(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussReleaseMtemp, shouldRun))
        return false;
    if (!shouldRun) return true;
    bool ok = true;
    for (int64_t a : tree_.ancestors(nodeID))
        gauss_.releaseMTemp(nodeID, a);
    return finishTask(states_, nodeID, NodeTaskKind::GaussReleaseMtemp, ok);
}

bool NodeTaskExecutor::runGaussReleaseReducedDiag(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::GaussReleaseReducedDiag,
                   shouldRun))
        return false;
    if (!shouldRun) return true;
    gauss_.releaseReducedDiagK(nodeID);
    gauss_.releaseReducedDiagM(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::GaussReleaseReducedDiag,
                      true);
}

bool NodeTaskExecutor::runCompressLeft(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::CompressLeft, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = temp_.create(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::CompressLeft, ok);
}

bool NodeTaskExecutor::runCompressUpdate(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::CompressUpdate, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = temp_.update(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::CompressUpdate, ok);
}

bool NodeTaskExecutor::runCompressRight(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::CompressRight, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = temp_.rightMultiply(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::CompressRight, ok);
}

bool NodeTaskExecutor::runCompressRelease(int64_t nodeID) {
    bool shouldRun = false;
    if (!beginTask(states_, nodeID, NodeTaskKind::ReleaseTempBlock, shouldRun))
        return false;
    if (!shouldRun) return true;
    const bool ok = temp_.release(nodeID);
    return finishTask(states_, nodeID, NodeTaskKind::ReleaseTempBlock, ok);
}

// ---- 调度 ----

void NodeTaskExecutor::execDagTask(const std::shared_ptr<void>& ctxVoid,
                                   std::size_t taskIdx) {
    auto* dag = static_cast<DagContext*>(ctxVoid.get());
    using Runner = bool (NodeTaskExecutor::*)(int64_t);
    static constexpr Runner kRunners[] = {
        &NodeTaskExecutor::runSolveLocalEigenPair,
        &NodeTaskExecutor::runGaussConstraint,
        &NodeTaskExecutor::runGaussK,
        &NodeTaskExecutor::runGaussMtemp,
        &NodeTaskExecutor::runGaussM,
        &NodeTaskExecutor::runGaussReleaseMtemp,
        &NodeTaskExecutor::runGaussReleaseReducedDiag,
        &NodeTaskExecutor::runCompressLeft,
        &NodeTaskExecutor::runCompressUpdate,
        &NodeTaskExecutor::runCompressRight,
        &NodeTaskExecutor::runCompressRelease,
    };
    static_assert(sizeof(kRunners) / sizeof(kRunners[0]) == kDagTaskCount);

    // 刷新本 worker 的 dgemm 借线程预算：池并发 ÷ 在飞任务数（含排队
    // 即将启动的）。关键链尾部独占机器时在飞=1，可借满整池——近根
    // CompressUpdate 的批量 dgemm 占链时长的大头。
    {
        const int cap = dag->pool ? dag->pool->size() : 1;
        const int run =
            std::max(1, dag->inFlight.load(std::memory_order_relaxed));
        opsTlsThreads() = std::max(1, std::min(cap, cap / run));
    }

    bool ok = false;
    const auto tStart = std::chrono::steady_clock::now();
    if (timelineOn_) {
        const std::size_t slot =
            static_cast<std::size_t>(dag->nodeID) * kDagTaskCount + taskIdx;
        taskStartNs_[slot].store(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tStart.time_since_epoch())
                    .count()) -
                t0Ns_,
            std::memory_order_relaxed);
    }
    try {
        ok = (this->*kRunners[taskIdx])(dag->nodeID);
    } catch (const std::exception& e) {
        AMLS_LOG_ERROR("节点 {} 任务 {} 未捕获异常：{}", dag->nodeID, taskIdx,
                       e.what());
        ok = false;
    }
    const auto tEnd = std::chrono::steady_clock::now();
    if (timelineOn_) {
        const std::size_t slot =
            static_cast<std::size_t>(dag->nodeID) * kDagTaskCount + taskIdx;
        taskEndNs_[slot].store(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    tEnd.time_since_epoch())
                    .count()) -
                t0Ns_,
            std::memory_order_relaxed);
    }
    taskNanos_[taskIdx].fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tEnd - tStart)
                .count()),
        std::memory_order_relaxed);
    taskCounts_[taskIdx].fetch_add(1, std::memory_order_relaxed);
    if (!ok) dag->failure.store(1, std::memory_order_release);

    // 成功才级联后继；先增 inFlight 再入队，保证等待谓词不丢唤醒。
    // fetch_sub 返回 1 的线程恰好是清零该依赖的那一个，因此无重复入队。
    if (ok) {
        const std::uint16_t succ = kDagSuccMask[taskIdx];
        std::vector<std::size_t> ready;
        for (std::size_t s = 0; s < kDagTaskCount; ++s) {
            if (((succ >> s) & 1u) == 0) continue;
            if (dag->pendingDeps[s].fetch_sub(
                    1, std::memory_order_acq_rel) == 1)
                ready.push_back(s);
        }
        if (!ready.empty()) {
            dag->inFlight.fetch_add(static_cast<int32_t>(ready.size()),
                                    std::memory_order_acq_rel);
            for (std::size_t s : ready)
                dag->pool->enqueue([this, ctxVoid, s](int) {
                    execDagTask(ctxVoid, s);
                }, dag->prio);
        }
    }

    {
        std::lock_guard<std::mutex> lock(dag->mtx);
        dag->inFlight.fetch_sub(1, std::memory_order_acq_rel);
        dag->settled.fetch_add(1, std::memory_order_acq_rel);
    }
    dag->cv.notify_all();
}

bool NodeTaskExecutor::runNodeParallelDag(int64_t nodeID,
                                          ThreadPool& branchPool) {
    auto ctx = std::make_shared<DagContext>();
    ctx->nodeID = nodeID;
    ctx->pool = &branchPool;
    ctx->prio = depthAt_(nodeID);
    for (std::size_t t = 0; t < kDagTaskCount; ++t) {
        const auto mask = kDagDepMask[t];
        ctx->pendingDeps[t] = static_cast<int32_t>(__builtin_popcount(mask));
    }
    // 无前置任务：SolveLocalEigenPair 与 GaussConstraint。
    ctx->inFlight.store(2, std::memory_order_release);
    ctx->pool->enqueue([this, ctx](int) {
        execDagTask(ctx, idx(NodeTaskKind::SolveLocalEigenPair));
    }, ctx->prio);
    ctx->pool->enqueue([this, ctx](int) {
        execDagTask(ctx, idx(NodeTaskKind::GaussConstraint));
    }, ctx->prio);
    std::unique_lock<std::mutex> lock(ctx->mtx);
    ctx->cv.wait(lock, [&ctx]() {
        return ctx->inFlight.load(std::memory_order_acquire) == 0 &&
               (ctx->settled.load(std::memory_order_acquire) ==
                    static_cast<int32_t>(kDagTaskCount) ||
                ctx->failure.load(std::memory_order_acquire) != 0);
    });
    return ctx->failure.load(std::memory_order_acquire) == 0;
}

void NodeTaskExecutor::execNodeTask(const std::shared_ptr<void>& ctxVoid,
                                    int64_t nodeID) {
    auto* order = static_cast<OrderContext*>(ctxVoid.get());
    bool ok = false;
    try {
        ok = runNodeParallelDag(nodeID, *order->pool);
        if (!ok)
            AMLS_LOG_ERROR("节点 {} 11 任务 DAG 失败。", nodeID);
    } catch (const std::exception& e) {
        AMLS_LOG_ERROR("节点 {} 任务调度异常：{}", nodeID, e.what());
        ok = false;
    }
    completedNodes_.fetch_add(1, std::memory_order_relaxed);
    if (!ok) order->failure.store(1, std::memory_order_release);

    // 节点级级联：全部孩子节点完成后入队父节点。
    const int64_t parent = tree_.node(nodeID).parent;
    if (ok && parent >= 0) {
        if (order->unfinishedChildren[static_cast<size_t>(parent)].fetch_sub(
                1, std::memory_order_acq_rel) == 1) {
            order->inFlight.fetch_add(1, std::memory_order_acq_rel);
            order->nodePool->enqueue([this, ctxVoid, parent](int) {
                execNodeTask(ctxVoid, parent);
            }, depthAt_(parent));
        }
    }

    {
        std::lock_guard<std::mutex> lock(order->mtx);
        order->inFlight.fetch_sub(1, std::memory_order_acq_rel);
        order->settled.fetch_add(1, std::memory_order_acq_rel);
    }
    order->cv.notify_all();
}

bool NodeTaskExecutor::runNodeOrder(ThreadPool& nodePool,
                                    ThreadPool& branchPool) {
    const int64_t count = tree_.nNodes();
    if (count <= 0) {
        AMLS_LOG_ERROR("runNodeOrder 失败：空树。");
        return false;
    }
    if (timelineOn_)
        t0Ns_ = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    auto ctx = std::make_shared<OrderContext>(count);
    ctx->nodePool = &nodePool;
    ctx->pool = &branchPool;
    for (int64_t i = 0; i < count; ++i)
        ctx->unfinishedChildren[static_cast<size_t>(i)] =
            static_cast<int32_t>(tree_.node(i).children.size());
    buildDepths_();

    // 叶子先入队；父节点由 execNodeTask 级联。
    int64_t leaves = 0;
    for (int64_t i = 0; i < count; ++i) {
        if (!tree_.node(i).children.empty()) continue;
        ctx->inFlight.fetch_add(1, std::memory_order_acq_rel);
        ++leaves;
        nodePool.enqueue([this, ctx, i](int) { execNodeTask(ctx, i); },
                         depthAt_(i));
    }
    if (leaves == 0) {
        AMLS_LOG_ERROR("runNodeOrder 失败：无叶子节点。");
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(ctx->mtx);
        ctx->cv.wait(lock, [&ctx, count]() {
            return ctx->inFlight.load(std::memory_order_acquire) == 0 &&
                   (ctx->settled.load(std::memory_order_acquire) ==
                        static_cast<int32_t>(count) ||
                    ctx->failure.load(std::memory_order_acquire) != 0);
        });
    }
    if (ctx->failure.load(std::memory_order_acquire) != 0) {
        AMLS_LOG_ERROR("runNodeOrder 失败：{} / {} 节点完成。",
                       completedNodes_.load(std::memory_order_relaxed), count);
        return false;
    }

    // 兜底整树释放（根 release 幂等；成功路径通常已在根 CompressRelease
    // 触发过）。
    if (!temp_.release(tree_.root()))
        AMLS_LOG_WARN("根节点临时块释放异常（可能已释放）。");
    return true;
}

std::vector<double> NodeTaskExecutor::profileTaskSeconds() const {
    std::vector<double> out(taskNanos_.size(), 0.0);
    for (std::size_t i = 0; i < taskNanos_.size(); ++i)
        out[i] = static_cast<double>(
                     taskNanos_[i].load(std::memory_order_relaxed)) * 1e-9;
    return out;
}

std::vector<int64_t> NodeTaskExecutor::profileTaskCounts() const {
    std::vector<int64_t> out(taskCounts_.size(), 0);
    for (std::size_t i = 0; i < taskCounts_.size(); ++i)
        out[i] = taskCounts_[i].load(std::memory_order_relaxed);
    return out;
}

void NodeTaskExecutor::buildDepths_() {
    const int64_t n = tree_.nNodes();
    depth_.assign(static_cast<size_t>(n), -1);
    for (int64_t i = 0; i < n; ++i) {
        if (depth_[static_cast<size_t>(i)] >= 0) continue;
        // 沿父链上溯到已知深度，再回填整条路径
        std::vector<int64_t> path;
        int64_t cur = i;
        while (cur >= 0 && depth_[static_cast<size_t>(cur)] < 0) {
            path.push_back(cur);
            cur = tree_.node(cur).parent;
        }
        int d = cur >= 0 ? depth_[static_cast<size_t>(cur)] + 1 : 0;
        for (auto it = path.rbegin(); it != path.rend(); ++it)
            depth_[static_cast<size_t>(*it)] = d++;
    }
}

void NodeTaskExecutor::dumpTimeline(const std::string& path) const {
    if (!timelineOn_) return;
    std::ofstream f(path);
    if (!f) {
        AMLS_LOG_WARN("时间线转储无法写入：{}", path);
        return;
    }
    f << "node,task,start_ns,end_ns\n";
    const int64_t n = tree_.nNodes();
    for (int64_t node = 0; node < n; ++node)
        for (std::size_t t = 0; t < kDagTaskCount; ++t) {
            const std::size_t slot =
                static_cast<std::size_t>(node) * kDagTaskCount + t;
            const std::uint64_t s =
                taskStartNs_[slot].load(std::memory_order_relaxed);
            const std::uint64_t e =
                taskEndNs_[slot].load(std::memory_order_relaxed);
            if (e > s) f << node << ',' << t << ',' << s << ',' << e << '\n';
        }
    // 树边：task=-1 行，start_ns 存 parent（供关键链离线分析）
    for (int64_t node = 0; node < n; ++node)
        f << node << ",-1,"
          << tree_.node(node).parent << ",0\n";
}

} // namespace aams
