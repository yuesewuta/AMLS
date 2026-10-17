#include "ams/temp_block_manager.hpp"

#include <algorithm>
#include <atomic>

#include "linalg/gemm_cols.hpp"

namespace aams {

bool TempBlockManager::initialize(const SeparatorTree& tree,
                                  GaussEliminationModule* gauss,
                                  const EigenPairSaver* eigenPairSaver,
                                  SubspaceMassSaver* massSaver) {
    if (gauss == nullptr || eigenPairSaver == nullptr || massSaver == nullptr ||
        !gauss->initialized() ||
        eigenPairSaver->nodeCount() != tree.nNodes()) {
        AMLS_LOG_ERROR("TempBlockManager 初始化失败：依赖模块缺失或节点数不符。");
        return false;
    }
    tree_ = &tree;
    gauss_ = gauss;
    eigenPairSaver_ = eigenPairSaver;
    massSaver_ = massSaver;
    useGemv_ = useColAccGemv();
    N_ = tree.nNodes();

    blocks_.clear();
    nodeReleaseBlocks_.assign(static_cast<size_t>(N_), {});
    gates_.clear();
    doneFlags_.clear();
    gates_.reserve(static_cast<size_t>(N_));
    doneFlags_.reserve(static_cast<size_t>(N_));
    for (int64_t i = 0; i < N_; ++i) {
        gates_.emplace_back(std::make_unique<NodeGate>());
        doneFlags_.emplace_back(std::make_unique<std::atomic<uint8_t>>(0));
    }
    for (int64_t i = 0; i < N_; ++i) {
        for (int64_t a : tree.ancestors(i)) {
            auto state = std::make_shared<BlockState>(Key{i, a});
            blocks_.emplace(Key{i, a}, state);
            nodeReleaseBlocks_[static_cast<size_t>(a)].push_back(
                std::move(state));
        }
    }
    // 批量加锁沿用相同 key 序，避免 update 与 release 死锁
    for (auto& states : nodeReleaseBlocks_)
        std::sort(states.begin(), states.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs->key < rhs->key;
                  });
    rootRelease_.store(RootRelease::Idle, std::memory_order_release);
    return true;
}

TempBlockManager::BlockState* TempBlockManager::findBlock_(
    const Key& key) const {
    const auto it = blocks_.find(key);
    return it == blocks_.end() ? nullptr : it->second.get();
}

TempBlockManager::NodeGate* TempBlockManager::gate_(int64_t nodeID) const {
    if (nodeID < 0 || nodeID >= N_) return nullptr;
    return gates_[static_cast<size_t>(nodeID)].get();
}

bool TempBlockManager::enterTask_(int64_t nodeID, bool isCreate) {
    if (!initialized()) {
        AMLS_LOG_ERROR("TempBlockManager::enterTask 失败：未初始化。");
        return false;
    }
    NodeGate* gate = gate_(nodeID);
    if (gate == nullptr) {
        AMLS_LOG_ERROR("TempBlockManager::enterTask 失败：节点 {} 越界。", nodeID);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(gate->mtx);
        // create 只写 (nodeID,·) 块，与本节点 release 的 (·,nodeID) 块不冲突
        if (!isCreate && (gate->releaseRequests > 0 || gate->released)) {
            AMLS_LOG_DEBUG(
                "TempBlockManager::enterTask 节点 {} 任务被释放请求拒绝。", nodeID);
            return false;
        }
        ++gate->activeCalls;
    }
    return true;
}

void TempBlockManager::leaveTask_(int64_t nodeID) {
    NodeGate* gate = gate_(nodeID);
    if (gate == nullptr) return;
    std::lock_guard<std::mutex> lock(gate->mtx);
    if (gate->activeCalls > 0) --gate->activeCalls;
    gate->cv.notify_all();
}

bool TempBlockManager::createColComp_(int64_t nodeID) {
    const std::vector<int64_t> ancestors = tree_->ancestors(nodeID);
    EigenPairSaver::View view;
    if (!eigenPairSaver_->getEigenPairView(nodeID, view)) {
        AMLS_LOG_DEBUG("create({}) 未就绪：特征对缺失。", nodeID);
        return false;
    }
    const DefaultTraits::DenseMatrix* phi = view.eigenvectors();
    if (phi == nullptr || phi->rows() < 0 || phi->cols() < 0) {
        AMLS_LOG_ERROR("create({}) 失败：特征对维度非法。", nodeID);
        return false;
    }
    const bool zeroModes = view.eigenvalues()->empty();
    const int64_t modeCount = phi->cols();

    for (int64_t j : ancestors) {
        BlockState* target = findBlock_(Key{nodeID, j});
        if (target == nullptr) {
            AMLS_LOG_ERROR("create({}) 失败：块 ({},{}) 缺失。", nodeID, nodeID, j);
            return false;
        }
        const ColCompMat* mtemp = gauss_->getMTemp(nodeID, j);
        if (mtemp == nullptr || !mtemp->numericReady()) {
            AMLS_LOG_DEBUG("create({}) 未就绪：M_temp ({},{}) 缺失。", nodeID,
                           nodeID, j);
            return false;
        }
        if (phi->rows() != mtemp->rows()) {
            AMLS_LOG_ERROR(
                "create({}) 失败：Φ 行数 {} 与 M_temp ({},{}) 行数 {} 不符。",
                nodeID, phi->rows(), nodeID, j, mtemp->rows());
            return false;
        }
        const int64_t updateTimes = depth_(nodeID) - depth_(j) - 1;

        // μ 列规划：active(M_temp) ∪ K̃(x,j) 预留列，x = j 通往 nodeID 的
        // 直接孩子（预留列定理：后续 update 经 Ψ_{x,j} 写入需要这些列）
        const std::vector<int32_t>* reserved = nullptr;
        if (!zeroModes && updateTimes > 0) {
            int64_t x = nodeID;
            int64_t parent = tree_->node(x).parent;
            while (parent >= 0 && parent != j) {
                x = parent;
                parent = tree_->node(x).parent;
            }
            if (parent != j) {
                AMLS_LOG_ERROR(
                    "create({}) 失败：祖先 {} 不在父链上。", nodeID, j);
                return false;
            }
            reserved = gauss_->getReservedColState(x, j);
            if (reserved == nullptr) {
                AMLS_LOG_DEBUG("create({}) 未就绪：预留列 ({},{}) 缺失。", nodeID,
                               x, j);
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(target->mtx);
        if (target->lifecycle.load(std::memory_order_relaxed) !=
            Lifecycle::Empty) {
            AMLS_LOG_ERROR("create({}) 失败：块 ({},{}) 已创建。", nodeID, nodeID,
                           j);
            return false;
        }
        if (zeroModes) {
            // 0 模态块只保留 0 × n_j 逻辑形状，不规划存储列
            ColCompMat zeroBlock(0, mtemp->totalCols());
            if (!zeroBlock.setLayout({}) || !zeroBlock.initializeZero()) {
                AMLS_LOG_ERROR("create({}) 失败：0 模态块 ({},{}) 初始化被拒。",
                               nodeID, nodeID, j);
                return false;
            }
            target->value = std::move(zeroBlock);
        } else if (!target->value.initializeMuFromInputs(modeCount, *mtemp,
                                                         reserved)) {
            AMLS_LOG_ERROR(
                "create({}) 失败：μ ({},{}) 布局初始化被拒 (模态数 {})。", nodeID,
                nodeID, j, modeCount);
            return false;
        } else if (!opsColAcc(useGemv_, *phi, true, *mtemp, target->value)) {
            AMLS_LOG_ERROR("create({}) 失败：μ ({},{}) = Φᵀ·M_temp 维度不匹配。",
                           nodeID, nodeID, j);
            return false;
        }
        target->remainingUpdates.store(updateTimes, std::memory_order_relaxed);
        target->lifecycle.store(Lifecycle::Active, std::memory_order_release);
    }
    return true;
}

bool TempBlockManager::updateColComp_(int64_t nodeID) {
    const std::vector<int64_t> ancestors = tree_->ancestors(nodeID);
    const std::vector<int64_t> descendants = tree_->descendants(nodeID);
    if (ancestors.empty() || descendants.empty()) return true;

    // 左块 (d,nodeID) 全部预检通过前不做写入，保持 NOT_READY 全或无语义
    std::vector<BlockState*> leftStates;
    leftStates.reserve(descendants.size());
    bool needsNumeric = false;
    for (int64_t d : descendants) {
        BlockState* left = findBlock_(Key{d, nodeID});
        if (left == nullptr) {
            AMLS_LOG_DEBUG("update({}) 未就绪：左块 ({},{}) 缺失。", nodeID, d,
                           nodeID);
            return false;
        }
        if (left->lifecycle.load(std::memory_order_acquire) !=
                Lifecycle::Active ||
            left->remainingUpdates.load(std::memory_order_acquire) != 0) {
            AMLS_LOG_DEBUG(
                "update({}) 未就绪：左块 ({},{}) 剩余更新未清零。", nodeID, d,
                nodeID);
            return false;
        }
        if (!left->value.numericReady()) {
            AMLS_LOG_ERROR("update({}) 失败：左块 ({},{}) 数值未就绪。", nodeID, d,
                           nodeID);
            return false;
        }
        needsNumeric = needsNumeric || left->value.rows() > 0;
        leftStates.push_back(left);
    }

    // 只有存在非零行左块才读 Ψ；全零行时仅完成 remainingUpdates 递减
    std::vector<const ColCompMat*> psis(static_cast<size_t>(ancestors.size()),
                                        nullptr);
    if (needsNumeric) {
        for (size_t a = 0; a < ancestors.size(); ++a) {
            psis[a] = gauss_->getPsi(nodeID, ancestors[static_cast<size_t>(a)]);
            if (psis[a] == nullptr || !psis[a]->numericReady()) {
                AMLS_LOG_DEBUG("update({}) 未就绪：Ψ ({},{}) 缺失。", nodeID,
                               nodeID, ancestors[static_cast<size_t>(a)]);
                return false;
            }
        }
    }

    const size_t nd = descendants.size();
    static const bool cupdateLog = std::getenv("AAMS_CUPDATE_LOG") != nullptr;
    const auto cuT0 = std::chrono::steady_clock::now();
    // 单个 worker 段（d 按 nthr 取模分配）：0 行 d 即时倒计数；数值 d
    // 预检并收集左块/目标块，段末单次堆叠批量更新（同活跃列签名的 d
    // 共享一次 C_all gather，行堆叠摊薄 OpenBLAS 每次 ~0.9ms 的 gemm
    // 固定开销——cant_20 链节点 nd 数百~上千、m=1 逐 d 调用时该开销
    // 占 CUpdate 80%+）。目标块按 d 划分互不相交，段间无锁竞争。
    auto updateStride = [&](int t, int nthr) -> bool {
        std::vector<const ColCompMat*> LList;
        std::vector<std::vector<BlockState*>> blockList;
        for (size_t di = static_cast<size_t>(t); di < nd;
             di += static_cast<size_t>(nthr)) {
            BlockState* left = leftStates[di];
            if (left->value.rows() == 0) {
                // 0 模态链路：目标块须同为 0 行零块，仅倒计数
                for (size_t a = 0; a < ancestors.size(); ++a) {
                    BlockState* target =
                        findBlock_(Key{descendants[di], ancestors[a]});
                    if (target == nullptr) {
                        AMLS_LOG_ERROR("update({}) 失败：目标块 ({},{}) 缺失。",
                                       nodeID, descendants[di], ancestors[a]);
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(target->mtx);
                    if (target->lifecycle.load(std::memory_order_relaxed) !=
                        Lifecycle::Active) {
                        AMLS_LOG_ERROR(
                            "update({}) 失败：目标块 ({},{}) 未激活。", nodeID,
                            descendants[di], ancestors[a]);
                        return false;
                    }
                    if (target->remainingUpdates.load(
                            std::memory_order_relaxed) <= 0) {
                        AMLS_LOG_ERROR(
                            "update({}) 失败：目标块 ({},{}) 剩余更新 {} 非正。",
                            nodeID, descendants[di], ancestors[a],
                            target->remainingUpdates.load(
                                std::memory_order_relaxed));
                        return false;
                    }
                    if (!target->value.numericReady() ||
                        target->value.rows() != 0) {
                        AMLS_LOG_ERROR(
                            "update({}) 失败：0 模态目标块 ({},{}) 不一致。",
                            nodeID, descendants[di], ancestors[a]);
                        return false;
                    }
                    target->remainingUpdates.fetch_sub(1,
                                                       std::memory_order_release);
                }
                continue;
            }
            // 数值路径预检 + 收集（写入延迟到段末堆叠批量）
            std::vector<BlockState*> tblocks;
            tblocks.reserve(ancestors.size());
            for (size_t a = 0; a < ancestors.size(); ++a) {
                BlockState* target =
                    findBlock_(Key{descendants[di], ancestors[a]});
                if (target == nullptr) {
                    AMLS_LOG_ERROR("update({}) 失败：目标块 ({},{}) 缺失。", nodeID,
                                   descendants[di], ancestors[a]);
                    return false;
                }
                if (target->lifecycle.load(std::memory_order_relaxed) !=
                    Lifecycle::Active) {
                    AMLS_LOG_ERROR("update({}) 失败：目标块 ({},{}) 未激活。",
                                   nodeID, descendants[di], ancestors[a]);
                    return false;
                }
                if (target->remainingUpdates.load(std::memory_order_relaxed) <=
                    0) {
                    AMLS_LOG_ERROR(
                        "update({}) 失败：目标块 ({},{}) 剩余更新 {} 非正。",
                        nodeID, descendants[di], ancestors[a],
                        target->remainingUpdates.load(
                            std::memory_order_relaxed));
                    return false;
                }
                tblocks.push_back(target);
            }
            LList.push_back(&left->value);
            blockList.push_back(std::move(tblocks));
        }
        if (!LList.empty()) {
            std::vector<std::vector<ColCompMat*>> tgtList(LList.size());
            std::vector<std::vector<std::mutex*>> mtxList(LList.size());
            for (size_t k = 0; k < LList.size(); ++k) {
                tgtList[k].reserve(blockList[k].size());
                mtxList[k].reserve(blockList[k].size());
                for (BlockState* b : blockList[k]) {
                    tgtList[k].push_back(&b->value);
                    mtxList[k].push_back(&b->mtx);
                }
            }
            // 列规划保证目标存储列覆盖 Ψ 有效列（覆盖性检查即预留定理
            // 的运行时断言）
            if (!opsColAccBatchStacked(LList, psis, tgtList, mtxList)) {
                AMLS_LOG_ERROR(
                    "update({}) 失败：堆叠批量 μ(d,{}) += μ(d,{})·Ψ({{},·) "
                    "出错（维度不匹配或覆盖缺失）。",
                    nodeID, nodeID, nodeID, nodeID);
                return false;
            }
            for (auto& blocks : blockList)
                for (BlockState* b : blocks)
                    b->remainingUpdates.fetch_sub(1, std::memory_order_release);
        }
        return true;
    };

    // 近根节点 descendant 数大，逐 d 串行是 REDUCTION 关键链主项
    // （cant_20 时间线：链节点 CUpdate 100-290ms 全在 d 循环）；按 d
    // 固定分段并行。实验证明按池空闲容量（opsTlsThreads）门控反而更慢：
    // 溢洪期并非纯吞吐受限——各节点 CUpdate 墙钟直接串进其后续节点的
    // 就绪时间，段内串行拉长尾延迟、拖慢整片洪泛的排空。
    const int nthr = nd < 32 ? 1 : static_cast<int>(std::min<size_t>(8, nd / 16));
    bool ok = true;
    if (nthr <= 1) {
        ok = updateStride(0, 1);
    } else {
        std::atomic<bool> allOk{true};
        {
            colcomp_detail::ThreadGroup tg(static_cast<size_t>(nthr - 1));
            for (int t = 0; t < nthr; ++t) {
                auto work = [&, t]() {
                    blas::pinThisThread();  // BLAS 全局单线程语义（见 solver 编排）
                    if (!updateStride(t, nthr))
                        allOk.store(false, std::memory_order_release);
                };
                if (t == nthr - 1)
                    work();
                else
                    tg.start(work);
            }
            tg.wait();
        }
        ok = allOk.load(std::memory_order_acquire);
    }
    if (cupdateLog) {
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cuT0)
                .count();
        AMLS_LOG_INFO("[cupdate] node={} nd={} thr={} wall={:.1f}ms", nodeID,
                      nd, nthr, ms);
    }
    return ok;
}

bool TempBlockManager::rightMultiplyColComp_(int64_t nodeID) {
    const std::vector<int64_t> descendants = tree_->descendants(nodeID);
    EigenPairSaver::View view;
    if (!eigenPairSaver_->getEigenPairView(nodeID, view)) {
        AMLS_LOG_DEBUG("rightMultiply({}) 未就绪：特征对缺失。", nodeID);
        return false;
    }
    const DefaultTraits::DenseMatrix* phi = view.eigenvectors();

    for (int64_t d : descendants) {
        BlockState* state = findBlock_(Key{d, nodeID});
        if (state == nullptr) {
            AMLS_LOG_DEBUG("rightMultiply({}) 未就绪：块 ({},{}) 缺失。", nodeID,
                           d, nodeID);
            return false;
        }
        // 无锁校验：remainingUpdates==0 即只读终态（release 序与写方同步）
        if (state->lifecycle.load(std::memory_order_acquire) ==
            Lifecycle::Released) {
            AMLS_LOG_ERROR("rightMultiply({}) 失败：块 ({},{}) 已释放。", nodeID,
                           d, nodeID);
            return false;
        }
        if (state->lifecycle.load(std::memory_order_acquire) !=
                Lifecycle::Active ||
            state->remainingUpdates.load(std::memory_order_acquire) != 0) {
            AMLS_LOG_DEBUG(
                "rightMultiply({}) 未就绪：块 ({},{}) 剩余更新未清零。", nodeID,
                d, nodeID);
            return false;
        }
        if (!state->value.numericReady() ||
            phi->rows() != state->value.totalCols()) {
            AMLS_LOG_ERROR(
                "rightMultiply({}) 失败：块 ({},{}) 或 Φ 维度不一致 (μ totalCols="
                "{}, Φ rows={})。",
                nodeID, d, nodeID, state->value.totalCols(), phi->rows());
            return false;
        }
        DefaultTraits::DenseMatrix product;
        if (state->value.rows() == 0 || phi->cols() == 0)
            product = DefaultTraits::DenseMatrix::Zero(state->value.rows(),
                                                       phi->cols());
        else if (!opsColDenseProd(state->value, *phi, product)) {
            AMLS_LOG_ERROR("rightMultiply({}) 失败：μ ({},{})·Φ 维度不匹配。",
                           nodeID, d, nodeID);
            return false;
        }
        if (!massSaver_->set(d, nodeID, std::move(product))) {
            AMLS_LOG_ERROR("rightMultiply({}) 失败：质量块 ({},{}) 提交被拒。",
                           nodeID, d, nodeID);
            return false;
        }
    }
    return true;
}

bool TempBlockManager::create(int64_t nodeID) {
    if (!enterTask_(nodeID, true)) return false;
    const bool ok = createColComp_(nodeID);
    if (ok) (*doneFlags_[static_cast<size_t>(nodeID)]).fetch_or(
        0x1, std::memory_order_release);
    leaveTask_(nodeID);
    return ok;
}

bool TempBlockManager::update(int64_t nodeID) {
    if (!enterTask_(nodeID, false)) return false;
    const bool ok = updateColComp_(nodeID);
    if (ok) (*doneFlags_[static_cast<size_t>(nodeID)]).fetch_or(
        0x2, std::memory_order_release);
    leaveTask_(nodeID);
    return ok;
}

bool TempBlockManager::rightMultiply(int64_t nodeID) {
    if (!enterTask_(nodeID, false)) return false;
    const bool ok = rightMultiplyColComp_(nodeID);
    if (ok) (*doneFlags_[static_cast<size_t>(nodeID)]).fetch_or(
        0x4, std::memory_order_release);
    leaveTask_(nodeID);
    return ok;
}

bool TempBlockManager::releaseNodeBlocks_(int64_t nodeID) {
    // 该索引只含 (d, nodeID)，初始化时已按 key 排序
    const auto& states = nodeReleaseBlocks_[static_cast<size_t>(nodeID)];
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(states.size());
    for (const auto& state : states) locks.emplace_back(state->mtx);
    for (const auto& state : states) {
        state->value = ColCompMat();
        state->remainingUpdates.store(-1, std::memory_order_relaxed);
        state->lifecycle.store(Lifecycle::Released, std::memory_order_release);
    }
    return true;
}

bool TempBlockManager::release(int64_t nodeID) {
    NodeGate* gate = gate_(nodeID);
    if (!initialized() || gate == nullptr) {
        AMLS_LOG_ERROR("TempBlockManager::release 失败：未初始化或节点越界。");
        return false;
    }

    if (nodeID != tree_->root()) {
        // 非根：关闭新任务入口，等已登记调用退出
        {
            std::unique_lock<std::mutex> lock(gate->mtx);
            while (gate->releaseInProgress) gate->cv.wait(lock);
            if (gate->released) return true;
            gate->releaseInProgress = true;
            ++gate->releaseRequests;
            gate->cv.wait(lock, [gate]() { return gate->activeCalls == 0; });
            if (gate->released) {
                --gate->releaseRequests;
                gate->releaseInProgress = false;
                gate->cv.notify_all();
                return true;
            }
        }
        const uint8_t done =
            (*doneFlags_[static_cast<size_t>(nodeID)]).load(std::memory_order_acquire);
        if ((done & 0x2) == 0 || (done & 0x4) == 0) {
            // update/rightMultiply 未完成：撤销请求，由调用方稍后重试
            std::lock_guard<std::mutex> lock(gate->mtx);
            --gate->releaseRequests;
            gate->releaseInProgress = false;
            gate->cv.notify_all();
            return false;
        }
        releaseNodeBlocks_(nodeID);
        {
            std::lock_guard<std::mutex> lock(gate->mtx);
            gate->released = true;
            gate->releaseInProgress = false;
            gate->cv.notify_all();
        }
        return true;
    }

    // 根节点：整树释放（单树直通，串行化由状态机保证）
    RootRelease expected = RootRelease::Idle;
    if (!rootRelease_.compare_exchange_strong(expected, RootRelease::Running,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        return expected == RootRelease::Done;  // 已完成幂等；进行中/失败为未就绪
    }
    if (rootRelease_.load(std::memory_order_acquire) == RootRelease::Failed)
        return false;

    // 关闭全树任务入口并等在途调用退出
    std::vector<NodeGate*> allGates;
    allGates.reserve(static_cast<size_t>(N_));
    for (int64_t i = 0; i < N_; ++i) allGates.push_back(gates_[static_cast<size_t>(i)].get());
    for (NodeGate* g : allGates) {
        std::lock_guard<std::mutex> lock(g->mtx);
        ++g->releaseRequests;
    }
    for (NodeGate* g : allGates) {
        std::unique_lock<std::mutex> lock(g->mtx);
        g->cv.wait(lock, [g]() { return g->activeCalls == 0; });
    }

    bool allDone = true;
    for (int64_t i = 0; i < N_; ++i) {
        const uint8_t done = (*doneFlags_[static_cast<size_t>(i)]).load(
            std::memory_order_acquire);
        if ((done & 0x7) != 0x7) {
            allDone = false;
            break;
        }
    }
    if (!allDone) {
        // 未完成：重新开放入口，由调用方稍后重试
        for (NodeGate* g : allGates) {
            std::lock_guard<std::mutex> lock(g->mtx);
            if (!g->released && g->releaseRequests > 0) --g->releaseRequests;
            g->cv.notify_all();
        }
        rootRelease_.store(RootRelease::Idle, std::memory_order_release);
        return false;
    }

    for (auto& [key, state] : blocks_) {
        std::lock_guard<std::mutex> lock(state->mtx);
        if (state->lifecycle.load(std::memory_order_relaxed) ==
            Lifecycle::Released)
            continue;
        state->value = ColCompMat();
        state->remainingUpdates.store(-1, std::memory_order_relaxed);
        state->lifecycle.store(Lifecycle::Released, std::memory_order_release);
    }
    for (NodeGate* g : allGates) {
        std::lock_guard<std::mutex> lock(g->mtx);
        g->released = true;
        if (g->releaseRequests == 0) g->releaseRequests = 1;
        g->cv.notify_all();
    }
    rootRelease_.store(RootRelease::Done, std::memory_order_release);
    return true;
}

void TempBlockManager::clear() {
    for (auto& [key, state] : blocks_) {
        std::lock_guard<std::mutex> lock(state->mtx);
        state->value = ColCompMat();
        state->remainingUpdates.store(-1, std::memory_order_relaxed);
        state->lifecycle.store(Lifecycle::Released, std::memory_order_release);
    }
    blocks_.clear();
    nodeReleaseBlocks_.clear();
    gates_.clear();
    doneFlags_.clear();
    tree_ = nullptr;
    gauss_ = nullptr;
    eigenPairSaver_ = nullptr;
    massSaver_ = nullptr;
    N_ = 0;
    rootRelease_.store(RootRelease::Idle, std::memory_order_release);
}

} // namespace aams
