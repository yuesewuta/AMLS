#include "ams/gauss_module.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "core/logging.hpp"
#include "linalg/blas.hpp"
#include "linalg/gemm_cols.hpp"
#include "ams/node_task_executor.hpp"

namespace aams {
namespace {

// parallelAncLoop 诊断计数（AAMS_RED_CLOSE_LOG 输出）。
std::atomic<int64_t> ancLoopCalls{0};
std::atomic<int64_t> ancLoopSerial{0};
std::atomic<int64_t> ancLoopGranted{0};

// 近根任务 m-循环并行（实验结论：借线程并行化在 cant_31 无净收益——
// 尾段 4 个并发任务时 K/M 互借导致部分串行 + helper 生成/同步开销，
// 实测 33s vs 串行 16s，已回退为串行。保留计数器与 [mtask] 打点。）
thread_local int tl_ancThreads = 1;
template <typename Fn>
bool parallelAncLoop(size_t n, Fn body) {
    tl_ancThreads = 1;
    for (size_t m = 0; m < n; ++m)
        if (!body(m)) return false;
    return true;
}

}  // namespace

GaussEliminationModule::Key GaussEliminationModule::reducedKey_(
    bool stiffness, int64_t i, int64_t j) const {
    const Kind kind = i == j ? (stiffness ? Kind::ReducedDiagK : Kind::ReducedDiagM)
                             : (stiffness ? Kind::ReducedCoupK : Kind::ReducedCoupM);
    return {kind, std::min(i, j), std::max(i, j)};
}

GaussEliminationModule::Entry* GaussEliminationModule::slot_(
    const Key& key) const {
    const auto it = slots_.find(key);
    return it == slots_.end() ? nullptr : it->second.get();
}

const DefaultTraits::SparseMatrix* GaussEliminationModule::originalBlock_(
    bool stiffness, int64_t desc, int64_t anc) const {
    if (desc == anc)
        return stiffness ? &blocks_->diagK(desc) : &blocks_->diagM(desc);
    return stiffness ? blocks_->findSubK(desc, anc)
                     : blocks_->findSubM(desc, anc);
}

std::vector<int64_t> GaussEliminationModule::originalColumns_(
    bool stiffness, int64_t desc, int64_t anc) const {
    const SparseMatrix* src = originalBlock_(stiffness, desc, anc);
    std::vector<int64_t> cols;
    if (!src) return cols;  // 无结构耦合 → 零块
    for (int64_t c = 0; c < src->cols(); ++c)
        if (src->outerIndexPtr()[c + 1] > src->outerIndexPtr()[c])
            cols.push_back(c);
    return cols;
}

bool GaussEliminationModule::ensureReduced_(bool stiffness, int64_t desc,
                                            int64_t anc, Entry& entry,
                                            ColCompMat*& out) {
    out = &entry.matrix;
    if (out->numericReady()) return true;
    if (!out->layoutReady()) {
        AMLS_LOG_ERROR(
            "ensureReduced 失败：{} 块 ({},{}) 布局未就绪（规划应已覆盖）。",
            stiffness ? "K" : "M", desc, anc);
        return false;
    }
    const SparseMatrix* src = originalBlock_(stiffness, desc, anc);
    if (!src || (src->nonZeros() == 0 && desc != anc)) {
        // 缺失耦合块（无结构耦合，未建槽）：零初始化（布局内全为预留占位列）
        if (!out->initializeZero()) {
            AMLS_LOG_ERROR("ensureReduced 失败：{} 耦合块 ({},{}) 零初始化被拒。",
                           stiffness ? "K" : "M", desc, anc);
            return false;
        }
        return true;
    }
    if (!out->initializeFromSparse(*src)) {
        AMLS_LOG_ERROR(
            "ensureReduced 失败：{} 块 ({},{}) 从原始子块注入被拒 (块: {}x{}, "
            "validCols={}; src: {}x{})。",
            stiffness ? "K" : "M", desc, anc, out->rows(), out->totalCols(),
            out->validCols(), src->rows(), src->cols());
        return false;
    }
    return true;
}

bool GaussEliminationModule::initialize(const OriginalBlocks& blocks) {
    const auto gstart = std::chrono::steady_clock::now();
    blocks_ = &blocks;
    tree_ = &blocks.tree();
    N_ = blocks.nNodes();
    useGemv_ = useColAccGemv();

    // 1. 槽位：K/M 对角 + 耦合 + Ψ/M_temp 方向槽位（运行期零 map 插入）。
    //    分配（2.2M Entry）并行预构造，map 插入串行搬移 unique_ptr。
    int64_t sumAnc = 0;
    for (int64_t i = 0; i < N_; ++i)
        sumAnc += static_cast<int64_t>(tree_->ancestors(i).size());
    slots_.clear();
    slots_.reserve(static_cast<size_t>(N_ * 2 + 4 * sumAnc));
    std::vector<std::pair<Key, std::unique_ptr<Entry>>> staged;
    staged.reserve(static_cast<size_t>(N_ * 2 + 4 * sumAnc));
    std::vector<int64_t> off(static_cast<size_t>(N_ + 1), 0);
    for (int64_t i = 0; i < N_; ++i)
        off[static_cast<size_t>(i) + 1] =
            off[static_cast<size_t>(i)] + 2 + 4 * static_cast<int64_t>(
                                                   tree_->ancestors(i).size());
    staged.resize(static_cast<size_t>(off[static_cast<size_t>(N_)]));
    {
        const int nthr = 8;
        std::atomic<int64_t> next{0};
        colcomp_detail::ThreadGroup tg(nthr - 1);
        auto fill = [&] {
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= N_) break;
                size_t p = static_cast<size_t>(off[static_cast<size_t>(i)]);
                const int64_t ni = blocks.size(i);
                for (bool stiffness : {true, false}) {
                    staged[p++] = {reducedKey_(stiffness, i, i),
                                   std::make_unique<Entry>(ni, ni)};
                }
                for (int64_t a : tree_->ancestors(i)) {
                    const int64_t na = blocks.size(a);
                    for (bool stiffness : {true, false}) {
                        staged[p++] = {reducedKey_(stiffness, i, a),
                                       std::make_unique<Entry>(ni, na)};
                    }
                    staged[p++] = {Key{Kind::Psi, i, a},
                                   std::make_unique<Entry>()};
                    staged[p++] = {Key{Kind::MTemp, i, a},
                                   std::make_unique<Entry>()};
                }
            }
        };
        for (int t = 0; t < nthr; ++t) {
            if (t == nthr - 1) fill();
            else tg.start(fill);
        }
        tg.wait();
    }
    for (auto& [key, entry] : staged)
        slots_.emplace(key, std::move(entry));

    // 2. 自底向上列规划。
    const auto gsplit = std::chrono::steady_clock::now();
    if (!planColumns_()) {
        blocks_ = nullptr;
        tree_ = nullptr;
        slots_.clear();
        return false;
    }
    if (std::getenv("AAMS_RED_CLOSE_LOG")) {
        // 布局内存盘点：稠密 frontal 化（n_d×n_a 全满）vs 现列压缩
        //（validCols×rows）的对比，评估 frontal 化可行性。
        double denseBytes = 0, planBytes = 0;
        for (const auto& [key, entry] : slots_) {
            const ColCompMat& m = entry->matrix;
            if (!m.layoutReady()) continue;
            denseBytes += static_cast<double>(m.rows()) * m.totalCols() * 8;
            planBytes += static_cast<double>(m.rows()) * m.validCols() * 8;
        }
        std::fprintf(stderr,
                     "[ginit] slots=%.1fms plan=%.1fms mem dense=%.2fGB "
                     "planned=%.2fGB ratio=%.2f\n",
                     std::chrono::duration<double, std::milli>(gsplit - gstart)
                         .count(),
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - gsplit)
                         .count(),
                     denseBytes / 1e9, planBytes / 1e9,
                     planBytes > 0 ? denseBytes / planBytes : 0.0);
    }
    return true;
}

bool GaussEliminationModule::planColumns_() {
    const std::vector<int64_t> order = tree_->postOrder();
    auto fullColumns = [](int64_t n) {
        std::vector<int64_t> cols(static_cast<size_t>(n));
        std::iota(cols.begin(), cols.end(), int64_t{0});
        return cols;
    };

    // 单节点规划：对角满列 + 该节点全部祖先对的耦合布局。只读孩子的
    // 布局（后序序中孩子先规划），节点间互不相交 → 按层级并行。
    auto planNode = [&](int64_t j) -> bool {
        const std::vector<int64_t> full = fullColumns(blocks_->size(j));
        for (bool stiffness : {true, false}) {
            auto* diag = slot_(reducedKey_(stiffness, j, j));
            if (!diag || !diag->matrix.setLayout(full)) {
                AMLS_LOG_ERROR(
                    "planColumns 失败：{} 对角块 ({},{}) 缺失或布局被拒。",
                    stiffness ? "K" : "M", j, j);
                return false;
            }
        }
        const std::vector<int64_t>& children = tree_->node(j).children;
        for (int64_t k : tree_->ancestors(j)) {
            const int64_t nk = blocks_->size(k);
            // ---- K 侧 ----
            std::vector<int64_t> rk = originalColumns_(true, j, k);
            bool kFull = static_cast<int64_t>(rk.size()) == nk;
            for (int64_t c : children) {
                if (kFull) break;
                const auto* childK = slot_(reducedKey_(true, c, k));
                if (!childK || !childK->matrix.layoutReady()) {
                    AMLS_LOG_ERROR(
                        "planColumns 失败：孩子 K 块 ({},{}) 缺失或布局未就绪"
                        "（规划 ({},{}) 时）。",
                        c, k, j, k);
                    return false;
                }
                if (childK->matrix.validCols() == nk) {
                    kFull = true;
                    break;
                }
                mergeColumns_(rk, childK->matrix.storedCols());
                kFull = static_cast<int64_t>(rk.size()) == nk;
            }
            auto* kjk = slot_(reducedKey_(true, j, k));
            if (!kjk ||
                !kjk->matrix.setLayout(kFull ? fullColumns(nk) : rk)) {
                AMLS_LOG_ERROR(
                    "planColumns 失败：K 耦合块 ({},{}) 缺失或布局被拒 ({} 列, "
                    "n_k={})。",
                    j, k, kFull ? nk : static_cast<int64_t>(rk.size()), nk);
                return false;
            }
            // ---- M 侧 ----
            std::vector<int64_t> rm = originalColumns_(false, j, k);
            bool mFull = static_cast<int64_t>(rm.size()) == nk;
            for (int64_t c : children) {
                if (mFull) break;
                const auto* childM = slot_(reducedKey_(false, c, k));
                const auto* childK = slot_(reducedKey_(true, c, k));
                if (!childM || !childM->matrix.layoutReady() || !childK) {
                    AMLS_LOG_ERROR(
                        "planColumns 失败：孩子 M/K 块 ({},{}) 缺失或布局未就绪"
                        "（规划 ({},{}) 时）。",
                        c, k, j, k);
                    return false;
                }
                if (childM->matrix.validCols() == nk ||
                    childK->matrix.validCols() == nk) {
                    mFull = true;
                    break;
                }
                mergeColumns_(rm, childM->matrix.storedCols());
                mergeColumns_(rm, childK->matrix.storedCols());
                mFull = static_cast<int64_t>(rm.size()) == nk;
            }
            auto* mjk = slot_(reducedKey_(false, j, k));
            if (!mjk ||
                !mjk->matrix.setLayout(mFull ? fullColumns(nk) : rm)) {
                AMLS_LOG_ERROR(
                    "planColumns 失败：M 耦合块 ({},{}) 缺失或布局被拒 ({} 列, "
                    "n_k={})。",
                    j, k, mFull ? nk : static_cast<int64_t>(rm.size()), nk);
                return false;
            }
        }
        return true;
    };

    // 层级划分：lvl(j) = 1 + max(lvl(孩子))。同层节点依赖已全部就绪。
    std::vector<int> lvl(static_cast<size_t>(N_), 0);
    int maxLvl = 0;
    for (int64_t j : order) {
        int l = 0;
        for (int64_t c : tree_->node(j).children)
            l = std::max(l, lvl[static_cast<size_t>(c)] + 1);
        lvl[static_cast<size_t>(j)] = l;
        maxLvl = std::max(maxLvl, l);
    }
    std::vector<std::vector<int64_t>> byLvl(
        static_cast<size_t>(maxLvl + 1));
    for (int64_t j : order)
        byLvl[static_cast<size_t>(lvl[static_cast<size_t>(j)])].push_back(j);

    const int nthr = 8;
    for (auto& bucket : byLvl) {
        if (bucket.empty()) continue;
        if (nthr <= 1 || bucket.size() < 4) {
            for (int64_t j : bucket)
                if (!planNode(j)) return false;
            continue;
        }
        std::atomic<size_t> next{0};
        std::atomic<bool> fail{false};
        colcomp_detail::ThreadGroup tg(static_cast<size_t>(nthr - 1));
        auto work = [&] {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= bucket.size() ||
                    fail.load(std::memory_order_relaxed))
                    break;
                if (!planNode(bucket[i]))
                    fail.store(true, std::memory_order_relaxed);
            }
        };
        for (int t = 0; t < nthr; ++t) {
            if (t == nthr - 1) work();
            else tg.start(work);
        }
        tg.wait();
        if (fail.load(std::memory_order_relaxed)) return false;
    }
    return true;
}

bool GaussEliminationModule::computeConstraint(int64_t nodeI) {
    const std::vector<int64_t> anc = tree_->ancestors(nodeI);
    if (anc.empty()) return true;

    ColCompMat* kii = nullptr;
    {
        Entry* entry = slot_(reducedKey_(true, nodeI, nodeI));
        if (!entry) {
            AMLS_LOG_ERROR("computeConstraint 失败：K 对角块 ({},{}) 槽位缺失。",
                           nodeI, nodeI);
            return false;
        }
        if (!ensureReduced_(true, nodeI, nodeI, *entry, kii)) return false;
    }
    DenseMatrix lu = kii->dense();
    std::vector<int32_t> ipiv;
    if (lu.rows() > 0 && sytrfFactor(lu, ipiv) != 0) {
        AMLS_LOG_ERROR("computeConstraint 失败：K 对角块 ({},{}) 分解失败 (阶数 {})。",
                       nodeI, nodeI, lu.rows());
        return false;
    }

    for (int64_t j : anc) {
        ColCompMat* kij = nullptr;
        {
            Entry* entry = slot_(reducedKey_(true, nodeI, j));
            if (!entry) {
                AMLS_LOG_ERROR("computeConstraint 失败：K 耦合块 ({},{}) 槽位缺失。",
                               nodeI, j);
                return false;
            }
            if (!ensureReduced_(true, nodeI, j, *entry, kij)) return false;
        }
        Entry* psiEntry = slot_(Key{Kind::Psi, nodeI, j});
        if (!psiEntry) {
            AMLS_LOG_ERROR("computeConstraint 失败：Ψ ({},{}) 槽位缺失。", nodeI, j);
            return false;
        }
        ColCompMat* psi = &psiEntry->matrix;
        if (lu.rows() == 0) {
            if (!psi->initializeConstraintFromRhs(*kij)) {
                AMLS_LOG_ERROR("computeConstraint 失败：零阶求解 Ψ ({},{}) 被拒。",
                               nodeI, j);
                return false;
            }
        } else if (!opsColSolve(lu.rows(), lu, ipiv, *kij, *psi)) {
            AMLS_LOG_ERROR("computeConstraint 失败：Ψ ({},{}) 回代失败 (阶数 {})。",
                           nodeI, j, lu.rows());
            return false;
        }
        psi->dense().array() *= -1.0;  // Ψ = −K̂⁻¹·K̃_ij
    }
    return true;
}

bool GaussEliminationModule::accumulateReducedStiffness(int64_t nodeI) {
    const std::vector<int64_t> anc = tree_->ancestors(nodeI);
    for (int64_t j : anc) {
        const auto* psi = getPsi(nodeI, j);
        if (!psi || !psi->numericReady()) {
            AMLS_LOG_ERROR(
                "accumulateReducedStiffness 失败：Ψ ({},{}) 缺失或未数值就绪。",
                nodeI, j);
            return false;
        }
    }
    static const bool noSuffix =
        std::getenv("AAMS_NO_TRANS_SUFFIX") != nullptr;
    static const bool legacyTransK =
        std::getenv("AAMS_TRANS_FULL") != nullptr;
    // 全积路径（AAMS_TRANS_FULL=1 实验）：左端 Ψ(I,·) 堆叠 + 右端 K̃(I,·)
    // 一次全积 gemm。实测负结果（cant_31 REDUCTION 13.65 → 15.84s）：
    // DAG 洪泛期 opsTlsThreads = cap/inFlight ≈ 1，大 gemm 不涨 GF/s 只涨
    // flops（trans 4.95e11 → 6.48e11，GF/s 17.0 → 15.4）。默认走后缀路径。
    if (!noSuffix && legacyTransK && !anc.empty()) {
        std::vector<const ColCompMat*> lefts, rights;
        std::vector<std::vector<ColCompMat*>> targetsPerLeft(anc.size());
        std::vector<std::vector<std::mutex*>> mtxsPerLeft(anc.size());
        lefts.reserve(anc.size());
        rights.reserve(anc.size());
        for (int64_t j : anc) lefts.push_back(getPsi(nodeI, j));
        for (size_t t = 0; t < anc.size(); ++t) {
            const int64_t k = anc[t];
            ColCompMat* source = nullptr;
            Entry* entry = slot_(reducedKey_(true, nodeI, k));
            if (!entry) {
                AMLS_LOG_ERROR(
                    "accumulateReducedStiffness 失败：K 源块 ({},{}) 槽位缺失。",
                    nodeI, k);
                return false;
            }
            if (!ensureReduced_(true, nodeI, k, *entry, source)) return false;
            rights.push_back(source);
        }
        for (size_t m = 0; m < anc.size(); ++m) {
            const int64_t j = anc[m];
            for (size_t t = m; t < anc.size(); ++t) {
                const int64_t k = anc[t];
                Entry* targetEntry = slot_(reducedKey_(true, j, k));
                if (!targetEntry) {
                    AMLS_LOG_ERROR("accumulateReducedStiffness 失败：K 目标块 "
                                   "({},{}) 槽位缺失。",
                                   j, k);
                    return false;
                }
                {
                    std::lock_guard<std::mutex> lock(targetEntry->mtx);
                    ColCompMat* target = nullptr;
                    if (!ensureReduced_(true, j, k, *targetEntry, target))
                        return false;
                    targetsPerLeft[m].push_back(target);
                }
                mtxsPerLeft[m].push_back(&targetEntry->mtx);
            }
        }
        return opsColAccFullProductTrans(lefts, rights, targetsPerLeft,
                                         mtxsPerLeft);
    }
    // 共享右端：全部 k ∈ anc 的 K̃(nodeI,k) 有效列只拼一次；anc 升序，
    // j = anc[m] 的目标 k≥j 恰为段后缀 [m, nSeg)。|anc|² 次小乘法摊成
    // |anc| 次 + 一遍 gather。
    TransRightEnd re;
    if (!noSuffix) {
        std::vector<const ColCompMat*> sources;
        sources.reserve(anc.size());
        for (int64_t k : anc) {
            ColCompMat* source = nullptr;
            Entry* entry = slot_(reducedKey_(true, nodeI, k));
            if (!entry) {
                AMLS_LOG_ERROR(
                    "accumulateReducedStiffness 失败：K 源块 ({},{}) 槽位缺失。",
                    nodeI, k);
                return false;
            }
            if (!ensureReduced_(true, nodeI, k, *entry, source)) return false;
            sources.push_back(source);
        }
        if (!sources.empty() &&
            !opsTransBuildRightEnd(sources.front()->rows(), sources, re))
            return false;
    }
    const bool okAll = parallelAncLoop(anc.size(), [&](size_t m) -> bool {
        const int64_t j = anc[m];
        const auto* psi = getPsi(nodeI, j);
        // 固定 (nodeI, j)：全部 k≥j 的目标块一次批量更新。
        std::vector<const ColCompMat*> srcs;
        std::vector<ColCompMat*> targets;
        std::vector<std::mutex*> mtxs;
        targets.reserve(anc.size() - m);
        mtxs.reserve(anc.size() - m);
        if (noSuffix) srcs.reserve(anc.size() - m);
        for (size_t t = m; t < anc.size(); ++t) {
            const int64_t k = anc[t];
            if (noSuffix) {
                ColCompMat* source = nullptr;
                Entry* entry = slot_(reducedKey_(true, nodeI, k));
                if (!entry) {
                    AMLS_LOG_ERROR("accumulateReducedStiffness 失败：K 源块 "
                                   "({},{}) 槽位缺失。",
                                   nodeI, k);
                    return false;
                }
                if (!ensureReduced_(true, nodeI, k, *entry, source))
                    return false;
                srcs.push_back(source);
            }
            Entry* targetEntry = slot_(reducedKey_(true, j, k));
            if (!targetEntry) {
                AMLS_LOG_ERROR("accumulateReducedStiffness 失败：K 目标块 ({},{}) "
                               "槽位缺失。",
                               j, k);
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(targetEntry->mtx);
                ColCompMat* target = nullptr;
                if (!ensureReduced_(true, j, k, *targetEntry, target))
                    return false;
                targets.push_back(target);
            }
            mtxs.push_back(&targetEntry->mtx);
        }
        const bool ok = noSuffix
                            ? opsColAccBatchTrans(*psi, srcs, targets, mtxs)
                            : opsColAccBatchTransSuffix(*psi, re,
                                                        static_cast<int64_t>(m),
                                                        targets, mtxs);
        if (!ok) {
            AMLS_LOG_ERROR(
                "accumulateReducedStiffness 失败：K̃ (j,{}) += Ψ ({},{})ᵀ·"
                "K̃ ({},{},·) 批量更新维度不匹配。",
                j, nodeI, j, nodeI);
            return false;
        }
        return true;
    });
    return okAll;
}

bool GaussEliminationModule::computeMtemp(int64_t nodeI) {
    ColCompMat* mii = nullptr;
    {
        Entry* entry = slot_(reducedKey_(false, nodeI, nodeI));
        if (!entry) {
            AMLS_LOG_ERROR("computeMtemp 失败：M 对角块 ({},{}) 槽位缺失。", nodeI,
                           nodeI);
            return false;
        }
        if (!ensureReduced_(false, nodeI, nodeI, *entry, mii)) return false;
    }
    for (int64_t j : tree_->ancestors(nodeI)) {
        ColCompMat* mij = nullptr;
        {
            Entry* entry = slot_(reducedKey_(false, nodeI, j));
            if (!entry) {
                AMLS_LOG_ERROR("computeMtemp 失败：M 耦合块 ({},{}) 槽位缺失。",
                               nodeI, j);
                return false;
            }
            if (!ensureReduced_(false, nodeI, j, *entry, mij)) return false;
        }
        const auto* psi = getPsi(nodeI, j);
        if (!psi || !psi->numericReady()) {
            AMLS_LOG_ERROR("computeMtemp 失败：Ψ ({},{}) 缺失或未数值就绪。", nodeI,
                           j);
            return false;
        }
        Entry* mtempEntry = slot_(Key{Kind::MTemp, nodeI, j});
        if (!mtempEntry) {
            AMLS_LOG_ERROR("computeMtemp 失败：M_temp ({},{}) 槽位缺失。", nodeI, j);
            return false;
        }
        ColCompMat* mtemp = &mtempEntry->matrix;
        if (!mtemp->initializeMtempFromInputs(*mij, *psi)) {
            AMLS_LOG_ERROR("computeMtemp 失败：M_temp ({},{}) 并集布局初始化被拒。",
                           nodeI, j);
            return false;
        }
        if (!opsColAcc(useGemv_, *mii, false, *psi, *mtemp)) {
            AMLS_LOG_ERROR(
                "computeMtemp 失败：M_temp ({},{}) += M̃ ({},{})·Ψ ({},{}) 维度"
                "不匹配。",
                nodeI, j, nodeI, nodeI, nodeI, j);
            return false;
        }
    }
    return true;
}

bool GaussEliminationModule::accumulateReducedMass(int64_t nodeI) {
    namespace chrono = std::chrono;
    static const bool profTask = std::getenv("AAMS_PROFILE") != nullptr;
    const auto taskT0 = chrono::steady_clock::now();
    const std::vector<int64_t> anc = tree_->ancestors(nodeI);
    for (int64_t j : anc) {
        const auto* psiJ = getPsi(nodeI, j);
        const auto* mtemp = getMTemp(nodeI, j);
        if (!psiJ || !mtemp || !psiJ->numericReady() || !mtemp->numericReady()) {
            AMLS_LOG_ERROR(
                "accumulateReducedMass 失败：Ψ/M_temp ({},{}) 缺失或未数值就绪。",
                nodeI, j);
            return false;
        }
    }
    static const bool noSuffix =
        std::getenv("AAMS_NO_TRANS_SUFFIX") != nullptr;
    // 共享右端 ×2（项1 B=Ψ_j × C=M̃(nodeI,·)；项2 B=M_temp_j × C=Ψ(nodeI,·)），
    // anc 升序 → j = anc[m] 的 k≥j 为段后缀 [m, nSeg)。
    TransRightEnd re1, re2;
    if (!noSuffix) {
        std::vector<const ColCompMat*> src1, src2;
        src1.reserve(anc.size());
        src2.reserve(anc.size());
        for (int64_t k : anc) {
            const auto* psiK = getPsi(nodeI, k);
            if (!psiK || !psiK->numericReady()) {
                AMLS_LOG_ERROR(
                    "accumulateReducedMass 失败：Ψ ({},{}) 缺失或未数值就绪。",
                    nodeI, k);
                return false;
            }
            ColCompMat* source = nullptr;
            {
                Entry* entry = slot_(reducedKey_(false, nodeI, k));
                if (!entry) {
                    AMLS_LOG_ERROR(
                        "accumulateReducedMass 失败：M 源块 ({},{}) 槽位缺失。",
                        nodeI, k);
                    return false;
                }
                if (!ensureReduced_(false, nodeI, k, *entry, source))
                    return false;
            }
            src1.push_back(source);
            src2.push_back(psiK);
        }
        if (!src1.empty()) {
            if (!opsTransBuildRightEnd(src1.front()->rows(), src1, re1) ||
                !opsTransBuildRightEnd(src2.front()->rows(), src2, re2))
                return false;
        }
    }
    const auto reT = chrono::steady_clock::now();
    std::atomic<uint64_t> ensureNs{0}, opsNs{0};
    const bool okAll = parallelAncLoop(anc.size(), [&](size_t m) -> bool {
        const auto mT0 = chrono::steady_clock::now();
        const int64_t j = anc[m];
        const auto* psiJ = getPsi(nodeI, j);
        const auto* mtemp = getMTemp(nodeI, j);
        std::vector<const ColCompMat*> src1, src2;
        std::vector<ColCompMat*> targets;
        std::vector<std::mutex*> mtxs;
        targets.reserve(anc.size() - m);
        mtxs.reserve(anc.size() - m);
        if (noSuffix) {
            src1.reserve(anc.size() - m);
            src2.reserve(anc.size() - m);
        }
        for (size_t t = m; t < anc.size(); ++t) {
            const int64_t k = anc[t];
            if (noSuffix) {
                const auto* psiK = getPsi(nodeI, k);
                if (!psiK || !psiK->numericReady()) {
                    AMLS_LOG_ERROR(
                        "accumulateReducedMass 失败：Ψ ({},{}) 缺失或未数值就绪。",
                        nodeI, k);
                    return false;
                }
                ColCompMat* source = nullptr;
                {
                    Entry* entry = slot_(reducedKey_(false, nodeI, k));
                    if (!entry) {
                        AMLS_LOG_ERROR(
                            "accumulateReducedMass 失败：M 源块 ({},{}) 槽位缺失。",
                            nodeI, k);
                        return false;
                    }
                    if (!ensureReduced_(false, nodeI, k, *entry, source))
                        return false;
                }
                src1.push_back(source);
                src2.push_back(psiK);
            }
            Entry* targetEntry = slot_(reducedKey_(false, j, k));
            if (!targetEntry) {
                AMLS_LOG_ERROR(
                    "accumulateReducedMass 失败：M 目标块 ({},{}) 槽位缺失。", j, k);
                return false;
            }
            {
                std::lock_guard<std::mutex> lock(targetEntry->mtx);
                ColCompMat* target = nullptr;
                if (!ensureReduced_(false, j, k, *targetEntry, target))
                    return false;
                targets.push_back(target);
            }
            mtxs.push_back(&targetEntry->mtx);
        }
        const auto tOps0 = chrono::steady_clock::now();
        ensureNs.fetch_add(
            static_cast<uint64_t>(
                chrono::duration_cast<chrono::nanoseconds>(tOps0 - mT0)
                    .count()),
            std::memory_order_relaxed);
        const bool ok = noSuffix
                            ? opsColAccBatchTrans(*psiJ, src1, targets, mtxs) &&
                                  opsColAccBatchTrans(*mtemp, src2, targets, mtxs)
                            : opsColAccBatchTransSuffix(*psiJ, re1,
                                                        static_cast<int64_t>(m),
                                                        targets, mtxs) &&
                                  opsColAccBatchTransSuffix(*mtemp, re2,
                                                            static_cast<int64_t>(m),
                                                            targets, mtxs);
        opsNs.fetch_add(
            static_cast<uint64_t>(
                chrono::duration_cast<chrono::nanoseconds>(
                    chrono::steady_clock::now() - tOps0)
                    .count()),
            std::memory_order_relaxed);
        if (!ok) {
            AMLS_LOG_ERROR(
                "accumulateReducedMass 失败：M̃ (j,·) += Ψ ({},{})ᵀ·M̃ + "
                "M_temp ({},{})ᵀ·Ψ 批量更新维度不匹配。",
                nodeI, j, nodeI, j);
            return false;
        }
        return true;
    });
    if (profTask) {
        const uint64_t total = static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - taskT0)
                .count());
        if (total > 20000000)
            std::fprintf(stderr,
                         "[mtask] node=%lld anc=%zu thr=%d busy=%d lent=%d "
                         "size=%d total=%.1fms "
                         "rightEnd=%.1fms ensure=%.1fms ops=%.1fms\n",
                         (long long)nodeI, anc.size(), tl_ancThreads,
                         tl_lastGateBusy, tl_lastGateLent, tl_lastGateSize,
                         total / 1e6,
                         chrono::duration_cast<chrono::nanoseconds>(reT -
                                                                    taskT0)
                                 .count() /
                             1e6,
                         ensureNs.load(std::memory_order_relaxed) / 1e6,
                         opsNs.load(std::memory_order_relaxed) / 1e6);
    }
    return okAll;
}

const DefaultTraits::DenseMatrix* GaussEliminationModule::getReducedDiagK(
    int64_t j) const {
    const Entry* entry = slot_(reducedKey_(true, j, j));
    if (!entry || !entry->matrix.numericReady()) {
        AMLS_LOG_ERROR(
            "getReducedDiagK({}) 失败：K 对角块缺失或未数值就绪 (found={}, "
            "numericReady={})。",
            j, entry != nullptr, entry != nullptr && entry->matrix.numericReady());
        return nullptr;
    }
    return &entry->matrix.dense();  // 对角恒满列恒等布局，dense() 即全矩阵
}

const DefaultTraits::DenseMatrix* GaussEliminationModule::getReducedDiagM(
    int64_t j) const {
    const Entry* entry = slot_(reducedKey_(false, j, j));
    if (!entry || !entry->matrix.numericReady()) {
        AMLS_LOG_ERROR(
            "getReducedDiagM({}) 失败：M 对角块缺失或未数值就绪 (found={}, "
            "numericReady={})。",
            j, entry != nullptr, entry != nullptr && entry->matrix.numericReady());
        return nullptr;
    }
    return &entry->matrix.dense();
}

void GaussEliminationModule::releaseReducedDiagK(int64_t j) {
    if (Entry* entry = slot_(reducedKey_(true, j, j))) {
        std::lock_guard<std::mutex> lock(entry->mtx);
        entry->matrix.releaseNumeric();
    }
}

void GaussEliminationModule::releaseReducedDiagM(int64_t j) {
    if (Entry* entry = slot_(reducedKey_(false, j, j))) {
        std::lock_guard<std::mutex> lock(entry->mtx);
        entry->matrix.releaseNumeric();
    }
}

const ColCompMat* GaussEliminationModule::getPsi(int64_t i, int64_t j) const {
    const Entry* entry = slot_(Key{Kind::Psi, i, j});
    if (!entry || !entry->matrix.numericReady()) return nullptr;
    return &entry->matrix;
}

void GaussEliminationModule::releasePsi(int64_t i, int64_t j) {
    if (Entry* entry = slot_(Key{Kind::Psi, i, j})) {
        std::lock_guard<std::mutex> lock(entry->mtx);
        entry->matrix.releaseNumeric();
    }
}

const ColCompMat* GaussEliminationModule::getMTemp(int64_t i, int64_t j) const {
    const Entry* entry = slot_(Key{Kind::MTemp, i, j});
    if (!entry || !entry->matrix.numericReady()) return nullptr;
    return &entry->matrix;
}

void GaussEliminationModule::releaseMTemp(int64_t i, int64_t j) {
    if (Entry* entry = slot_(Key{Kind::MTemp, i, j})) {
        std::lock_guard<std::mutex> lock(entry->mtx);
        entry->matrix.releaseNumeric();
    }
}

const std::vector<int32_t>* GaussEliminationModule::getReservedColState(
    int64_t i, int64_t j) const {
    const Entry* entry = slot_(reducedKey_(true, i, j));
    if (!entry || !entry->matrix.layoutReady()) {
        AMLS_LOG_ERROR(
            "getReservedColState ({},{}) 失败：K 耦合块缺失或布局未就绪。", i, j);
        return nullptr;
    }
    return &entry->matrix.colState();
}

void reportAncLoopStats() {
    std::fprintf(stderr,
                 "[redclose] ancLoop calls=%lld serial=%lld grantedSum=%lld\n",
                 (long long)ancLoopCalls.load(std::memory_order_relaxed),
                 (long long)ancLoopSerial.load(std::memory_order_relaxed),
                 (long long)ancLoopGranted.load(std::memory_order_relaxed));
}

} // namespace aams
