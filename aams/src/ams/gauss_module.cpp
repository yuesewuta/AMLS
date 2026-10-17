#include "ams/gauss_module.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "core/logging.hpp"

namespace aams {

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

const DefaultTraits::SparseMatrix& GaussEliminationModule::originalBlock_(
    bool stiffness, int64_t desc, int64_t anc) const {
    if (desc == anc)
        return stiffness ? blocks_->diagK(desc) : blocks_->diagM(desc);
    return stiffness ? blocks_->subK(desc, anc) : blocks_->subM(desc, anc);
}

std::vector<int64_t> GaussEliminationModule::originalColumns_(
    bool stiffness, int64_t desc, int64_t anc) const {
    const SparseMatrix& src = originalBlock_(stiffness, desc, anc);
    std::vector<int64_t> cols;
    for (int64_t c = 0; c < src.cols(); ++c)
        if (src.outerIndexPtr()[c + 1] > src.outerIndexPtr()[c])
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
    const SparseMatrix& src = originalBlock_(stiffness, desc, anc);
    if (src.nonZeros() == 0 && desc != anc) {
        // 缺失耦合块：零初始化（布局内全为预留占位列）
        if (!out->initializeZero()) {
            AMLS_LOG_ERROR("ensureReduced 失败：{} 耦合块 ({},{}) 零初始化被拒。",
                           stiffness ? "K" : "M", desc, anc);
            return false;
        }
        return true;
    }
    if (!out->initializeFromSparse(src)) {
        AMLS_LOG_ERROR(
            "ensureReduced 失败：{} 块 ({},{}) 从原始子块注入被拒 (块: {}x{}, "
            "validCols={}; src: {}x{})。",
            stiffness ? "K" : "M", desc, anc, out->rows(), out->totalCols(),
            out->validCols(), src.rows(), src.cols());
        return false;
    }
    return true;
}

bool GaussEliminationModule::initialize(const OriginalBlocks& blocks) {
    blocks_ = &blocks;
    tree_ = &blocks.tree();
    N_ = blocks.nNodes();
    useGemv_ = useColAccGemv();

    // 1. 槽位：K/M 对角 + 耦合 + Ψ/M_temp 方向槽位（运行期零 map 插入）。
    slots_.clear();
    slots_.reserve(static_cast<size_t>(N_) * 4);
    for (int64_t i = 0; i < N_; ++i) {
        const int64_t ni = blocks.size(i);
        for (bool stiffness : {true, false}) {
            slots_[reducedKey_(stiffness, i, i)] =
                std::make_unique<Entry>(ni, ni);
        }
        for (int64_t a : tree_->ancestors(i)) {
            const int64_t na = blocks.size(a);
            for (bool stiffness : {true, false}) {
                slots_[reducedKey_(stiffness, i, a)] =
                    std::make_unique<Entry>(ni, na);
            }
            slots_[Key{Kind::Psi, i, a}] = std::make_unique<Entry>();
            slots_[Key{Kind::MTemp, i, a}] = std::make_unique<Entry>();
        }
    }

    // 2. 自底向上列规划。
    if (!planColumns_()) {
        blocks_ = nullptr;
        tree_ = nullptr;
        slots_.clear();
        return false;
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

    // 对角块：满列布局。
    for (int64_t i : order) {
        const std::vector<int64_t> full = fullColumns(blocks_->size(i));
        for (bool stiffness : {true, false}) {
            auto* diag = slot_(reducedKey_(stiffness, i, i));
            if (!diag || !diag->matrix.setLayout(full)) {
                AMLS_LOG_ERROR("planColumns 失败：{} 对角块 ({},{}) 缺失或布局被拒。",
                               stiffness ? "K" : "M", i, i);
                return false;
            }
        }
    }

    // 耦合块：rk = 原始 K(j,k) 列 ∪ 孩子 RK(c,k) 列；rm = 原始 M(j,k) 列 ∪
    // 孩子 (RM∪RK)(c,k) 列。postOrder 保证孩子布局先行就绪。
    for (int64_t j : order) {
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
            if (!kjk || !kjk->matrix.setLayout(kFull ? fullColumns(nk) : rk)) {
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
            if (!mjk || !mjk->matrix.setLayout(mFull ? fullColumns(nk) : rm)) {
                AMLS_LOG_ERROR(
                    "planColumns 失败：M 耦合块 ({},{}) 缺失或布局被拒 ({} 列, "
                    "n_k={})。",
                    j, k, mFull ? nk : static_cast<int64_t>(rm.size()), nk);
                return false;
            }
        }
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
        for (int64_t k : anc) {
            if (k < j) continue;  // 目标 (j,k)：j ≤ k，k ∈ A_j ∪ {j}
            ColCompMat* source = nullptr;
            {
                Entry* entry = slot_(reducedKey_(true, nodeI, k));
                if (!entry) {
                    AMLS_LOG_ERROR(
                        "accumulateReducedStiffness 失败：K 源块 ({},{}) 槽位缺失。",
                        nodeI, k);
                    return false;
                }
                if (!ensureReduced_(true, nodeI, k, *entry, source)) return false;
            }
            Entry* targetEntry = slot_(reducedKey_(true, j, k));
            if (!targetEntry) {
                AMLS_LOG_ERROR("accumulateReducedStiffness 失败：K 目标块 ({},{}) "
                               "槽位缺失。",
                               j, k);
                return false;
            }
            std::lock_guard<std::mutex> targetLock(targetEntry->mtx);
            ColCompMat* target = nullptr;
            if (!ensureReduced_(true, j, k, *targetEntry, target)) return false;
            if (!opsColAcc(useGemv_, *psi, true, *source, *target)) {
                AMLS_LOG_ERROR(
                    "accumulateReducedStiffness 失败：K̃ ({},{}) += Ψ ({},{})ᵀ·"
                    "K̃ ({},{}) 维度不匹配。",
                    j, k, nodeI, j, nodeI, k);
                return false;
            }
        }
    }
    return true;
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
        for (int64_t k : anc) {
            if (k < j) continue;
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
                if (!ensureReduced_(false, nodeI, k, *entry, source)) return false;
            }
            Entry* targetEntry = slot_(reducedKey_(false, j, k));
            if (!targetEntry) {
                AMLS_LOG_ERROR(
                    "accumulateReducedMass 失败：M 目标块 ({},{}) 槽位缺失。", j, k);
                return false;
            }
            std::lock_guard<std::mutex> targetLock(targetEntry->mtx);
            ColCompMat* target = nullptr;
            if (!ensureReduced_(false, j, k, *targetEntry, target)) return false;
            if (!opsColAcc(useGemv_, *psiJ, true, *source, *target) ||
                !opsColAcc(useGemv_, *mtemp, true, *psiK, *target)) {
                AMLS_LOG_ERROR(
                    "accumulateReducedMass 失败：M̃ ({},{}) += Ψ ({},{})ᵀ·M̃ "
                    "({},{}) + M_temp ({},{})ᵀ·Ψ ({},{}) 维度不匹配。",
                    j, k, nodeI, j, nodeI, k, nodeI, j, nodeI, k);
                return false;
            }
        }
    }
    return true;
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

} // namespace aams
