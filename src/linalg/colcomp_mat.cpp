#include "linalg/colcomp_mat.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "core/logging.hpp"

namespace amls {

std::vector<int64_t> ColCompMat::activeColList() const {
    std::vector<int64_t> result;
    result.reserve(static_cast<size_t>(nnzCount_ > 0 ? nnzCount_ : 0));
    for (int64_t s = 0; s < validCols(); ++s)
        if (colActive(s)) result.push_back(origColAt(s));
    return result;
}

void ColCompMat::releaseNumeric() {
    if (lifecycle_ != Lifecycle::Numeric) return;
    dense_ = Dense();
    lifecycle_ = Lifecycle::Released;
}

std::vector<int64_t> ColCompMat::storedCols() const {
    std::vector<int64_t> result;
    result.reserve(state_.size());
    for (int64_t s = 0; s < validCols(); ++s) result.push_back(origColAt(s));
    return result;
}

void ColCompMat::markColNonZero(const std::vector<int64_t>& cols) {
    for (int64_t c : cols) markColNonZero(c);
}

bool ColCompMat::setLayout(const std::vector<int64_t>& cols) {
    if (layoutReady()) {
        const bool same = storedCols() == cols;
        if (!same) {
            AMLS_LOG_ERROR(
                "[ccm] setLayout 失败：布局已固定 (rows={}, totalCols={}, "
                "validCols={})，拒绝新列数 {}。",
                rows_, totalCols_, validCols(), cols.size());
        }
        return same;
    }
    if (rows_ < 0 || totalCols_ < 0) {
        AMLS_LOG_ERROR("[ccm] setLayout 失败：负逻辑形状 (rows={}, totalCols={})。",
                       rows_, totalCols_);
        return false;
    }
    for (size_t s = 0; s < cols.size(); ++s) {
        if (cols[s] < 0 || cols[s] >= totalCols_ ||
            (s && cols[s - 1] >= cols[s])) {
            AMLS_LOG_ERROR(
                "[ccm] setLayout 失败：第 {} 列非法或非升序 (col={}, prev={}, "
                "rows={}, totalCols={})。",
                s, cols[s], s ? cols[s - 1] : -1, rows_, totalCols_);
            return false;
        }
    }
    std::vector<int32_t> state;
    state.reserve(cols.size());
    for (int64_t c : cols) state.push_back(static_cast<int32_t>(-(c + 1)));
    state_ = std::move(state);
    lifecycle_ = Lifecycle::Layout;
    return true;
}

bool ColCompMat::initializeFromSparse(const Sparse& src) {
    if (numericReady()) return true;
    if (numericReleased()) {
        AMLS_LOG_ERROR(
            "[ccm] initializeFromSparse 失败：数值已释放，禁止重建 "
            "(rows={}, totalCols={})。",
            rows_, totalCols_);
        return false;
    }
    if (!layoutReady() || src.rows() != rows_ || src.cols() != totalCols_) {
        AMLS_LOG_ERROR(
            "[ccm] initializeFromSparse 失败：布局或形状不匹配 (块: {}x{}, "
            "layoutReady={}, validCols={}; src: {}x{})。",
            rows_, totalCols_, layoutReady(), validCols(), src.rows(),
            src.cols());
        return false;
    }
    Dense data = Dense::Zero(rows_, validCols());
    for (int64_t c = 0; c < totalCols_; ++c) {
        const int64_t sc = storageCol(c);
        if (sc < 0) continue;  // 布局外列：原始数据不进缓冲
        for (Sparse::InnerIterator it(src, static_cast<int>(c)); it; ++it)
            data(static_cast<int64_t>(it.row()), sc) += it.value();
    }
    dense_ = std::move(data);
    lifecycle_ = Lifecycle::Numeric;
    for (int64_t c = 0; c < totalCols_; ++c) {
        const int64_t sc = storageCol(c);
        if (sc < 0) continue;
        for (Sparse::InnerIterator it(src, static_cast<int>(c)); it; ++it) {
            (void)it;
            markColNonZero(c);
            break;  // 该列有非零元即激活一次
        }
    }
    return true;
}

bool ColCompMat::initializeZero() {
    if (numericReady()) return true;
    if (!layoutReady() || numericReleased()) {
        AMLS_LOG_ERROR(
            "[ccm] initializeZero 失败：布局未就绪或数值已释放 (rows={}, "
            "totalCols={}, lifecycle={})。",
            rows_, totalCols_, static_cast<int>(lifecycle_));
        return false;
    }
    dense_ = Dense::Zero(rows_, validCols());
    lifecycle_ = Lifecycle::Numeric;
    return true;
}

void ColCompMat::inflate(Dense& out) const {
    if (!numericReady())
        throw std::logic_error("ColCompMat numeric data is not initialized");
    Dense result = Dense::Zero(rows_, totalCols_);
    for (int64_t s = 0; s < validCols(); ++s)
        result.col(origColAt(s)) = dense_.col(s);
    out = std::move(result);
}

bool ColCompMat::initializeConstraintFromRhs(const ColCompMat& rhs) {
    if (!rhs.numericReady()) {
        AMLS_LOG_ERROR(
            "[ccm] initializeConstraintFromRhs 失败：rhs 未数值化 "
            "(rhs: {}x{}, validCols={}, activeCols={})。",
            rhs.rows_, rhs.totalCols_, rhs.validCols(), rhs.activeCols());
        return false;
    }
    ColCompMat result(rhs.rows_, rhs.totalCols_);
    std::vector<int64_t> cols;
    for (int64_t s = 0; s < rhs.validCols(); ++s)
        if (rhs.colActive(s)) cols.push_back(rhs.origColAt(s));
    result.setLayout(cols);
    result.dense_ = Dense::Zero(rhs.rows_, result.validCols());
    result.lifecycle_ = Lifecycle::Numeric;
    for (int64_t s = 0; s < result.validCols(); ++s) {
        const int64_t source = rhs.storageCol(result.origColAt(s));
        result.dense_.col(s) = rhs.dense_.col(source);
    }
    result.markColNonZero(cols);
    *this = std::move(result);
    return true;
}

bool ColCompMat::initializeMtempFromInputs(const ColCompMat& mass,
                                           const ColCompMat& psi) {
    if (!mass.numericReady() || !psi.numericReady() ||
        mass.rows_ != psi.rows_ || mass.totalCols_ != psi.totalCols_) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMtempFromInputs 失败：输入未就绪或形状不匹配 "
            "(mass: {}x{}, validCols={}, numericReady={}; psi: {}x{}, "
            "validCols={}, numericReady={})。",
            mass.rows_, mass.totalCols_, mass.validCols(), mass.numericReady(),
            psi.rows_, psi.totalCols_, psi.validCols(), psi.numericReady());
        return false;
    }
    ColCompMat result(mass.rows_, mass.totalCols_);
    std::vector<int64_t> cols;
    int64_t m = 0, p = 0;
    while (m < mass.validCols() || p < psi.validCols()) {
        while (m < mass.validCols() && !mass.colActive(m)) ++m;
        while (p < psi.validCols() && !psi.colActive(p)) ++p;
        if (m == mass.validCols() && p == psi.validCols()) break;
        const int64_t mc =
            m < mass.validCols() ? mass.origColAt(m) : mass.totalCols_;
        const int64_t pc = p < psi.validCols() ? psi.origColAt(p) : psi.totalCols_;
        cols.push_back(std::min(mc, pc));
        if (mc <= pc) ++m;
        if (pc <= mc) ++p;
    }
    result.setLayout(cols);
    result.dense_ = Dense::Zero(mass.rows_, result.validCols());
    result.lifecycle_ = Lifecycle::Numeric;
    for (int64_t s = 0; s < mass.validCols(); ++s) {
        if (!mass.colActive(s)) continue;
        const int64_t c = mass.origColAt(s), dest = result.storageCol(c);
        result.dense_.col(dest) = mass.dense_.col(s);
        result.markColNonZero(c);
    }
    *this = std::move(result);
    return true;
}

bool ColCompMat::initializeMuFromInputs(
    int64_t rows, const ColCompMat& mtemp,
    const std::vector<int32_t>* reservedColState) {
    if (layoutReady()) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMuFromInputs 失败：目标已布局 (rows={}, "
            "totalCols={})。",
            rows, totalCols_);
        return false;
    }
    if (rows < 0) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMuFromInputs 失败：负行数 {} (mtemp: {}x{}, "
            "validCols={})。",
            rows, mtemp.rows_, mtemp.totalCols_, mtemp.validCols());
        return false;
    }
    if (this == &mtemp) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMuFromInputs 失败：目标与 mtemp 别名 (rows={})。",
            rows);
        return false;
    }
    if (!mtemp.numericReady()) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMuFromInputs 失败：mtemp 未数值化 (mtemp: {}x{}, "
            "validCols={})。",
            mtemp.rows_, mtemp.totalCols_, mtemp.validCols());
        return false;
    }
    const int64_t nCols = mtemp.totalCols_;
    // 列集 = mtemp 实际有效列 ∪ reservedColState 解码列（两侧均升序，单次归并）
    std::vector<int64_t> cols;
    int64_t s = 0;
    size_t r = 0;
    int64_t prevReserved = -1;
    while (s < mtemp.validCols() ||
           (reservedColState && r < reservedColState->size())) {
        while (s < mtemp.validCols() && !mtemp.colActive(s)) ++s;
        const int64_t ca =
            s < mtemp.validCols() ? mtemp.origColAt(s) : nCols;
        int64_t cb = nCols;
        if (reservedColState && r < reservedColState->size()) {
            const int64_t v = (*reservedColState)[r];
            if (v == 0) {
                AMLS_LOG_ERROR(
                    "[ccm] initializeMuFromInputs 失败：预留状态项 0 非法 "
                    "(index={}, rows={}, nCols={})。",
                    r, rows, nCols);
                return false;
            }
            cb = (v > 0 ? v : -v) - 1;
            if (cb < 0 || cb >= nCols || cb <= prevReserved) {
                AMLS_LOG_ERROR(
                    "[ccm] initializeMuFromInputs 失败：预留状态项 {} 解码越界"
                    "或非升序 (col={}, prevReserved={}, nCols={}, rows={})。",
                    v, cb, prevReserved, nCols, rows);
                return false;
            }
        }
        if (ca >= nCols && cb >= nCols) break;
        cols.push_back(std::min(ca, cb));
        if (ca <= cb) ++s;
        if (cb <= ca) {
            prevReserved = cb;  // 仅在真正消费时才推进升序校验基准
            ++r;
        }
    }
    rows_ = rows;
    totalCols_ = nCols;
    if (!setLayout(cols)) {
        AMLS_LOG_ERROR(
            "[ccm] initializeMuFromInputs 失败：setLayout 拒绝 {} 合并列 "
            "(rows={}, nCols={})。",
            cols.size(), rows, nCols);
        return false;
    }
    dense_ = Dense::Zero(rows_, validCols());
    lifecycle_ = Lifecycle::Numeric;
    return true;
}

} // namespace amls
