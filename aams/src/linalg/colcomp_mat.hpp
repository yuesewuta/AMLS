#pragma once
// 列压缩稠密矩阵 ColCompMat —— 移植自 third_party/amlsPKU utils/amls_block。
//
// ColCompMat（CCM）：逻辑形状 rows_ × totalCols_ 固定；仅存储 state_ 列出的
// 子集列，state_[s] = ±(原始列号 + 1)：正 = 已有非零数值的有效列，负 = 预留
// 占位列（数值恒零，布局内为其预留缓冲，写入时激活转正）。dense_ 为
// rows_ × validCols() 的列主序紧凑缓冲（Eigen 列主序，与 PKU MatXd 同布局）。
// 满列布局（validCols == totalCols）退化为恒等映射，storageCol O(1)。
//
// 生命周期单向：Unplanned → Layout → Numeric → Released。
// 布局规划与数值初始化分离：setLayout/initializeMuFromInputs 等只建布局，
// initializeFromSparse/initializeZero 按需数值化，releaseNumeric 释放数值
// 缓冲但保留 state_（供子空间缩聚 μ 查预留列），Released 后不可重建。
#include <cstdint>
#include <vector>

#include "core/types.hpp"

namespace aams {

class ColCompMat {
public:
    using Dense = DefaultTraits::DenseMatrix;
    using Sparse = DefaultTraits::SparseMatrix;

    ColCompMat() = default;
    ColCompMat(int64_t rows, int64_t cols) : rows_(rows), totalCols_(cols) {}

    /// 展开为 rows × totalCols 稠密（须 numericReady，否则抛异常）。
    void inflate(Dense& out) const;

    int64_t rows() const { return rows_; }
    int64_t totalCols() const { return totalCols_; }
    /// 存储列数（含预留占位列）。
    int64_t validCols() const { return static_cast<int64_t>(state_.size()); }
    /// 实际有效列数（正状态列）。
    int64_t activeCols() const { return nnzCount_; }
    const Dense& dense() const { return dense_; }
    Dense& dense() { return dense_; }
    /// 存储列占逻辑列的比例（1 - validCols/totalCols 为零列占比）。
    double zeroColRatio() const {
        return totalCols_ > 0
                   ? 1.0 - static_cast<double>(validCols()) /
                               static_cast<double>(totalCols_)
                   : 0.0;
    }
    /// 存储列 s 的原始（逻辑）列号。
    int64_t origColAt(int64_t s) const {
        const int64_t v = state_[static_cast<size_t>(s)];
        return (v > 0 ? v : -v) - 1;
    }
    /// 存储列 s 是否为有效列。
    bool colActive(int64_t s) const {
        return state_[static_cast<size_t>(s)] > 0;
    }
    /// 全部存储列的原始列号（升序，拷贝）。
    std::vector<int64_t> storedCols() const;
    /// 原始列号 → 存储列下标，不存在返回 -1；满列恒等 O(1)。
    int64_t storageCol(int64_t origCol) const {
        if (static_cast<int64_t>(state_.size()) == totalCols_)
            return (origCol >= 0 && origCol < totalCols_) ? origCol : -1;
        int64_t lo = 0, hi = validCols();
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (origColAt(mid) < origCol)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < validCols() && origColAt(lo) == origCol ? lo : -1;
    }
    /// 激活预留列（负转正），幂等；未数值化或列不在布局内则忽略。
    void markColNonZero(int64_t origCol) {
        const int64_t s = storageCol(origCol);
        if (!numericReady() || s < 0 || colActive(s)) return;
        state_[static_cast<size_t>(s)] *= -1;
        ++nnzCount_;
    }
    void markColNonZero(const std::vector<int64_t>& origCols);
    /// 按已知存储列号直接激活（负转正），供散加内核免重复二分；调用方
    /// 须保证 s 来自 storageCol 查得且矩阵已数值化（语义同 markColNonZero）。
    void activateStorageCol(int64_t s) {
        auto& v = state_[static_cast<size_t>(s)];
        if (v < 0) {
            v = -v;
            ++nnzCount_;
        }
    }
    /// Ψ 初始化：布局 = rhs 有效列，零缓冲（分解阶数为 0 时走此路径）。
    bool initializeConstraintFromRhs(const ColCompMat& rhs);
    /// M_temp 初始化：布局 = mass 与 psi 有效列并集，数值取 mass 对应列，
    /// 其余置零预留。
    bool initializeMtempFromInputs(const ColCompMat& massCoupling,
                                   const ColCompMat& psi);
    /// 仅建布局（升序列号，全负占位），不动数值；已布局时重入须同列集。
    bool setLayout(const std::vector<int64_t>& cols);
    /// 从原始稀疏块数值化（仅填布局内列；满对称存储；Released 拒绝）。
    bool initializeFromSparse(const Sparse& src);
    /// 零初始化数值缓冲（占位语义）。
    bool initializeZero();
    /// 布局已规划（Layout/Numeric/Released）。
    bool layoutReady() const { return lifecycle_ != Lifecycle::Unplanned; }
    /// 数值缓冲可用（Numeric）。
    bool numericReady() const { return lifecycle_ == Lifecycle::Numeric; }
    /// 数值已释放（Released）。
    bool numericReleased() const { return lifecycle_ == Lifecycle::Released; }
    /// 释放数值缓冲，保留 state_ 布局；幂等。
    void releaseNumeric();
    /// 有效列的原始列号列表（拷贝）。
    std::vector<int64_t> activeColList() const;
    /// state_ 只读视图（±(列+1) 编码，供 μ 预留列查表）。
    const std::vector<int32_t>& colState() const { return state_; }
    /// μ 子块布局初始化：列集 = mtemp 实际有效列 ∪ reservedColState（±(列+1)
    /// 编码）解码列，全部置为预留（负）状态，数值缓冲置零；rows 为 μ 行数
    /// （Φ_c 列数），列维度取自 mtemp。
    bool initializeMuFromInputs(int64_t rows, const ColCompMat& mtemp,
                                const std::vector<int32_t>* reservedColState);

private:
    enum class Lifecycle : std::uint8_t { Unplanned, Layout, Numeric, Released };
    Lifecycle lifecycle_ = Lifecycle::Unplanned;
    int64_t rows_ = 0;
    int64_t totalCols_ = 0;
    int64_t nnzCount_ = 0;    ///< 正状态列数（= activeCols）
    std::vector<int32_t> state_;  ///< ±(原始列号+1) 编码的存储列状态
    Dense dense_;             ///< rows_ × validCols() 列主序紧凑数值缓冲
};

} // namespace aams
