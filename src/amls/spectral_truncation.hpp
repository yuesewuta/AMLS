#pragma once
// 截断策略（设计文档 §4.4,可插拔;P9：局部谱求解按区间/前 k 阶,不允许全谱后截断）。
#include <cstdint>

#include "core/types.hpp"

namespace amls {

template <typename T>
class ISpectralTruncation {
public:
    using Vector = typename T::Vector;
    using Index  = typename T::Index;

    /// 给定局部特征值（升序）与保留上限 maxKeep,返回保留个数 kᵢ ∈ [0, maxKeep]。
    virtual Index select(const Vector& localEigenvalues,
                         Index maxKeep) const = 0;

    /// 区间提示：若策略支持"直接解值区间",返回 true 并给出 (lo, hi]。
    /// Transformer 据此走 solveInterval,避免全谱（P9）。
    virtual bool intervalHint(double& lo, double& hi) const {
        (void)lo; (void)hi;
        return false;
    }

    /// 前 k 阶提示（1-based [il, iu]）：固定数目策略用。
    virtual bool indexHint(Index& il, Index& iu) const {
        (void)il; (void)iu;
        return false;
    }
    virtual ~ISpectralTruncation() = default;
};

/// 阈值策略：保留 λ ≤ λ_cut（构造时给定）。
template <typename T>
class ThresholdTruncation final : public ISpectralTruncation<T> {
public:
    using Vector = typename T::Vector;
    using Index  = typename T::Index;

    explicit ThresholdTruncation(double lambdaCut) : lambdaCut_(lambdaCut) {}

    Index select(const Vector& w, Index maxKeep) const override {
        Index k = 0;
        while (k < maxKeep && k < w.size() && w[k] <= lambdaCut_)
            ++k;
        return k;
    }

    bool intervalHint(double& lo, double& hi) const override {
        lo = -1e300;
        hi = lambdaCut_;
        return true;
    }

private:
    double lambdaCut_;
};

/// 固定数目策略：每节点保留前 k 个（k=0 时保留全部）。
template <typename T>
class FixedCountTruncation final : public ISpectralTruncation<T> {
public:
    using Vector = typename T::Vector;
    using Index  = typename T::Index;

    explicit FixedCountTruncation(Index k) : k_(k) {}

    Index select(const Vector& w, Index maxKeep) const override {
        if (k_ <= 0)
            return maxKeep;
        return k_ < maxKeep ? k_ : maxKeep;
    }

    /// 前 k 阶提示：走 dsygvx 索引区间。
    bool indexHint(Index& il, Index& iu) const override {
        if (k_ <= 0)
            return false;
        il = 1;
        iu = k_;
        return true;
    }

    Index k() const { return k_; }

private:
    Index k_;
};

} // namespace amls
