#pragma once
// 移位反演 Lanczos（SIL）：稀疏 pencil 局部谱的替代内核（Gao TOMS'08 §4.2）。
//
// C = (K − σM)⁻¹·M 在 M-内积下自伴，用 M-内积 Lanczos 保持三对角对称性：
//   α_j = q_jᵀMw, z = w − Σ(q_lᵀMw)q_l, β_j = ‖z‖_M, w = Cq_j
// Ritz 对 λ = 1/ν+σ，Ritz 向量 x = Qs（基已 M-正交，无需再归一）。
// 适用场景（实测 crossover，debug/t_stats.cpp）：铅笔稀疏（nnz/n²≲15%）
// 且 n≳200 时比稠密 dsygvx 快 5–20×；稠密填充铅笔应走稠密路径。
//
// σ 取 0 左侧小量：K 半正定时 K−σM 亦 SPD（σ<0），LDLᵀ 无主元分解安全。
// 线程安全：无共享状态，可在节点级线程池中并发调用。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "core/types.hpp"
#include "linalg/sparse_direct.hpp"

namespace amls {
namespace sil {

using SM = DefaultTraits::SparseMatrix;
using DM = DefaultTraits::DenseMatrix;
using Vec = DefaultTraits::Vector;

/// 持有 K−σM 因子的一次 SIL 求解（M-内积 Lanczos）。
class ShiftInvertLanczos {
public:
    /// A = K − σM 分解（SPD 要求）。
    ShiftInvertLanczos(const DM& Kd, const DM& Md, double sigma)
        : n_(Kd.rows()), shift_(sigma) {
        SM A = Kd.sparseView(1e-14);
        M_ = Md.sparseView(1e-14);
        A -= sigma * M_;
        fact_.factorize(A);
    }

    /// 跑至多 ncv 步 M-内积全重正交 Lanczos。Ritz 谱经 nu()/res() 取用。
    int64_t iterate(int64_t ncv, unsigned seed = 12345u) {
        ncv = std::min<int64_t>(ncv, n_);
        alpha_.assign(static_cast<size_t>(ncv), 0.0);
        beta_.assign(static_cast<size_t>(ncv), 0.0);
        Q_.resize(static_cast<size_t>(ncv));
        std::srand(seed);
        Q_[0] = Vec::Random(n_);
        // M-归一
        double mq = std::sqrt(std::max(Q_[0].dot(M_ * Q_[0]), 1e-300));
        Q_[0] /= mq;
        int64_t m = 0;
        for (int64_t j = 0; j < ncv; ++j) {
            Vec w;
            applyC_(Q_[static_cast<size_t>(j)], w);
            const double aj = Q_[static_cast<size_t>(j)].dot(M_ * w);
            alpha_[static_cast<size_t>(j)] = aj;
            // 全重正交（含 j：α 已取出，z 与 q_j 也去耦）
            for (int t = 0; t < 2; ++t)
                for (int64_t l = 0; l <= j; ++l)
                    w -= Q_[static_cast<size_t>(l)].dot(M_ * w) *
                         Q_[static_cast<size_t>(l)];
            const double bj = std::sqrt(std::max(w.dot(M_ * w), 0.0));
            m = j + 1;
            if (bj < 1e-13 || m == ncv)
                break;
            beta_[static_cast<size_t>(j)] = bj;
            Q_[static_cast<size_t>(j + 1)] = w / bj;
        }
        m_ = m;
        DM T = DM::Zero(m, m);
        for (int64_t j = 0; j < m; ++j) {
            T(j, j) = alpha_[static_cast<size_t>(j)];
            if (j + 1 < m) {
                T(j, j + 1) = beta_[static_cast<size_t>(j)];
                T(j + 1, j) = beta_[static_cast<size_t>(j)];
            }
        }
        Eigen::SelfAdjointEigenSolver<DM> es(T);
        nu_ = es.eigenvalues();
        S_ = es.eigenvectors();
        return m;
    }

    int64_t m() const { return m_; }
    const Vec& nu() const { return nu_; }
    /// Ritz 值换算：λ = 1/ν + σ。
    double lambda(int64_t j) const { return 1.0 / nu_[j] + shift_; }
    /// M-范数残差估计 β_m·|s_mj|。
    double res(int64_t j) const {
        const double bm = m_ > 1 ? beta_[static_cast<size_t>(m_ - 1)] : 0.0;
        return std::abs(bm * S_(m_ - 1, j));
    }
    /// Ritz 向量 x = Σ s_lj q_l（Q 已 M-正交 ⇒ xᵀMx = 1）。
    Vec ritzVector(int64_t j) const {
        Vec x = Vec::Zero(n_);
        for (int64_t l = 0; l < m_; ++l)
            x += S_(l, j) * Q_[static_cast<size_t>(l)];
        return x;
    }

private:
    void applyC_(const Vec& x, Vec& y) const {
        const Vec Mx = M_ * x;
        DM B(n_, 1);
        B.col(0) = Mx;
        DM X;
        fact_.solveWithFactor(B, X);
        y = X.col(0);
    }

    int64_t n_ = 0, m_ = 0;
    double shift_ = 0.0;
    SM M_;
    EigenSimplicialLDLTSolver<DefaultTraits> fact_;
    std::vector<Vec> Q_;
    std::vector<double> alpha_, beta_;
    Vec nu_;
    DM S_;
};

/// 区间 (lo, hi] 内全部 Ritz 对（升序，残差 ≤ tol·νscale 的收敛对）。
/// ncv 自适应倍增（因子分解只做一次），直到出现 >hi 的 Ritz 且边界收敛。
inline void solveInterval(const DM& Kd, const DM& Md, double lo, double hi,
                          Vec& wOut, DM& VOut, double tol = 1e-9) {
    const int64_t n = Kd.rows();
    double scale = 0.0;
    for (int64_t j = 0; j < n; ++j)
        scale = std::max(scale, std::abs(Kd(j, j)));
    // σ = min(0, lo) − ε·scale；lo = −1e300 一类的"负无穷"哨兵截断到 0 侧
    const double loSane = (lo > -1e100) ? lo : 0.0;
    const double sigma = std::min(0.0, loSane) -
                         std::max(1e-10, 1e-9 * (scale > 0.0 ? scale : 1.0));
    ShiftInvertLanczos lz(Kd, Md, sigma);
    int64_t ncv = std::min<int64_t>(n, 96);
    const int64_t cap = std::min<int64_t>(n, 1536);
    for (;;) {
        const int64_t m = lz.iterate(ncv);
        int64_t cnt = 0;
        for (int64_t j = 0; j < m; ++j)
            if (lz.lambda(j) <= hi)
                ++cnt;
        if (cnt < m && cnt > 0 &&
            lz.res(cnt - 1) <= tol * std::max(1.0, lz.nu()[m - 1]))
            break;   // 已见区间以上模式且边界收敛
        if (ncv >= cap || m < ncv)
            break;
        ncv = std::min<int64_t>(cap, ncv * 2);
    }
    std::vector<int64_t> pick;
    for (int64_t j = 0; j < lz.m(); ++j) {
        const double lam = lz.lambda(j);
        if (lam > hi || lam <= lo)
            continue;
        if (lz.res(j) > 100 * tol * std::max(1.0, lz.nu()[lz.m() - 1]))
            continue;   // 未收敛的丢弃
        pick.push_back(j);
    }
    const int64_t kk = static_cast<int64_t>(pick.size());
    wOut.resize(kk);
    VOut = DM::Zero(n, kk);
    for (int64_t t = 0; t < kk; ++t) {
        wOut[t] = lz.lambda(pick[static_cast<size_t>(t)]);
        VOut.col(t) = lz.ritzVector(pick[static_cast<size_t>(t)]);
    }
}

/// 索引区间 [il, iu]（1-based，仅 il==1）：最小 k 个特征对（升序）。
inline void solveIndex(const DM& Kd, const DM& Md, int64_t il, int64_t iu,
                       Vec& wOut, DM& VOut, double tol = 1e-10) {
    const int64_t n = Kd.rows();
    const int64_t k = iu - il + 1;
    double scale = 0.0;
    for (int64_t j = 0; j < n; ++j)
        scale = std::max(scale, std::abs(Kd(j, j)));
    const double sigma = -std::max(1e-10, 1e-9 * (scale > 0.0 ? scale : 1.0));
    ShiftInvertLanczos lz(Kd, Md, sigma);
    const int64_t cap = std::min<int64_t>(n, 1536);
    int64_t ncv = std::min<int64_t>(n, std::max<int64_t>(4 * k + 16, 48));
    // 不足收敛则扩 ncv 重跑（因子复用）
    for (;;) {
        const int64_t m = lz.iterate(ncv);
        const int64_t want = std::min<int64_t>(k, m);
        bool ok = true;
        for (int64_t t = 0; t < want; ++t)
            if (lz.res(m - 1 - t) > tol * std::max(1.0, lz.nu()[m - 1]))
                ok = false;
        if (ok || ncv >= cap || m < ncv)
            break;
        ncv = std::min<int64_t>(cap, ncv * 2);
    }
    const int64_t m = lz.m();
    const int64_t kk = std::min<int64_t>(k, m);
    wOut.resize(kk);
    VOut = DM::Zero(n, kk);
    for (int64_t t = 0; t < kk; ++t) {
        const int64_t j = m - 1 - t;   // ν 降序 = λ 升序
        wOut[t] = lz.lambda(j);
        VOut.col(t) = lz.ritzVector(j);
    }
}

} // namespace sil
} // namespace amls
