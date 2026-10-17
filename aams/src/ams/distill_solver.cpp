#include "ams/distill_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace aams {
namespace rea {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;
// 4π²：Abaqus cutoff 公式 dVar9 = fmax·39.478417604357354·fmax + shift
constexpr double kFourPiSq = 39.478417604357354;

bool dumpOn() { return std::getenv("AMLS_DUMP_DISTILL") != nullptr; }

// 取 (K, M) 的列子集（keep 为升序列索引）。
void extractSub(const DefaultTraits::DenseMatrix& K,
                const DefaultTraits::DenseMatrix& M,
                const std::vector<int64_t>& keep,
                DefaultTraits::DenseMatrix& Ks,
                DefaultTraits::DenseMatrix& Ms) {
    const int64_t p2 = static_cast<int64_t>(keep.size());
    Ks.resize(p2, p2);
    Ms.resize(p2, p2);
    for (int64_t i = 0; i < p2; ++i)
        for (int64_t j = 0; j < p2; ++j) {
            Ks(i, j) = K(keep[static_cast<size_t>(i)],
                         keep[static_cast<size_t>(j)]);
            Ms(i, j) = M(keep[static_cast<size_t>(i)],
                         keep[static_cast<size_t>(j)]);
        }
}

// M-正交归一：G = Yᵀ M Y，Y ← Y·G^{-1/2}（LLT 主路，失败回退特征分解去秩亏）。
// 返回 false 表示 Y 列空间完全退化（调用方应报错/返回空）。
bool mOrthoNormalize(DefaultTraits::DenseMatrix& Y,
                     const DefaultTraits::DenseMatrix& M) {
    if (Y.cols() == 0) return true;
    DefaultTraits::DenseMatrix G =
        DefaultTraits::DenseMatrix(Y.transpose() * M * Y);
    G = 0.5 * (G + G.transpose()).eval();
    Eigen::LLT<DefaultTraits::DenseMatrix> llt(G);
    if (llt.info() == Eigen::Success) {
        // Y ← Y·L^{-T} ⟺ 解 L·X = Yᵀ 再转置（X = L^{-1}Yᵀ, Xᵀ = Y·L^{-T}）
        DefaultTraits::DenseMatrix Yt =
            llt.matrixL().solve(Y.transpose());
        Y = Yt.transpose();
        return true;
    }
    // 回退：G 特征分解，丢弃近零方向（秩亏）
    Eigen::SelfAdjointEigenSolver<DefaultTraits::DenseMatrix> es(G);
    if (es.info() != Eigen::Success) return false;
    const auto& w = es.eigenvalues();
    const double wmax = w.size() > 0 ? w(w.size() - 1) : 0.0;
    if (!(wmax > 0.0)) return false;
    const double tol = 1e-12 * wmax;
    std::vector<int64_t> keep;
    for (int64_t i = 0; i < w.size(); ++i)
        if (w(i) > tol) keep.push_back(i);
    if (keep.empty()) return false;
    DefaultTraits::DenseMatrix Q(Y.rows(),
                                 static_cast<int64_t>(keep.size()));
    for (int64_t k = 0; k < static_cast<int64_t>(keep.size()); ++k) {
        const int64_t i = keep[static_cast<size_t>(k)];
        Q.col(k) = es.eigenvectors().col(i) / std::sqrt(w(i));
    }
    Y = (Y * Q).eval();
    if (dumpOn())
        std::fprintf(stderr,
                     "[distill] M-正交化 LLT 失败, 特征分解去秩亏: %lld → %lld\n",
                     (long long)w.size(), (long long)keep.size());
    return true;
}

}  // namespace

DistillResult distillRitz(const DefaultTraits::DenseMatrix& Kr,
                          const DefaultTraits::DenseMatrix& Mr,
                          const DistillParams& p) {
    DistillResult res;
    const int64_t n = Kr.rows();
    res.inputSize = n;
    if (n == 0 || Mr.rows() != n) return res;

    // ---- 阈值（Abaqus pme_SolverParameters 语义）----
    const double dVar9 = kFourPiSq * p.fmaxHz * p.fmaxHz + p.shift;
    const double lamDistill = p.cutoff2 * p.cutoff2 * dVar9;   // 蒸馏阈值 λ2
    const double lamInit = p.cutoff3 * p.cutoff3 * dVar9;      // 初始子空间 λ3
    const double bandLo =
        p.fminHz > 0.0 ? kFourPiSq * p.fminHz * p.fminHz : -1e300;
    const double bandHi = kFourPiSq * p.fmaxHz * p.fmaxHz;

    // ---- 1. 蒸馏过滤（含初始子空间过滤，取两者较紧）----
    // Rayleigh 商 ρ_i = Kr(i,i)/Mr(i,i)；缩减基下对角商即该基向量的谱坐标。
    // 阈值过滤后按 pme_SolverParameters[0xa0]=3000（子空间迭代规模上限 [R]）
    // 用最低 ρ 的剩余向量补足——Abaqus 的迭代子空间由蒸馏保留 + 规模上限
    // 共同决定，而非仅阈值（纯阈值过滤会把起始子空间削得过穷）。
    const double lamKeep = std::min(lamDistill, lamInit);
    std::vector<std::pair<double, int64_t>> rhoAll;
    rhoAll.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double mii = Mr(i, i);
        const double rho = mii > 0.0 ? Kr(i, i) / mii
                                     : std::numeric_limits<double>::infinity();
        rhoAll.emplace_back(rho, i);
    }
    const int64_t cap = std::min(p.subspaceMax, n);
    std::vector<int64_t> keep;
    keep.reserve(static_cast<size_t>(cap));
    for (const auto& [rho, i] : rhoAll)
        if (rho <= lamKeep) keep.push_back(i);
    if (static_cast<int64_t>(keep.size()) < cap) {
        std::vector<std::pair<double, int64_t>> rest;
        for (const auto& [rho, i] : rhoAll)
            if (rho > lamKeep) rest.push_back({rho, i});
        std::sort(rest.begin(), rest.end());
        for (const auto& [rho, i] : rest) {
            if (static_cast<int64_t>(keep.size()) >= cap) break;
            keep.push_back(i);
        }
    }
    std::sort(keep.begin(), keep.end());
    res.distilledSize = static_cast<int64_t>(keep.size());
    if (dumpOn())
        std::fprintf(stderr,
                     "[distill] p=%lld λ2=%.6e λ3=%.6e 蒸馏后 p'=%lld\n",
                     (long long)n, lamDistill, lamInit,
                     (long long)res.distilledSize);
    if (keep.empty()) return res;

    DefaultTraits::DenseMatrix K2, M2;
    extractSub(Kr, Mr, keep, K2, M2);
    const int64_t p2 = res.distilledSize;

    // ---- 2. 一次子空间迭代 + Ritz ----
    DefaultTraits::Vector lam;
    DefaultTraits::DenseMatrix V;   // p2×k，M2 正交归一
    if (p.subspaceIteration) {
        // Y = (K2 − σM2)⁻¹ M2（Abaqus OneSubspaceIterationWithRigidBodyModes）
        // σ = shift；奇异/非正定时 escalation（PerturbDiagonal 语义）。
        double sigma = p.shift;
        DefaultTraits::DenseMatrix Y;
        bool factored = false;
        for (int attempt = 0; attempt < 6 && !factored; ++attempt) {
            DefaultTraits::DenseMatrix S = K2 - sigma * M2;
            Eigen::LDLT<DefaultTraits::DenseMatrix> ldlt(S);
            if (ldlt.info() == Eigen::Success) {
                Y = ldlt.solve(M2);
                factored = true;
            } else {
                const double scale =
                    attempt == 0 ? 1e-10 : 100.0;
                sigma = attempt == 0
                            ? 1e-10 * std::max(1.0, std::abs(lamDistill))
                            : sigma * scale;
                if (dumpOn())
                    std::fprintf(stderr,
                                 "[distill] LDLT 失败, shift escalation σ=%.6e\n",
                                 sigma);
            }
        }
        if (!factored)
            throw std::runtime_error(
                "distillRitz: 子空间迭代 (K−σM) 分解失败（shift escalation 用尽）");
        // M2 正交归一（Gram-Schmidt 主路 LLT）
        if (!mOrthoNormalize(Y, M2))
            throw std::runtime_error("distillRitz: 子空间退化（M 正交化失败）");
        // Ritz 分析：K̂ = YᵀK2Y（M̂ ≈ I，标准特征问题）
        DefaultTraits::DenseMatrix Kh =
            DefaultTraits::DenseMatrix(Y.transpose() * K2 * Y);
        Kh = 0.5 * (Kh + Kh.transpose()).eval();
        Eigen::SelfAdjointEigenSolver<DefaultTraits::DenseMatrix> es(Kh);
        if (es.info() != Eigen::Success)
            throw std::runtime_error("distillRitz: Ritz 特征求解失败");
        lam = es.eigenvalues();
        V = (Y * es.eigenvectors()).eval();
    } else {
        // 无子空间迭代：直接对 (K2, M2) 做广义 Ritz（dsygvd）
        Eigen::GeneralizedSelfAdjointEigenSolver<DefaultTraits::DenseMatrix>
            ges(K2, M2);
        if (ges.info() != Eigen::Success)
            throw std::runtime_error("distillRitz: 广义特征求解失败");
        lam = ges.eigenvalues();
        V = ges.eigenvectors();
    }

    // ---- 3. TruncateEigenSolution：频段下界剔除 + 上界过滤 + nev 截断 ----
    std::vector<int64_t> band;
    for (int64_t i = 0; i < lam.size(); ++i)
        if (lam(i) > bandLo && lam(i) <= bandHi) band.push_back(i);
    int64_t m = static_cast<int64_t>(band.size());
    if (p.nev > 0 && m > p.nev) m = p.nev;
    if (p.extractOrder > 0 && m > p.extractOrder) m = p.extractOrder;
    if (dumpOn())
        std::fprintf(stderr,
                     "[distill] 频段 (%g, %g] 内 %lld 模态, 截断后 %lld\n",
                     bandLo, bandHi, (long long)band.size(), (long long)m);
    if (m == 0) {
        res.distilledSize = p2;
        return res;
    }
    res.eigenvalues.resize(m);
    DefaultTraits::DenseMatrix Vb(p2, m);
    for (int64_t k = 0; k < m; ++k) {
        res.eigenvalues(k) = lam(band[static_cast<size_t>(k)]);
        Vb.col(k) = V.col(band[static_cast<size_t>(k)]);
    }

    // ---- 4. 块 Gram-Schmidt M2 正交归一（数值保险一遍）----
    if (!mOrthoNormalize(Vb, M2))
        throw std::runtime_error("distillRitz: 最终 M 归一失败");

    // ---- 5. 嵌回全 p 维（剔除列补零），对齐 PKU 恢复路径的子空间偏移 ----
    res.eigenvectors.setZero(n, m);
    for (int64_t k = 0; k < m; ++k)
        for (int64_t i = 0; i < p2; ++i)
            res.eigenvectors(keep[static_cast<size_t>(i)], k) = Vb(i, k);
    return res;
}

}  // namespace rea
}  // namespace aams
