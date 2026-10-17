#include "ams/distill_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <Eigen/Dense>

#include "linalg/blas.hpp"
#include "linalg/lapack.h"

namespace aams {
namespace rea {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;
// 4π²：Abaqus cutoff 公式 dVar9 = fmax·39.478417604357354·fmax + shift
constexpr double kFourPiSq = 39.478417604357354;

bool dumpOn() { return std::getenv("AMLS_DUMP_DISTILL") != nullptr; }

// AAMS_PROFILE=1 时输出蒸馏子阶段耗时（性能剖析门控）。
bool profOn() { return std::getenv("AAMS_PROFILE") != nullptr; }
struct ProfScope {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    bool done = false;
    explicit ProfScope(const char* n)
        : name(n), t0(std::chrono::steady_clock::now()) {}
    void end() {
        if (done || !profOn()) { done = true; return; }
        done = true;
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[prof] distill/%s = %.3f s\n", name, s);
    }
    ~ProfScope() { end(); }
};

// 本阶段释放 OpenBLAS 多线程（Reduction 的线程池已销毁，无互踩）；
// 析构恢复单线程（工程全局约定 BLAS 单线程防过度订阅）。
#ifdef _WIN32
struct CpuWallProbe {
    FILETIME c0{}, e0{}, k0{}, u0{};
    std::chrono::steady_clock::time_point t0;
    CpuWallProbe() : t0(std::chrono::steady_clock::now()) {
        GetProcessTimes(GetCurrentProcess(), &c0, &e0, &k0, &u0);
    }
    ~CpuWallProbe() {
        if (!profOn()) return;
        FILETIME c1{}, e1{}, k1{}, u1{};
        GetProcessTimes(GetCurrentProcess(), &c1, &e1, &k1, &u1);
        auto q = [](const FILETIME& f) {
            ULARGE_INTEGER v;
            v.LowPart = f.dwLowDateTime;
            v.HighPart = f.dwHighDateTime;
            return static_cast<double>(v.QuadPart) * 1e-7;
        };
        const double cpu = q(k1) + q(u1) - q(k0) - q(u0);
        const double wall =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[prof] distill cpu=%.2fs wall=%.2fs (%.1fx)\n",
                     cpu, wall, cpu / (wall > 0 ? wall : 1));
    }
};
#endif

struct BlasThreadsGuard {
    explicit BlasThreadsGuard(int t)
        : on(t > 1) {
        if (on) {
            blas::setNumThreads(t);
            if (profOn())
                std::fprintf(stderr,
                             "[prof] distill blas threads=%d omp_max=%d\n", t,
                             blas::omp_get_max_threads());
        }
    }
    ~BlasThreadsGuard() {
        if (on) blas::setNumThreads(1);
    }
    bool on;
};

// 取 (K, M) 的列子集（keep 为升序列索引）。按行块拷贝（随机访问 gather）。
void extractSub(const DefaultTraits::DenseMatrix& K,
                const DefaultTraits::DenseMatrix& M,
                const std::vector<int64_t>& keep,
                DefaultTraits::DenseMatrix& Ks,
                DefaultTraits::DenseMatrix& Ms) {
    const int64_t p2 = static_cast<int64_t>(keep.size());
    Ks.resize(p2, p2);
    Ms.resize(p2, p2);
    for (int64_t i = 0; i < p2; ++i) {
        const int64_t r = keep[static_cast<size_t>(i)];
        for (int64_t j = 0; j < p2; ++j) {
            const int64_t c = keep[static_cast<size_t>(j)];
            Ks(i, j) = K(r, c);
            Ms(i, j) = M(r, c);
        }
    }
}

// M-正交归一：G = Yᵀ M Y（dgemm + dpotrf），Y ← Y·L^{-T}（dtrsm）。
// dpotrf 失败（秩亏）回退 dsyevd 特征分解丢近零方向。
// 返回 false 表示 Y 列空间完全退化（调用方应报错/返回空）。
bool mOrthoNormalize(DefaultTraits::DenseMatrix& Y,
                     const DefaultTraits::DenseMatrix& M) {
    if (Y.cols() == 0) return true;
    const int32_t n = static_cast<int32_t>(Y.rows());
    const int32_t k = static_cast<int32_t>(Y.cols());
    DefaultTraits::DenseMatrix MY, G;
    blas::gemm0(CblasNoTrans, CblasNoTrans, M, Y, MY);
    G.resize(k, k);
    blas::gemm(CblasTrans, CblasNoTrans, 1.0, Y, MY, 0.0, G);
    int32_t info = 0;
    dpotrf_("L", &k, G.data(), &k, &info);
    if (info == 0) {
        // Xt = L⁻¹·Yᵀ（dtrsm 原地解 L·Xt = Yᵀ）；Y ← Xtᵀ = Y·L⁻ᵀ
        DefaultTraits::DenseMatrix Yt = Y.transpose();
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, k, n, 1.0, G.data(), k, Yt.data(), k);
        Y = Yt.transpose();
        return true;
    }
    // 秩亏回退：G 特征分解，丢弃近零方向
    {
        DefaultTraits::DenseMatrix MY2;
        blas::gemm0(CblasNoTrans, CblasNoTrans, M, Y, MY2);
        blas::gemm(CblasTrans, CblasNoTrans, 1.0, Y, MY2, 0.0, G);
    }
    std::vector<double> w(static_cast<size_t>(k));
    double wk = 0.0;
    int32_t iwk = 0;
    int32_t lq = -1;
    dsyevd_("V", "L", &k, G.data(), &k, w.data(), &wk, &lq, &iwk, &lq, &info);
    if (info != 0) return false;
    std::vector<double> work(static_cast<size_t>(wk));
    std::vector<int32_t> iwork(static_cast<size_t>(iwk));
    const int32_t lw1 = static_cast<int32_t>(work.size());
    const int32_t liw1 = static_cast<int32_t>(iwork.size());
    dsyevd_("V", "L", &k, G.data(), &k, w.data(), work.data(),
            &lw1, iwork.data(), &liw1, &info);
    if (info != 0) return false;
    const double wmax = w.empty() ? 0.0 : w.back();
    if (!(wmax > 0.0)) return false;
    const double tol = 1e-12 * wmax;
    std::vector<int64_t> keepc;
    for (int32_t i = 0; i < k; ++i)
        if (w[static_cast<size_t>(i)] > tol) keepc.push_back(i);
    if (keepc.empty()) return false;
    DefaultTraits::DenseMatrix Q(Y.rows(),
                                 static_cast<int64_t>(keepc.size()));
    for (int64_t j = 0; j < static_cast<int64_t>(keepc.size()); ++j) {
        const int32_t i = static_cast<int32_t>(keepc[static_cast<size_t>(j)]);
        Q.col(j) = G.col(i) / std::sqrt(w[static_cast<size_t>(i)]);
    }
    Y = (Y * Q).eval();
    if (dumpOn())
        std::fprintf(stderr,
                     "[distill] M-正交化 LLT 失败, 特征分解去秩亏: %d → %lld\n",
                     k, (long long)keepc.size());
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
    BlasThreadsGuard blasGuard(p.blasThreads);
#ifdef _WIN32
    CpuWallProbe cpuProbe;
#endif

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
    ProfScope profFilter("filter");
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
    const int64_t natCount = static_cast<int64_t>(keep.size());
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
    profFilter.end();
    if (dumpOn())        std::fprintf(stderr,
                     "[distill] p=%lld λ2=%.6e λ3=%.6e 蒸馏后 p'=%lld "
                     "(自然过滤 %lld, 补足 %lld)\n",
                     (long long)n, lamDistill, lamInit,
                     (long long)res.distilledSize, (long long)natCount,
                     (long long)(res.distilledSize - natCount));
    if (keep.empty()) return res;

    DefaultTraits::DenseMatrix K2, M2;
    {
        ProfScope profExtract("extract");
        extractSub(Kr, Mr, keep, K2, M2);
    }
    const int64_t p2 = res.distilledSize;

    // ---- 2. 一次子空间迭代 + Ritz ----
    DefaultTraits::Vector lam;
    DefaultTraits::DenseMatrix V;   // p2×k，M2 正交归一
    if (p.subspaceIteration) {
        // Y = (K2 − σM2)⁻¹ M2（Abaqus OneSubspaceIterationWithRigidBodyModes）
        // σ = shift；SPD 失败走 Bunch–Kaufman（Abaqus LDLT 语义），仍失败再
        // shift escalation（PerturbDiagonal 语义）。
        ProfScope profSolve("subspace_iter_solve");
        double sigma = p.shift;
        DefaultTraits::DenseMatrix Y, S;
        std::vector<int32_t> ipiv;
        bool factored = false;
        bool indefinite = false;   // true = 用 dsytrf/dsytrs 因子
        int32_t info = 0;
        for (int attempt = 0; attempt < 6 && !factored; ++attempt) {
                S = K2 - sigma * M2;
            int32_t n32 = static_cast<int32_t>(p2);
            dpotrf_("L", &n32, S.data(), &n32, &info);
            if (info < 0)
                throw std::runtime_error("distillRitz: dpotrf 非法参数");
            if (info == 0) {
                indefinite = false;
                factored = true;
                break;
            }
            // 非 SPD：Bunch–Kaufman 一次（对称不定，Abaqus LDLT 语义）
            if (attempt == 0) {
                S = K2 - sigma * M2;
                ipiv.assign(static_cast<size_t>(p2), 0);
                double wk = 0.0;
                int32_t lq = -1;
                dsytrf_("L", &n32, S.data(), &n32, ipiv.data(), &wk, &lq,
                        &info);
                if (info == 0) {
                    std::vector<double> work(static_cast<size_t>(wk));
                    const int32_t lw2 = static_cast<int32_t>(work.size());
                    S = K2 - sigma * M2;
                    dsytrf_("L", &n32, S.data(), &n32, ipiv.data(),
                            work.data(), &lw2, &info);
                    if (info == 0) {
                        indefinite = true;
                        factored = true;
                        break;
                    }
                }
            }
            const double scale = attempt == 0 ? 1e-10 : 100.0;
            sigma = attempt == 0
                        ? 1e-10 * std::max(1.0, std::abs(lamDistill))
                        : sigma * scale;
            if (dumpOn())
                std::fprintf(stderr,
                             "[distill] 分解失败, shift escalation σ=%.6e\n",
                             sigma);
        }
        if (!factored)
            throw std::runtime_error(
                "distillRitz: 子空间迭代 (K−σM) 分解失败（shift escalation 用尽）");
        Y = M2;   // 多右端回代原地覆盖
        if (indefinite) {
            int32_t n32 = static_cast<int32_t>(p2);
            int32_t nrhs = static_cast<int32_t>(p2);
            dsytrs_("L", &n32, &nrhs, S.data(), &n32, ipiv.data(), Y.data(),
                    &n32, &info);
        } else {
            int32_t n32 = static_cast<int32_t>(p2);
            int32_t nrhs = static_cast<int32_t>(p2);
            dpotrs_("L", &n32, &nrhs, S.data(), &n32, Y.data(), &n32, &info);
        }
        if (info != 0)
            throw std::runtime_error("distillRitz: 回代求解失败");
        profSolve.end();
        // M2 正交归一（LLT 主路）
        {
            ProfScope profOrtho("ortho_y");
            if (!mOrthoNormalize(Y, M2))
                throw std::runtime_error(
                    "distillRitz: 子空间退化（M 正交化失败）");
        }
        // Ritz 分析：K̂ = YᵀK2Y（M̂ ≈ I，标准特征问题）。消费端只取
        // (bandLo, bandHi] 频段，故用 dsyevr(MRRR) 区间求解替代 dsyevd
        // 全谱分治，且 V = Y·Z 只需 p2×m 而非 p2×p2 全乘。
        ProfScope profRitz("ritz");
        DefaultTraits::DenseMatrix K2Y, Kh;
        blas::gemm0(CblasNoTrans, CblasNoTrans, K2, Y, K2Y);
        Kh.resize(p2, p2);
        blas::gemm(CblasTrans, CblasNoTrans, 1.0, Y, K2Y, 0.0, Kh);
        int32_t n32 = static_cast<int32_t>(p2);
        std::vector<double> w(static_cast<size_t>(p2));
        std::vector<int32_t> isuppz(static_cast<size_t>(2 * p2));
        double wk = 0.0;
        int32_t iwk = 0;
        int32_t lq = -1;
        const double vl = bandLo, vu = bandHi;
        const double abstol = 0.0;
        int32_t mFound = 0;
        int32_t il = 0, iu = 0;
        // range='V' 时 M 事先未知，Z 须备满 p2 列；且 Z 不得与 A 别名
        // （DORMTR 回代要读 A 内 Householder 反射子）。
        DefaultTraits::DenseMatrix Z(p2, p2);
        dsyevr_("V", "V", "L", &n32, Kh.data(), &n32, &vl, &vu, &il, &iu,
                &abstol, &mFound, w.data(), Z.data(), &n32, isuppz.data(),
                &wk, &lq, &iwk, &lq, &info);
        if (info != 0)
            throw std::runtime_error("distillRitz: dsyevr 工作区查询失败");
        std::vector<double> work(static_cast<size_t>(wk));
        std::vector<int32_t> iwork(static_cast<size_t>(iwk));
        const int32_t lw3 = static_cast<int32_t>(work.size());
        const int32_t liw3 = static_cast<int32_t>(iwork.size());
        dsyevr_("V", "V", "L", &n32, Kh.data(), &n32, &vl, &vu, &il, &iu,
                &abstol, &mFound, w.data(), Z.data(), &n32, isuppz.data(),
                work.data(), &lw3, iwork.data(), &liw3, &info);
        if (info != 0)
            throw std::runtime_error("distillRitz: Ritz 特征求解失败");
        // 只回乘频段列：V = Y·Z(:,1:mFound)
        const int64_t mB = static_cast<int64_t>(mFound);
        if (mB > 0)
            blas::gemm0(CblasNoTrans, CblasNoTrans, Y,
                        Z.leftCols(mB), V);
        lam.resize(mB);
        for (int64_t i = 0; i < mB; ++i) lam(i) = w[static_cast<size_t>(i)];
        profRitz.end();
    } else {
        // 无子空间迭代：直接对 (K2, M2) 做广义 Ritz（dsygvd 分治）
        ProfScope profGes("ges");
        DefaultTraits::DenseMatrix A2 = K2, B2 = M2;
        int32_t n32 = static_cast<int32_t>(p2);
        int32_t itype = 1;
        int32_t info = 0;
        std::vector<double> w(static_cast<size_t>(p2));
        double wk = 0.0;
        int32_t iwk = 0;
        int32_t lq = -1;
        dsygvd_(&itype, "V", "L", &n32, A2.data(), &n32, B2.data(), &n32,
                w.data(), &wk, &lq, &iwk, &lq, &info);
        if (info != 0)
            throw std::runtime_error("distillRitz: dsygvd 工作区查询失败");
        std::vector<double> work(static_cast<size_t>(wk));
        std::vector<int32_t> iwork(static_cast<size_t>(iwk));
        const int32_t lw4 = static_cast<int32_t>(work.size());
        const int32_t liw4 = static_cast<int32_t>(iwork.size());
        dsygvd_(&itype, "V", "L", &n32, A2.data(), &n32, B2.data(), &n32,
                w.data(), work.data(), &lw4,
                iwork.data(), &liw4, &info);
        if (info != 0)
            throw std::runtime_error("distillRitz: 广义特征求解失败");
        lam.resize(p2);
        for (int64_t i = 0; i < p2; ++i) lam(i) = w[static_cast<size_t>(i)];
        V = std::move(A2);
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
