// AMLS 求解器协调器实现：四阶段流水线编排。
//
//   (1) step1 剖分与重排：|K|+|M| 邻接图递归二分 → 分割节点树 + 排列 P
//   (2) step2a 高斯消去（GaussianElimination 单趟后序）：
//       块分解 K̂ = LDLᵀ → W/X/Y 回代 → K Schur + M 同余更新（Hyun 式 2/5）
//       —— 与截断无关，两趟阈值估计只跑一次
//   (3) step2b 子空间缩聚（SubspaceCondensation）：
//       并行局部谱（式 4，线程池）→ 后序投影（式 7/8、16/17）
//       M̃ 树块表（P4/P8），K̃ 对角（P3）
//   (4) step3 降阶广义特征问题（式 21，稠密 dsygvd / dsygvx）
//   (5) step4 恢复 z = L⁻ᵀ S q（可选；因子与 Φ_i 不释放）→ 反排列 + M-归一
//
// 截断阈值 λ_cut = η·λ̂，λ̂ 为全局第 nev 个特征值的 Ritz 上界，
// 由趟 A（固定小数目截断）的降阶解给出（Ritz 值从上方逼近真值，安全）。
#include "amls/solver.hpp"

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Sparse>

#include "amls/gaussian_elimination.hpp"
#include "amls/separator_tree.hpp"
#include "amls/subspace_condensation.hpp"
#include "core/logging.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "partition/partitioner.hpp"

namespace amls {

namespace {

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

/// 按 perm（perm[new] = old）重排对称稀疏矩阵：B = P A Pᵀ。
DefaultTraits::SparseMatrix permuteSymmetric(
    const DefaultTraits::SparseMatrix& A,
    const std::vector<int64_t>& perm) {
    using Scalar = DefaultTraits::Scalar;
    using Index  = DefaultTraits::Index;
    // perm 是 new→old；装配需要 old→new（逆排列）
    std::vector<Index> inv(static_cast<size_t>(A.rows()));
    for (Index j = 0; j < static_cast<Index>(perm.size()); ++j)
        inv[static_cast<size_t>(perm[static_cast<size_t>(j)])] = j;
    std::vector<Eigen::Triplet<Scalar, Index>> trips;
    trips.reserve(static_cast<size_t>(A.nonZeros()));
    for (Index c = 0; c < A.outerSize(); ++c) {
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it) {
            trips.emplace_back(
                inv[static_cast<size_t>(it.row())],
                inv[static_cast<size_t>(it.col())], it.value());
        }
    }
    DefaultTraits::SparseMatrix B(A.rows(), A.cols());
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}

/// 装配降阶问题的稠密 M̃（对角块 = I，P8；非对角树块来自缩聚）。
DefaultTraits::DenseMatrix assembleMprojDense(
    const SubspaceCondensation::Result& r) {
    DefaultTraits::DenseMatrix M =
        DefaultTraits::DenseMatrix::Identity(r.p, r.p);
    for (const auto& blk : r.mprojBlocks) {
        const int64_t r0 = r.projOffset[static_cast<size_t>(blk.rowNode)];
        const int64_t c0 = r.projOffset[static_cast<size_t>(blk.colNode)];
        const int64_t m  = blk.block.rows(), n = blk.block.cols();
        M.block(r0, c0, m, n) = blk.block;
        M.block(c0, r0, n, m) = blk.block.transpose();
    }
    return M;
}

/// K̃ 对角装配（P3：零计算，直接铺保留特征值）。
void assembleKprojDiag(const SubspaceCondensation::Result& r,
                       const SeparatorTree& tree,
                       DefaultTraits::DenseMatrix& Kp) {
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const int64_t o = r.projOffset[static_cast<size_t>(i)];
        const int64_t k = r.k[static_cast<size_t>(i)];
        for (int64_t j = 0; j < k; ++j)
            Kp(o + j, o + j) = r.mu[static_cast<size_t>(i)][j];
    }
}

} // namespace

AmlsSolver::Solution AmlsSolver::solve(const PencilT& KM,
                                       const AmlsOptions& opt) {
    Solution sol;
    AmlsOptions optN = opt;
    const auto t0 = Clock::now();
    trace::initFromEnv();

    if (optN.nev <= 0)
        throw std::invalid_argument("nev 必须为正");
    const int64_t n = KM.K.rows();
    if (n == 0)
        throw std::invalid_argument("空矩阵");
    optN.nev = std::min<int64_t>(optN.nev, n);   // nev 不超过问题规模

    // (0) 可选移位：K ← K + shift·M（零谱/半正定保护；特征值平移 shift，特征向量不变）
    PencilT shifted;
    const PencilT* pencil = &KM;
    if (optN.shift != 0.0) {
        shifted.K = KM.K;
        shifted.K += optN.shift * KM.M;
        shifted.M = KM.M;
        pencil = &shifted;
    }

    // ---- step1 剖分与重排 ----
    MetisPartitioner<DefaultTraits> partitioner;
    auto part = partitioner.recursiveBisection(
        pencil->K + pencil->M, optN.nLevels, optN.leafSize);
    SeparatorTree& tree = part.tree;
    lastTreeNodes_ = tree.nNodes();
    PencilT reordered;
    reordered.K = permuteSymmetric(pencil->K, part.perm);
    reordered.M = permuteSymmetric(pencil->M, part.perm);
    const auto t1 = Clock::now();
    sol.timing["partition"] = seconds(t0, t1);

    LapackDenseEigensolver<DefaultTraits> dense;
    EigenpairSolver eig(dense);
    const int threads = optN.numThreads <= 0 ? 0 : optN.numThreads;

    // OpenBLAS（OpenMP 版）线程与本工程 std::thread 池互踩严重：
    // 实测 12 工作线程 × BLAS 内部线程使局部谱阶段慢 9 倍。
    // 策略：BLAS 永久单线程，并行只走节点级线程池（OOC 论文的同层并行）。
    blas::setNumThreads(1);

    auto solveProjected =
        [&](const SubspaceCondensation::Result& r,
            int64_t want) -> std::pair<Vector, DenseMatrix> {
        if (r.p <= 0)
            throw std::runtime_error("降阶问题为空（所有节点截断为空）");
        DenseMatrix Mp = assembleMprojDense(r);
        DenseMatrix Kp = DenseMatrix::Zero(r.p, r.p);
        assembleKprojDiag(r, tree, Kp);
        Vector w;
        DenseMatrix V;
        want = std::min<int64_t>(want, r.p);
        if (want < r.p)
            dense.solveIndex(Kp, Mp, 1, want, w, V);
        else
            dense.solveAll(Kp, Mp, w, V);
        return {w, V};
    };

    // ---- step2a+2b 融合流水线（逐层：消去 → 局部谱 → 投影 → 释放）----
    // 内存关键：列节点在第 l 层的祖先稠密耦合块 (a,i) 只在
    // runLevel(l)…projectLevel(l) 之间存活，处理完该层立即 releaseNode
    // 回收（只保留因子与 orig* 稀疏原块；GE 只依赖 orig*，可确定性重跑）。
    // 趟 A 释放一切只留 λ̂；趟 B 重跑整个流水线（recoverVectors 时保留）。
    GaussianElimination ge(reordered, tree);
    SubspaceCondensation cond(ge, eig);

    const int64_t nTreeNodes = tree.nNodes();
    auto initRes = [&](SubspaceCondensation::Result& r) {
        r = SubspaceCondensation::Result();
        r.k.assign(static_cast<size_t>(nTreeNodes), 0);
        r.mu.resize(static_cast<size_t>(nTreeNodes));
        r.Z.resize(static_cast<size_t>(nTreeNodes));
        r.projOffset.assign(static_cast<size_t>(nTreeNodes), 0);
    };

    const bool keepFactors = optN.recoverVectors;
    // 调试开关：AMLS_NO_FUSE=1 走旧路径（GE 一趟跑完不释放 + cond.run 两阶段），
    // 用于对照融合流水线的数值与性能。
    const bool noFuse = std::getenv("AMLS_NO_FUSE") != nullptr;
    bool geRan = false;
    auto runFused = [&](const ISpectralTruncation<DefaultTraits>& tr,
                        SubspaceCondensation::Result& out) {
        initRes(out);
        if (noFuse) {
            if (!geRan) {
                ge.run(threads);
                geRan = true;
            }
            out = cond.run(tr, threads);
            return;
        }
        // 统一 DAG 路径（effThreads>1）：GE 节点内子任务 + 缩聚
        // SPEC/f/g/h/rel 全部进同一 DagRunner——消去与缩聚跨阶段重叠
        // （子树消去完成即可局部谱+投影），块内存随 rel(i) 边算边放，
        // 500k 规模常驻块不再堆积。rel 只缩数据不删表项（桶链并发安全）。
        const int effThreads =
            threads > 0 ? threads
                        : static_cast<int>(std::thread::hardware_concurrency());
        if (effThreads > 1) {
            ge.resetBlocks();
            cond.beginPass();
            DagRunner dag(effThreads);
            GaussianElimination::GeDagTasks geInfo;
            ge.buildDag(dag, geInfo);
            cond.buildDag(dag, out, tr, geInfo, !keepFactors);
            for (int64_t i = 0; i < nTreeNodes; ++i)
                if (tree.node(i).children.empty())
                    dag.ready(*geInfo.f[static_cast<size_t>(i)]);
            dag.waitIdle();
            cond.finalize(out);
            return;
        }
        cond.beginPass();
        // 每趟消去前清空动态块表：keepFactors（恢复向量）时趟 A 不释放，
        // 不清理则趟 B 会在趟 A 残留块上累加（K̂ 失去正定性）。
        ge.resetBlocks();
        const int64_t nLevels = ge.maxLevel() + 1;
        for (int64_t l = 0; l < nLevels; ++l) {
            ge.runLevel(l, threads);
            cond.spectralLevel(tr, l, threads, out);
            cond.projectLevel(l, threads, out);
            if (!keepFactors)
                for (int64_t i : ge.levelNodes(l))
                    ge.releaseNode(i);
            if (amls::log::global()->should_log(spdlog::level::debug)) {
                int64_t zc, tc, zb, tb;
                ge.zeroColStats(zc, tc, zb, tb);
                AMLS_LOG_DEBUG(
                    "[ge] 层 {} 后稠密块={:.2f}GB 零列={}/{} 全零块={}/{}", l,
                    ge.denseBytes() / 1e9, zc, tc, zb, tb);
            }
        }
        cond.finalize(out);
    };

    SubspaceCondensation::Result res;
    double lambdaCut = -1.0;

    if (optN.fixedKeep > 0) {
        FixedCountTruncation<DefaultTraits> tr(optN.fixedKeep);
        trace::mark("solver", "pass.A", 0);
        runFused(tr, res);
    } else {
        if (!optN.twoPass)
            throw std::invalid_argument(
                "twoPass=false 且 fixedKeep=0 时无法确定截断阈值；请提供 fixedKeep");
        // 趟 A：固定小数目截断 → 降阶解 → λ̂（Ritz 上界）
        int64_t k0 = std::max<int64_t>(
            4, (3 * optN.nev + tree.nNodes() - 1) / tree.nNodes());
        FixedCountTruncation<DefaultTraits> trA(k0);
        SubspaceCondensation::Result ra;
        trace::mark("solver", "pass.A", 0);
        runFused(trA, ra);
        auto [wA, vA] = solveProjected(ra, std::min<int64_t>(optN.nev, ra.p));
        (void)vA;
        if (static_cast<int64_t>(wA.size()) < optN.nev)
            throw std::runtime_error(
                "趟 A 降阶问题规模不足（p < nev）；请减小 nLevels 或增大 leafSize");
        const double lambdaHat = wA[optN.nev - 1];
        lambdaCut = optN.thresholdFactor * lambdaHat;
        if (optN.verbosity > 0)
            std::cout << "[amls] 趟A: p0=" << ra.p << " λ̂=" << lambdaHat
                      << " λ_cut=" << lambdaCut << "\n";

        // 趟 B：阈值截断
        ThresholdTruncation<DefaultTraits> trB(lambdaCut);
        trace::mark("solver", "pass.B", 0);
        runFused(trB, res);
    }
    const auto t3 = Clock::now();
    const double phase2 = seconds(t1, t3);
    const double geTime =
        ge.lastTiming().count("par.total") ? ge.lastTiming().at("par.total") : 0.0;
    sol.timing["phase2"] = phase2;
    sol.timing["gauss"] = geTime;
    sol.timing["condense"] = phase2 - geTime;
    if (optN.verbosity > 0) {
        for (const auto& [k, v] : ge.lastTiming())
            std::cout << "[amls]   消去." << k << "=" << v << "s\n";
        for (const auto& [k, v] : cond.lastStageTiming())
            std::cout << "[amls]   缩聚." << k << "=" << v << "s\n";
    }
    sol.p = res.p;
    if (optN.verbosity > 0)
        std::cout << "[amls] 缩聚完成: p=" << res.p
                  << " 树节点=" << tree.nNodes() << "\n";
    if (res.p > optN.denseProjectedMax && optN.verbosity > 0)
        std::cout << "[amls] 警告: p=" << res.p
                  << " 超过 denseProjectedMax=" << optN.denseProjectedMax
                  << "（P12 切换点；仍用稠密求解）\n";

    // ---- step3 降阶广义特征问题（式 21）----
    const int64_t want = std::min<int64_t>(optN.nev, res.p);
    auto [theta, Q] = solveProjected(res, want);
    const auto t4 = Clock::now();
    sol.timing["reduced"] = seconds(t3, t4);

    // ---- step4 恢复（可选）----
    if (optN.recoverVectors) {
        DenseMatrix z = cond.backtransform(res, Q);
        const auto t5a = Clock::now();
        sol.timing["backtransform"] = seconds(t4, t5a);

        // 反排列 + M-归一
        const int64_t ncols = z.cols();
        DenseMatrix X(n, ncols);
        for (int64_t j = 0; j < n; ++j)
            X.row(part.perm[static_cast<size_t>(j)]) = z.row(j);
        for (int64_t c = 0; c < ncols; ++c) {
            const double mz = X.col(c).dot(pencil->M * X.col(c));
            X.col(c) /= std::sqrt(std::max(mz, 1e-300));
        }
        sol.Z = std::move(X);
    } else {
        sol.timing["backtransform"] = 0.0;
    }

    sol.theta = theta;
    trace::dump();
    return sol;
}

} // namespace amls
