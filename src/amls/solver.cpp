// AMLS 求解器协调器实现：四阶段流水线编排。
//
//   (1) step1 剖分与重排：|K|+|M| 邻接图递归二分 → 分割节点树 + 排列 P
//   (2) step2a 高斯消去（GaussianElimination）：
//       N1 约束模态回代（一次分解 K̂_ii、逐祖先窄右端）→ N3 Mtemp →
//       N4 K Schur + N5 M 更新（Mtemp 形式）——与截断无关
//   (3) step2b 子空间缩聚（SubspaceCondensation）：
//       N7 局部谱 → N8 μ̃ 种子（MassCouplingStore）→ N9 折叠 / N10 传播
//       M̃ 树块表（P4/P8），K̃ 对角（P3）
//   (4) step3 降阶广义特征问题（稠密 dsygvd / dsygvx）
//   (5) step4 恢复 z_i = eigenvectors_i·q_i + Σ_a u(i,a)·z_a（纯 GEMM，
//       可选；只需 u 与 eigenvectors 存活）→ 反排列 + M-归一
//
// 截断阈值 λ_cut = η·λ̂，λ̂ 为全局第 nev 个特征值的 Ritz 上界，
// 由趟 A（固定小数目截断）的降阶解给出（Ritz 值从上方逼近真值，安全）。
#include "amls/solver.hpp"

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
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
DefaultTraits::DenseMatrix assembleReducedMassDense(
    const SubspaceCondensation::Result& res) {
    DefaultTraits::DenseMatrix reducedMass =
        DefaultTraits::DenseMatrix::Identity(res.reducedSize, res.reducedSize);
    for (const auto& blk : res.reducedMassBlocks) {
        const int64_t r0 =
            res.reducedOffset[static_cast<size_t>(blk.rowNode)];
        const int64_t c0 =
            res.reducedOffset[static_cast<size_t>(blk.colNode)];
        const int64_t m = blk.block.rows(), n = blk.block.cols();
        reducedMass.block(r0, c0, m, n) = blk.block;
        reducedMass.block(c0, r0, n, m) = blk.block.transpose();
    }
    return reducedMass;
}

/// K̃ 对角装配（P3：零计算，直接铺保留特征值）。
void assembleReducedStiffnessDiag(const SubspaceCondensation::Result& res,
                                  const SeparatorTree& tree,
                                  DefaultTraits::DenseMatrix& reducedK) {
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const int64_t o = res.reducedOffset[static_cast<size_t>(i)];
        const int64_t k = res.keptModes[static_cast<size_t>(i)];
        for (int64_t j = 0; j < k; ++j)
            reducedK(o + j, o + j) =
                res.eigenvalues[static_cast<size_t>(i)][j];
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
        [&](const SubspaceCondensation::Result& res,
            int64_t want) -> std::pair<Vector, DenseMatrix> {
        if (res.reducedSize <= 0)
            throw std::runtime_error("降阶问题为空（所有节点截断为空）");
        DenseMatrix reducedMass = assembleReducedMassDense(res);
        DenseMatrix reducedK =
            DenseMatrix::Zero(res.reducedSize, res.reducedSize);
        assembleReducedStiffnessDiag(res, tree, reducedK);
        Vector projectedEigenvalues;
        DenseMatrix projectedEigenvectors;
        want = std::min<int64_t>(want, res.reducedSize);
        if (want < res.reducedSize)
            dense.solveIndex(reducedK, reducedMass, 1, want,
                             projectedEigenvalues, projectedEigenvectors);
        else
            dense.solveAll(reducedK, reducedMass, projectedEigenvalues,
                           projectedEigenvectors);
        return {projectedEigenvalues, projectedEigenvectors};
    };

    // ---- step2a+2b 融合流水线（逐层：消去 → 局部谱 → 投影 → 释放）----
    // 内存关键：回代恢复只需 u 与 eigenvectors（纯 GEMM），耦合列/diag/
    // Mtemp 在该层投影完成后即可释放（recoverVectors 时仅多保 u 列）。
    // GE 只依赖 orig* 稀疏原块，趟间 resetBlocks 后可确定性重跑。
    // 趟 A 释放一切只留 λ̂；趟 B 重跑整个流水线。
    GaussianElimination ge(reordered, tree);
    SubspaceCondensation cond(ge, eig);

    const int64_t nTreeNodes = tree.nNodes();
    auto initRes = [&](SubspaceCondensation::Result& res) {
        res = SubspaceCondensation::Result();
        res.keptModes.assign(static_cast<size_t>(nTreeNodes), 0);
        res.eigenvalues.resize(static_cast<size_t>(nTreeNodes));
        res.eigenvectors.resize(static_cast<size_t>(nTreeNodes));
        res.reducedOffset.assign(static_cast<size_t>(nTreeNodes), 0);
    };

    const bool keepVectors = optN.recoverVectors;
    // 调试开关：AMLS_NO_FUSE=1 走旧路径（GE 一趟跑完不释放 + cond.run 两阶段），
    // 用于对照融合流水线的数值与性能。
    const bool noFuse = std::getenv("AMLS_NO_FUSE") != nullptr;
    bool geRan = false;
    auto runFused = [&](const ISpectralTruncation<DefaultTraits>& tr,
                        SubspaceCondensation::Result& out) {
        GaussianElimination::phaseProfReset();
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
        // N7–N10/free 全部进同一 DagRunner——消去与缩聚跨阶段重叠
        //（子树消去完成即可局部谱+折叠传播），块内存随释放任务边算边放。
        // 释放只缩数据不删表项（预建键，无锁 find 并发安全）。
        const int effThreads =
            threads > 0 ? threads
                        : static_cast<int>(std::thread::hardware_concurrency());
        if (effThreads > 1) {
            ge.resetBlocks();
            cond.beginPass();
            DagRunner dag(effThreads);
            GaussianElimination::EliminationDagTasks geInfo;
            ge.buildDag(dag, geInfo);
            cond.buildDag(dag, out, tr, geInfo, !keepVectors);
            for (int64_t i = 0; i < nTreeNodes; ++i)
                if (tree.node(i).children.empty())
                    dag.ready(*geInfo.n1[static_cast<size_t>(i)]);
            dag.waitIdle();
            cond.finalize(out);
            return;
        }
        cond.beginPass();
        // 每趟消去前清空动态块表：趟 A 残留块会让趟 B 的重跑变成在旧值
        // 上累加（K̂ 失去正定性）。
        ge.resetBlocks();
        const int64_t nLevels = ge.maxLevel() + 1;
        for (int64_t l = 0; l < nLevels; ++l) {
            ge.runLevel(l, threads);
            cond.spectralLevel(tr, l, threads, out);
            cond.projectLevel(l, threads, out);
            // 回代恢复只需 u：diag/耦合列/Mtemp 一律释放，u 列按
            // recoverVectors 保留。
            for (int64_t i : ge.levelNodes(l))
                ge.releaseNodeData(i, keepVectors);
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

    if (optN.lambdaCut > 0.0) {
        // f_user 模式：绝对特征值截断，单趟，保留全部 λ≤λ_cut 的模态
        ThresholdTruncation<DefaultTraits> tr(optN.lambdaCut);
        trace::mark("solver", "pass.cut", 0);
        runFused(tr, res);
    } else if (optN.fixedKeep > 0) {
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
        SubspaceCondensation::Result resA;
        trace::mark("solver", "pass.A", 0);
        runFused(trA, resA);
        auto [passAEigenvalues, passAVectors] =
            solveProjected(resA,
                           std::min<int64_t>(optN.nev, resA.reducedSize));
        (void)passAVectors;
        if (static_cast<int64_t>(passAEigenvalues.size()) < optN.nev)
            throw std::runtime_error(
                "趟 A 降阶问题规模不足（p < nev）；请减小 nLevels 或增大 leafSize");
        const double lambdaHat = passAEigenvalues[optN.nev - 1];
        lambdaCut = optN.thresholdFactor * lambdaHat;
        if (optN.verbosity > 0)
            std::cout << "[amls] 趟A: p0=" << resA.reducedSize
                      << " λ̂=" << lambdaHat
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
        for (const auto& [key, value] : ge.lastTiming())
            std::cout << "[amls]   消去." << key << "=" << value << "s\n";
        {
            using GP = GaussianElimination::GePhaseProf;
            GP& prof = GaussianElimination::phaseProf();
            std::cout << "[amls]   探针.schurKdiag=" << prof.schurKdiag
                      << " schurKcoup=" << prof.schurKcoup
                      << " massDiagT1=" << prof.massDiagTerm1
                      << " massDiagT2=" << prof.massDiagTerm2
                      << " massCoupT1=" << prof.massCoupTerm1
                      << " massCoupT2=" << prof.massCoupTerm2
                      << " mtempBuild=" << prof.mtempBuild
                      << " solveRhs=" << prof.solveRhs << "\n";
        }
        for (const auto& [key, value] : cond.lastStageTiming())
            std::cout << "[amls]   缩聚." << key << "=" << value << "s\n";
    }
    sol.reducedSize = res.reducedSize;
    if (optN.verbosity > 0)
        std::cout << "[amls] 缩聚完成: p=" << res.reducedSize
                  << " 树节点=" << tree.nNodes() << "\n";
    if (res.reducedSize > optN.denseProjectedMax && optN.verbosity > 0)
        std::cout << "[amls] 警告: p=" << res.reducedSize
                  << " 超过 denseProjectedMax=" << optN.denseProjectedMax
                  << "（P12 切换点；仍用稠密求解）\n";

    // ---- step3 降阶广义特征问题 ----
    // f_user 模式输出全部截断内模态（p 个，约等于截止频率以下的模态总数）；
    // nev 模式仍按 nev 输出。
    const int64_t want = optN.lambdaCut > 0.0
                             ? res.reducedSize
                             : std::min<int64_t>(optN.nev, res.reducedSize);
    auto [eigenvalues, projectedVectors] = solveProjected(res, want);
    const auto t4 = Clock::now();
    sol.timing["reduced"] = seconds(t3, t4);

    // ---- step4 恢复（可选；纯 GEMM 回代，只需 u 与 eigenvectors）----
    if (optN.recoverVectors) {
        DenseMatrix recovered = cond.backtransform(res, projectedVectors);
        const auto t5a = Clock::now();
        sol.timing["backtransform"] = seconds(t4, t5a);

        // 反排列 + M-归一
        const int64_t ncols = recovered.cols();
        DenseMatrix eigenvectors(n, ncols);
        for (int64_t j = 0; j < n; ++j)
            eigenvectors.row(part.perm[static_cast<size_t>(j)]) =
                recovered.row(j);
        for (int64_t c = 0; c < ncols; ++c) {
            const double mz =
                eigenvectors.col(c).dot(pencil->M * eigenvectors.col(c));
            eigenvectors.col(c) /= std::sqrt(std::max(mz, 1e-300));
        }
        sol.eigenvectors = std::move(eigenvectors);
    } else {
        sol.timing["backtransform"] = 0.0;
    }

    sol.eigenvalues = eigenvalues;
    trace::dump();
    return sol;
}

} // namespace amls
