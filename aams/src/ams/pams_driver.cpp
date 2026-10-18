// std_pamsDriver 的 C++ 复原实现（三阶段 + PAMS Timing Data）。
// 结构注释见 pams_driver.hpp；各阶段与逆向符号的对应关系：
//   [1][3][4] METIS ORDERING / BUILD TREE / SYMBOLIC FACTOR → buildAbaqusTree
//   [7]       pme_SolverParameters                          → PamsParameters
//   [9]-Reduction   局部特征 + Ψ + 缩聚                      → 列压缩 DAG
//   [9]-Reduced     蒸馏 + 一次子空间迭代 + Ritz             → distillRitz
//   [9]-Recovery    沿树 Ψ 传播 + 反排列 + M 归一            → 分层回代
#include "ams/pams_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ams/distill_solver.hpp"
#include "ams/eigen_pair_saver.hpp"
#include "ams/gauss_module.hpp"
#include "ams/node_task_executor.hpp"
#include "ams/original_blocks.hpp"
#include "ams/separator_tree.hpp"
#include "ams/subspace_mass_saver.hpp"
#include "ams/temp_block_manager.hpp"
#include "ams/thread_pool.hpp"
#include "ams/tree_builder.hpp"
#include "core/logging.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"
#include "linalg/col_ops.hpp"
#include "linalg/gemm_cols.hpp"
#include "linalg/colcomp_mat.hpp"
#include "linalg/dense_eigensolver.hpp"

namespace aams {

namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

/// 按 perm（perm[new] = old）重排对称稀疏矩阵：B = P A Pᵀ。
DefaultTraits::SparseMatrix permuteSymmetric(
    const DefaultTraits::SparseMatrix& A,
    const std::vector<int64_t>& perm) {
    using Scalar = DefaultTraits::Scalar;
    using Index  = DefaultTraits::Index;
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

/// 树索引重映射到重排坐标（perm[new]=old 逆映射，节点 dofs 连续升序段）。
SeparatorTree remapTreeToReordered(const SeparatorTree& tree,
                                   const std::vector<int64_t>& perm) {
    std::vector<int64_t> inv(perm.size());
    for (size_t j = 0; j < perm.size(); ++j)
        inv[static_cast<size_t>(perm[j])] = static_cast<int64_t>(j);
    SeparatorTree out;
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const auto& src = tree.node(i).indices;
        std::vector<int64_t> idx;
        idx.reserve(src.size());
        for (int64_t old : src)
            idx.push_back(inv[static_cast<size_t>(old)]);
        std::sort(idx.begin(), idx.end());
        out.addNode(tree.node(i).role, std::move(idx));
    }
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const auto& ch = tree.node(i).children;
        if (ch.empty()) continue;
        out.setChildren(i, ch);
        for (int64_t c : ch) out.setParent(c, i);
    }
    out.relabelRoot();
    return out;
}

/// 节点重编号为后序（祖先 id 恒大于后代 id）。
SeparatorTree renumberPostOrder(const SeparatorTree& tree) {
    const auto order = tree.postOrder();
    std::vector<int64_t> map(static_cast<size_t>(tree.nNodes()), -1);
    SeparatorTree out;
    for (int64_t k = 0; k < static_cast<int64_t>(order.size()); ++k) {
        const auto& nd = tree.node(order[static_cast<size_t>(k)]);
        map[static_cast<size_t>(order[static_cast<size_t>(k)])] =
            out.addNode(nd.role, nd.indices);
    }
    for (int64_t k = 0; k < static_cast<int64_t>(order.size()); ++k) {
        const int64_t old = order[static_cast<size_t>(k)];
        const auto& ch = tree.node(old).children;
        if (ch.empty()) continue;
        std::vector<int64_t> kids;
        kids.reserve(ch.size());
        for (int64_t c : ch) kids.push_back(map[static_cast<size_t>(c)]);
        out.setChildren(map[static_cast<size_t>(old)], kids);
        for (int64_t c : kids) out.setParent(c, map[static_cast<size_t>(old)]);
    }
    out.relabelRoot();
    return out;
}

}  // namespace

AmsSolver::Solution AmsSolver::solve(const Pencil<DefaultTraits>& KM,
                                     const AmsOptions& optIn) {
    Solution sol;
    AmsOptions opt = optIn;
    const auto t0 = Clock::now();
    trace::initFromEnv();

    if (!(opt.fmaxHz > 0.0))
        throw std::invalid_argument("fmaxHz 必须为正（Hz）");
    if (!(opt.fminHz >= 0.0) || !(opt.fminHz < opt.fmaxHz))
        throw std::invalid_argument("要求 0 <= fmin < fmax");
    if (!(opt.cutoff1 > 0.0))
        throw std::invalid_argument("cutoff1 必须为正");
    const int64_t n = KM.K.rows();
    if (n == 0) throw std::invalid_argument("空矩阵");
    // Abaqus 在 PROCESSING CONSTRAINTS 阶段先行消化约束（大数法 1e36 占位
    // 由此消除）；矩阵工作流输入若仍含大数占位，特征问题被污染——明确报错
    // 而非产出垃圾模态。
    {
        double maxAbs = 0.0;
        for (int64_t c = 0; c < KM.K.outerSize(); ++c)
            for (DefaultTraits::SparseMatrix::InnerIterator it(KM.K, c); it; ++it)
                maxAbs = std::max(maxAbs, std::abs(it.value()));
        if (maxAbs > 1e20)
            throw std::invalid_argument(
                "K 含 |K|max>1e20 的大数约束占位（Abaqus 1e36 大数法），"
                "请先做约束后处理再求解");
    }

    // OpenBLAS（OpenMP 版）线程与本工程线程池互踩：BLAS 永久单线程。
    blas::setNumThreads(1);

    // ---- [7] pme_SolverParameters（隐藏参数走 ABA_PAMS_* 环境变量）----
    PamsParameters& par = sol.params;
    par.fmaxHz = opt.fmaxHz;
    par.fminHz = opt.fminHz;
    par.cutoff1 = opt.cutoff1;
    par.cutoff2 = opt.cutoff2;
    par.cutoff3 = opt.cutoff3;
    par.nev = opt.nev;
    if (opt.subspaceMax > 0) par.subspaceMax = opt.subspaceMax;
    par.lastCutoff = opt.lastCutoff;
    par.enhancedEnabled = opt.enhancedEnabled;
    par.enhancedCutoff = opt.enhancedCutoff;
    par.enhancedLevel = opt.enhancedLevel;
    par.numThreads = opt.numThreads;
    if (opt.readEnv) {
        // 显式传入的 shift/增强设置优先：先读 opt 的值，再让环境变量补齐
        // 未显式指定的部分（与 std_pamsDriver 的 getenv 分支一致）。
        PamsParameters envOnly;
        envOnly.fmaxHz = par.fmaxHz;
        envOnly.applyEnv();
        if (opt.globalShift != 0.0) par.globalShift = opt.globalShift;
        else par.globalShift = envOnly.globalShift;
        if (!opt.enhancedEnabled) {
            par.enhancedEnabled = envOnly.enhancedEnabled;
            par.enhancedCutoff = envOnly.enhancedCutoff;
            par.enhancedLevel = envOnly.enhancedLevel;
        }
        if (opt.lastCutoff == 0.0) par.lastCutoff = envOnly.lastCutoff;
    }

    // ---- [1][3][4] METIS ORDERING + BUILD TREE + SYMBOLIC FACTOR ----
    if (opt.verbosity > 0)
        std::cerr << "[aams] METIS ordering + build tree\n" << std::flush;
    rea::AbaqusTreeOptions ropt;
    ropt.maxBlockDofs = opt.maxBlockDofs;
    ropt.collapseMaxDofs = opt.collapseMaxDofs;
    ropt.leafAggrMaxDofs = opt.leafAggrMaxDofs;
    ropt.minBlockDofs = opt.minBlockDofs;
    ropt.collapseMode = opt.collapseMode;
    ropt.aggregateLeaves = opt.aggregateLeaves;
    ropt.dofsPerNode = opt.dofsPerNode;
    ropt.superRootDofs = n;   // 多根森林合成空 DOF 超根（单根约定）
    const bool profLog = std::getenv("AAMS_PROFILE") != nullptr;
    auto tpTree = Clock::now();
    const auto tkm = Clock::now();
    DefaultTraits::SparseMatrix kmSum = KM.K + KM.M;
    if (std::getenv("AAMS_PROFILE"))
        std::fprintf(stderr, "[prof] tree K+Madd=%.2fs\n",
                     seconds(tkm, Clock::now()));
    auto part = rea::buildAbaqusTree(kmSum, ropt);
    const double treeSec =
        std::chrono::duration<double>(Clock::now() - tpTree).count();
    auto tpPerm = Clock::now();
    SeparatorTree tree =
        renumberPostOrder(remapTreeToReordered(part.tree, part.perm));
    const int64_t nTreeNodes = tree.nNodes();
    Pencil<DefaultTraits> reordered;
    reordered.K = permuteSymmetric(KM.K, part.perm);
    reordered.M = permuteSymmetric(KM.M, part.perm);
    const double permSec =
        std::chrono::duration<double>(Clock::now() - tpPerm).count();
    if (profLog)
        std::cerr << "[prof] treebuild=" << treeSec << "s permute="
                  << permSec << "s\n"
                  << std::flush;
    sol.timing["METIS_ORDERING_BUILD_TREE"] = seconds(t0, Clock::now());

    // ---- [9] Reduction Phase：局部特征 + 约束模态 Ψ + 缩聚 ----
    const auto t1 = Clock::now();
    if (opt.verbosity > 0)
        std::cerr << "[aams] reduction: " << nTreeNodes << " substructures\n"
                  << std::flush;
    const bool redCloseLog = std::getenv("AAMS_RED_CLOSE_LOG") != nullptr;
    const auto ts0 = Clock::now();
    OriginalBlocks ob(reordered, tree);
    reordered.K.resize(0, 0);
    reordered.M.resize(0, 0);
    const auto ts1 = Clock::now();

    GaussEliminationModule gauss;
    const auto tg0 = Clock::now();
    if (!gauss.initialize(ob))
        throw std::runtime_error("消去模块初始化失败");
    const auto tg1 = Clock::now();
    EigenPairSaver saver;
    if (!saver.initialize(nTreeNodes, true))
        throw std::runtime_error("特征对存储初始化失败");
    SubspaceMassSaver massSaver;
    if (!massSaver.initialize(tree))
        throw std::runtime_error("子空间质量存储初始化失败");
    const auto tg2 = Clock::now();
    TempBlockManager temp;
    if (!temp.initialize(tree, &gauss, &saver, &massSaver))
        throw std::runtime_error("临时块管理器初始化失败");
    NodeTaskStateStore states(nTreeNodes);
    if (redCloseLog)
        std::fprintf(stderr,
                     "[redclose] OriginalBlocks=%.1fms gaussInit=%.1fms "
                     "savers=%.1fms tempInit=%.1fms\n",
                     seconds(ts0, ts1) * 1e3, seconds(tg0, tg1) * 1e3,
                     seconds(tg1, tg2) * 1e3, seconds(tg2, Clock::now()) * 1e3);

    // ApplyClassicCutoffs / ApplyEnhancedCutoffs 分发 [R]：
    //   默认 ω1²；根层（rootDepth==0）ωlast²；增强标记（levelFromLeaf ≤
    //   enhancedLevel，level 从叶计数 [R] 45683-45713）ωenh²。
    const int64_t maxDepth = [&] {
        int64_t d = 0;
        for (int64_t i = 0; i < nTreeNodes; ++i)
            d = std::max(d, static_cast<int64_t>(tree.ancestors(i).size()));
        return d;
    }();
    std::vector<double> lambdaHi(static_cast<size_t>(nTreeNodes));
    int64_t enhancedCount = 0;
    for (int64_t i = 0; i < nTreeNodes; ++i) {
        const int64_t rootDepth =
            static_cast<int64_t>(tree.ancestors(i).size());
        const int64_t levelFromLeaf = maxDepth - rootDepth;
        double cut = par.omega1Sq();
        if (rootDepth == 0)
            cut = par.omegaLastSq();
        if (par.enhancedEnabled && levelFromLeaf <= par.enhancedLevel) {
            cut = par.omegaEnhSq();
            ++enhancedCount;
        }
        lambdaHi[static_cast<size_t>(i)] = cut;
    }

    LapackDenseEigensolver<DefaultTraits> dense;
    NodeTaskExecutor::LocalEigenConfig cfg;
    cfg.dense = &dense;
    cfg.blocks = &ob;
    cfg.lambdaHi = par.omega1Sq();
    cfg.lambdaHiPerNode = &lambdaHi;
    NodeTaskExecutor exec(tree, gauss, temp, saver, states, cfg);

    const int threads =
        opt.numThreads > 0
            ? opt.numThreads
            : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int nodeThreads = std::max(1, static_cast<int>(
        std::llround(static_cast<double>(threads) / 1.5)));
    {
        const bool redCloseLog = std::getenv("AAMS_RED_CLOSE_LOG") != nullptr;
        const auto tp0 = Clock::now();
        ThreadPool nodePool(nodeThreads);
        ThreadPool branchPool(threads);
        const auto tp1 = Clock::now();
        if (!exec.runNodeOrder(nodePool, branchPool))
            throw std::runtime_error("Reduction Phase 失败（详见日志）");
        const auto tp2 = Clock::now();
        if (redCloseLog) {
            std::fprintf(stderr, "[redclose] poolCtor=%.1fms runOrder=%.1fms\n",
                         seconds(tp0, tp1) * 1e3, seconds(tp1, tp2) * 1e3);
            std::fprintf(stderr,
                         "[redclose] budget hungry=%lld granted=%lld "
                         "contended=%lld\n",
                         (long long)blas::ScopedReservation::nHungry.load(
                             std::memory_order_relaxed),
                         (long long)blas::ScopedReservation::nBudgetGranted
                             .load(std::memory_order_relaxed),
                         (long long)blas::ScopedReservation::nBudgetContended
                             .load(std::memory_order_relaxed));
            std::fprintf(stderr,
                         "[redclose] budget noTls=%lld nested=%lld "
                         "gate=%lld\n",
                         (long long)blas::ScopedReservation::nNoTls.load(
                             std::memory_order_relaxed),
                         (long long)blas::ScopedReservation::nNest.load(
                             std::memory_order_relaxed),
                         (long long)blas::ScopedReservation::nGate.load(
                             std::memory_order_relaxed));
            std::fprintf(stderr,
                         "[redclose] ancGate nodePool rejects=%lld busyMax=%d "
                         "branchPool rejects=%lld busyMax=%d\n",
                         (long long)nodePool.gateRejects(),
                         nodePool.gateBusyMax(),
                         (long long)branchPool.gateRejects(),
                         branchPool.gateBusyMax());
            GemmColsStats::report();
            reportAncLoopStats();
        }
    }
    sol.timing["REDUCTION"] = seconds(t1, Clock::now());
    colOpsDumpStats();
    if (std::getenv("AAMS_TASK_TIMELINE") != nullptr)
        exec.dumpTimeline("aams_timeline.csv");
    if (opt.verbosity > 0)
        std::cout << "[aams] reduction done: substructures=" << nTreeNodes
                  << " threads=" << threads << "（节点池 " << nodeThreads
                  << "） enhanced_marked=" << enhancedCount
                  << " time=" << sol.timing["REDUCTION"] << "s\n";

    // ---- [9] Reduced Eigensolution Phase：蒸馏 + 一次子空间迭代 + Ritz ----
    const auto t2 = Clock::now();
    DenseMatrix reducedK, reducedM;
    DenseMatrix recoveredVectors;   // 缩聚特征向量（p×m，缩聚坐标）
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    {
        const auto ta = Clock::now();
        if (!saver.assembleSubspaceStiffness(reducedK))
            throw std::runtime_error("子空间刚度组装失败");
        if (!massSaver.assembleSubspaceMass(saver, reducedM))
            throw std::runtime_error("子空间质量组装失败");
        sol.timing["ASSEMBLE_SUBSPACE"] = seconds(ta, Clock::now());
    }
    sol.reducedSize = reducedK.rows();
    if (opt.verbosity > 0) {
        std::cout << "[aams] reduced subspace p=" << sol.reducedSize << "\n";
        if (std::getenv("AAMS_PROFILE") != nullptr) {
            const auto names = {"SolveLocalEigenPair", "GaussConstraint",
                                "GaussKAccum", "GaussMTemp", "GaussMAccum",
                                "GaussReleaseMtemp", "GaussReleaseReducedDiag",
                                "CompressLeft", "CompressUpdate",
                                "CompressRight", "ReleaseTempBlock"};
            const auto secs = exec.profileTaskSeconds();
            const auto cnts = exec.profileTaskCounts();
            std::cerr << "[prof] reduction tasks (sum seconds | count):\n";
            std::size_t i = 0;
            for (const char* nm : names)
                std::cerr << "  " << nm << " = " << secs[i] << "s | "
                          << cnts[i++] << "\n";
        }
    }

    // 频段：[fmin,fmax]（Sturm 检查与特征值表的区间上界 = fmax [R]）。
    const double bandLo =
        par.fminHz > 0.0 ? kTwoPi * kTwoPi * par.fminHz * par.fminHz : -1e300;
    const double bandHi = kTwoPi * kTwoPi * par.fmaxHz * par.fmaxHz;
    if (opt.reducedMethod == "distill") {
        rea::DistillParams dp;
        const auto td = Clock::now();
        dp.fmaxHz = par.fmaxHz;
        dp.fminHz = par.fminHz;
        dp.cutoff2 = par.cutoff2;
        dp.cutoff3 = par.cutoff3;
        dp.shift = par.globalShift;
        dp.subspaceIteration = opt.subspaceIteration;
        dp.subspaceMax = par.subspaceMax;
        dp.extractOrder = par.nev;
        dp.blasThreads = threads;
        rea::DistillResult dr = rea::distillRitz(reducedK, reducedM, dp);
        reducedK.resize(0, 0);
        reducedM.resize(0, 0);
        if (dr.eigenvalues.size() == 0)
            throw std::runtime_error("关注频段内无模态（检查 fmin/fmax）");
        sol.eigenvalues = std::move(dr.eigenvalues);
        recoveredVectors = std::move(dr.eigenvectors);
        sol.timing["DISTILL"] = seconds(td, Clock::now());
        if (opt.verbosity > 0)
            std::cout << "[aams] distillation: p=" << dr.inputSize
                      << " -> p'=" << dr.distilledSize
                      << " modes=" << sol.eigenvalues.size() << "\n";
    } else {
        dense.solveInterval(reducedK, reducedM, bandLo, bandHi,
                            sol.eigenvalues, recoveredVectors);
        reducedK.resize(0, 0);
        reducedM.resize(0, 0);
        if (sol.eigenvalues.size() == 0)
            throw std::runtime_error("关注频段内无模态（检查 fmin/fmax）");
        if (par.nev > 0 && sol.eigenvalues.size() > par.nev) {
            // Abaqus 语义：按窗口提取全部模态，输出截断到 nev [R]。
            sol.eigenvalues = sol.eigenvalues.head(par.nev).eval();
            recoveredVectors = recoveredVectors.leftCols(par.nev).eval();
        }
    }
    sol.timing["REDUCED_EIGENSOLUTION"] = seconds(t2, Clock::now());

    // ---- [9] Recovery Phase：沿树 Ψ 传播恢复 + 反排列（M 归一由
    //      Φ M-正交 + 子空间解 S-正交组合恒等式保证）----
    if (opt.recoverVectors && recoveredVectors.cols() > 0) {
        const auto t3 = Clock::now();
        const int64_t want = recoveredVectors.cols();
        const DenseMatrix& projectedVectors = recoveredVectors;

        std::vector<int64_t> subOffset(static_cast<size_t>(nTreeNodes), 0);
        int64_t acc = 0;
        for (int64_t i = 0; i < nTreeNodes; ++i) {
            EigenPairSaver::View view;
            if (!saver.getEigenPairView(i, view))
                throw std::runtime_error("恢复失败：节点特征对缺失");
            subOffset[static_cast<size_t>(i)] = acc;
            acc += static_cast<int64_t>(view.eigenvalues()->size());
        }
        std::vector<int64_t> dofOffset(static_cast<size_t>(nTreeNodes), 0);
        for (int64_t i = 0; i < nTreeNodes; ++i) {
            const auto& idx = ob.indices(i);
            if (idx.empty()) continue;
            const int64_t off = idx.front();
            for (size_t r = 0; r < idx.size(); ++r)
                if (idx[r] != off + static_cast<int64_t>(r))
                    throw std::runtime_error(
                        "节点 dofs 非连续（剖分重排约定被破坏）");
            dofOffset[static_cast<size_t>(i)] = off;
        }

        std::vector<std::vector<int64_t>> byDepth(
            static_cast<size_t>(maxDepth) + 1);
        for (int64_t i = 0; i < nTreeNodes; ++i)
            byDepth[static_cast<size_t>(tree.ancestors(i).size())]
                .push_back(i);

        // 层并行回代：同层节点只读更浅层（已定稿）的祖先块 zn[a]。
        std::vector<DenseMatrix> zn(static_cast<size_t>(nTreeNodes));
        for (int64_t d = 0; d <= maxDepth; ++d) {
            const auto& layer = byDepth[static_cast<size_t>(d)];
            if (layer.empty()) continue;
            parallelFor(
                0, static_cast<int64_t>(layer.size()),
                [&](int64_t t) {
                    const int64_t i = layer[static_cast<size_t>(t)];
                    const auto& idx = ob.indices(i);
                    const int64_t ni = static_cast<int64_t>(idx.size());
                    EigenPairSaver::View view;
                    if (!saver.getEigenPairView(i, view)) return;
                    const DenseMatrix& phi = *view.eigenvectors();
                    const int64_t m =
                        static_cast<int64_t>(view.eigenvalues()->size());
                    DenseMatrix zi = DenseMatrix::Zero(ni, want);
                    if (m > 0)
                        zi.noalias() = phi * projectedVectors.middleRows(
                                                 subOffset[static_cast<size_t>(i)],
                                                 m);
                    for (const int64_t a : tree.ancestors(i)) {
                        const ColCompMat* psi = gauss.getPsi(i, a);
                        if (psi == nullptr) continue;
                        DenseMatrix prod;
                        if (opsColDenseProd(*psi,
                                            zn[static_cast<size_t>(a)], prod))
                            zi += prod;
                    }
                    zn[static_cast<size_t>(i)] = std::move(zi);
                    for (const int64_t a : tree.ancestors(i))
                        gauss.releasePsi(i, a);
                    saver.release(i);
                },
                opt.numThreads);
        }

        DenseMatrix recovered(n, want);
        for (int64_t i = 0; i < nTreeNodes; ++i) {
            const int64_t ni = static_cast<int64_t>(ob.indices(i).size());
            if (ni == 0) continue;
            recovered.middleRows(dofOffset[static_cast<size_t>(i)], ni) =
                zn[static_cast<size_t>(i)];
        }
        zn.clear();

        // 反排列（P A Pᵀ 坐标 → 原坐标）。
        DenseMatrix eigenvectors = std::move(recovered);
        Vector reorderedColumn(n);
        for (int64_t c = 0; c < want; ++c) {
            reorderedColumn = eigenvectors.col(c);
            const double* source = reorderedColumn.data();
            double* target = eigenvectors.col(c).data();
            for (int64_t j = 0; j < n; ++j)
                target[part.perm[static_cast<size_t>(j)]] = source[j];
        }
        sol.eigenvectors = std::move(eigenvectors);
        sol.timing["RECOVERY"] = seconds(t3, Clock::now());
    } else {
        sol.timing["RECOVERY"] = 0.0;
    }

    trace::dump();
    return sol;
}

}  // namespace aams
