// AMLS 求解器协调器实现 —— PKU 列压缩架构编排。
//
//   (1) step1 剖分与重排：METIS_NodeND 消去树（默认）或递归二分 →
//       分割节点树 + 排列 P；树索引重映射到重排坐标（perm[new] = old 按
//       后序拼接 ⇒ 每节点 dofs 在重排坐标下为连续升序段）。
//   (2) step2 原始子块抽取（OriginalBlocks，原始矩阵只读）→ 消去模块
//       （GaussEliminationModule：Ψ/Mtemp/Schur 更新，列压缩）→ 缩聚
//       （TempBlockManager：μ 创建/传播/右乘）。全部由 NodeTaskExecutor
//       双线程池节点序调度驱动（节点内 11 任务依赖位表 DAG，无阶段屏障）。
//       局部特征求解：统一稠密 dsygvx 区间 (-∞, (2π·γ·maxFreq)²]，
//       叶子读原始对角子块、分割节点读约化对角块。
//   (3) step3 子空间特征问题：K̃ 对角铺特征值（ΦᵀK̃Φ=Λ），S 子空间质量
//       （对角=1 约定）；dsygvx 频段区间求解，extractOrder>0 时截断。
//   (4) step4 恢复：按到根深度分层并行，X_i = Φ_i·z_i + Σ_a Ψ_{i,a}·X_a
//       （祖先层已定稿；Ψ 消费完即释放）；反排列后即为 M-归一特征向量
//       （Φ M-正交 + 子空间解 S-正交的组合恒等式，无事后归一化）。
#include "amls/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "amls/eigen_pair_saver.hpp"
#include "amls/gauss_module.hpp"
#include "amls/node_task_executor.hpp"
#include "amls/original_blocks.hpp"
#include "amls/separator_tree.hpp"
#include "amls/subspace_mass_saver.hpp"
#include "amls/temp_block_manager.hpp"
#include "amls/thread_pool.hpp"
#include "core/logging.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"
#include "linalg/col_ops.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "partition/partitioner.hpp"
#include "partition/external_tree.hpp"
#include "rea/distill_solver.hpp"
#include "rea/tree_builder.hpp"

namespace amls {

namespace {

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

constexpr double kTwoPi = 6.283185307179586476925286766559;

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

/// 树索引重映射到重排坐标：节点 i 的 indices = inv[原始 indices]。
/// 结构（children/parent/角色）不变；id 顺序保持（addNode 按序插入）。
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

/// 节点重编号为后序（祖先 id 恒大于后代 id）。消去模块的 (min,max) 键规约
/// 与多处调度逻辑依赖该约定；recursiveBisection 的分隔节点先于孩子创建，
/// 不满足，统一在此归一。
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

} // namespace

AmlsSolver::Solution AmlsSolver::solve(const PencilT& KM,
                                       const AmlsOptions& opt) {
    Solution sol;
    const AmlsOptions& optN = opt;
    const auto t0 = Clock::now();
    trace::initFromEnv();

    if (!(optN.maxFreq > 0.0))
        throw std::invalid_argument("maxFreq 必须为正（Hz）");
    if (!(optN.minFreq >= 0.0) || !(optN.minFreq < optN.maxFreq))
        throw std::invalid_argument("要求 0 <= minFreq < maxFreq");
    if (!(optN.gainCoefficient > 0.0))
        throw std::invalid_argument("gainCoefficient 必须为正");
    if (optN.extractOrder < 0)
        throw std::invalid_argument("extractOrder 不可为负");
    const int64_t n = KM.K.rows();
    if (n == 0)
        throw std::invalid_argument("空矩阵");

    // OpenBLAS（OpenMP 版）线程与本工程线程池互踩严重：
    // BLAS 永久单线程，并行只走节点级/任务级线程池。
    blas::setNumThreads(1);

    // ---- step1 剖分与重排 ----
    if (optN.leafSize <= 0)
        throw std::invalid_argument("leafSize 必须为正");
    const bool progress = std::getenv("AMLS_PROGRESS") != nullptr;
    auto reportStage = [&](const char* name) {
        if (progress) std::fprintf(stderr, "[solver] stage=%s\n", name);
    };
    reportStage("partition");
    if (optN.partitionMethod != "bisect" && optN.partitionMethod != "nd" &&
        optN.partitionMethod != "external" && optN.partitionMethod != "supernode" &&
        optN.partitionMethod != "rea")
        throw std::invalid_argument(
            "partition 必须为 bisect、nd、external、supernode 或 rea");
    MetisPartitioner<DefaultTraits> partitioner;
    IPartitioner<DefaultTraits>::Result part;
    if (optN.partitionMethod == "external") {
        if (optN.externalTreeFile.empty())
            throw std::invalid_argument("external partition 必须提供 externalTreeFile");
        ExternalTreePartitioner<DefaultTraits> external;
        part = external.fromFile(KM.K + KM.M, optN.externalTreeFile,
                                 optN.leafSize);
    } else if (optN.partitionMethod == "nd") {
        part = partitioner.nestedDissection(KM.K + KM.M, optN.leafSize);
    } else if (optN.partitionMethod == "supernode") {
        part = partitioner.supernodePartition(KM.K + KM.M, optN.maxBlockDofs);
    } else if (optN.partitionMethod == "rea") {
        // Abaqus AMS 风格树：NodeND → etree → exact 超节点 → Collapse2 +
        // AggregateLeaves 聚合。节点 indices 为原始坐标、perm[new]=old，
        // 与 nd 路径同契约（走 remapTreeToReordered 标准重映射）。
        rea::AbaqusTreeOptions ropt;
        ropt.maxBlockDofs = optN.reaMaxBlockDofs;
        ropt.collapseMaxDofs = optN.reaCollapseDofs;
        ropt.leafAggrMaxDofs = optN.reaLeafAggrDofs;
        ropt.dofsPerNode = optN.reaDofsPerNode;
        ropt.superRootDofs = n;   // 允许多根森林合成空 DOF 超根
        part = rea::buildAbaqusTree(KM.K + KM.M, ropt);
    } else {
        part = partitioner.recursiveBisection(KM.K + KM.M, optN.nLevels,
                                              optN.leafSize);
    }
    SeparatorTree tree;
    if (optN.partitionMethod == "supernode") {
        // supernode 树: buildBlockForest 已产出"重排坐标连续 + 后序"的树
        // （节点 indices 是重排坐标连续段, 祖先 id > 后代 id）。
        // 跳过 remapTreeToReordered/renumberPostOrder（那两套对 nd/bisect 的
        // 原始坐标树必要, 对 supernode 会二次映射/重排破坏不变式）。
        tree = part.tree;
    } else {
        tree = renumberPostOrder(remapTreeToReordered(part.tree, part.perm));
    }
    lastTreeNodes_ = tree.nNodes();
    const int64_t nTreeNodes = tree.nNodes();
    PencilT reordered;
    reordered.K = permuteSymmetric(KM.K, part.perm);
    reordered.M = permuteSymmetric(KM.M, part.perm);
    const auto t1 = Clock::now();
    sol.timing["partition"] = seconds(t0, t1);

    // ---- step2 消去 + 缩聚（NodeTaskExecutor 双池节点序）----
    reportStage("condense");
    OriginalBlocks ob(reordered, tree);
    reordered.K.resize(0, 0);   // 重排副本此后只读，提前释放
    reordered.M.resize(0, 0);

    GaussEliminationModule gauss;
    if (!gauss.initialize(ob))
        throw std::runtime_error("消去模块初始化失败");
    EigenPairSaver saver;
    if (!saver.initialize(nTreeNodes, true))
        throw std::runtime_error("特征对存储初始化失败");
    SubspaceMassSaver massSaver;
    if (!massSaver.initialize(tree))
        throw std::runtime_error("子空间质量存储初始化失败");
    TempBlockManager temp;
    if (!temp.initialize(tree, &gauss, &saver, &massSaver))
        throw std::runtime_error("临时块管理器初始化失败");
    NodeTaskStateStore states(nTreeNodes);

    LapackDenseEigensolver<DefaultTraits> dense;
    const double localCut =
        std::pow(kTwoPi * optN.gainCoefficient * optN.maxFreq, 2.0);
    NodeTaskExecutor::LocalEigenConfig cfg;
    cfg.dense = &dense;
    cfg.blocks = &ob;
    cfg.lambdaHi = localCut;
    NodeTaskExecutor exec(tree, gauss, temp, saver, states, cfg);

    const int threads =
        optN.numThreads > 0
            ? optN.numThreads
            : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int nodeThreads = std::max(1, static_cast<int>(
        std::llround(static_cast<double>(threads) / 1.5)));
    {
        ThreadPool nodePool(nodeThreads);
        ThreadPool branchPool(threads);
        if (!exec.runNodeOrder(nodePool, branchPool))
            throw std::runtime_error("节点任务 DAG 执行失败（详见日志）");
    }
    const auto t2 = Clock::now();
    sol.timing["phase2"] = seconds(t1, t2);
    sol.timing["condense"] = sol.timing["phase2"];
    if (optN.verbosity > 0)
        std::cout << "[amls] 消去+缩聚完成: 节点=" << nTreeNodes
                  << " 线程=" << threads << "（节点池 " << nodeThreads << "）"
                  << " 耗时=" << sol.timing["phase2"] << "s\n";

    // ---- step3 子空间特征问题 ----
    reportStage("reduced");
    DenseMatrix reducedK, reducedM;
    if (!saver.assembleSubspaceStiffness(reducedK))
        throw std::runtime_error("子空间刚度组装失败");
    if (!massSaver.assembleSubspaceMass(saver, reducedM))
        throw std::runtime_error("子空间质量组装失败");
    sol.reducedSize = reducedK.rows();
    if (std::getenv("AMLS_DUMP_SUBSPACE") != nullptr) {
        std::fprintf(stderr, "[subspace] p=%lld\n",
                     static_cast<long long>(sol.reducedSize));
        for (int64_t i = 0; i < sol.reducedSize; ++i) {
            std::fprintf(stderr, "[subspace] Mdiag[%lld]=%.16e\n",
                         static_cast<long long>(i), reducedM(i, i));
        }
        const int64_t p = sol.reducedSize;
        // 导出 K/B 到文件供外部分析 (AMLS_DUMP_SUBSPACE_FILE)
        if (const char* dumpFile = std::getenv("AMLS_DUMP_SUBSPACE_FILE")) {
            std::ofstream fk(std::string(dumpFile) + ".K.txt");
            for (int64_t r = 0; r < p; ++r) {
                for (int64_t c = 0; c < p; ++c)
                    fk << reducedK(r, c) << (c + 1 == p ? "\n" : " ");
            }
            std::ofstream fb(std::string(dumpFile) + ".B.txt");
            for (int64_t r = 0; r < p; ++r) {
                for (int64_t c = 0; c < p; ++c)
                    fb << reducedM(r, c) << (c + 1 == p ? "\n" : " ");
            }
            std::fprintf(stderr, "[subspace] dumped K/B to %s.K.txt/.B.txt\n",
                         dumpFile);
        }
        // 最大特征值（幂法）
        Eigen::VectorXd x = Eigen::VectorXd::Random(p);
        x.normalize();
        double lamMax = 0.0;
        for (int k = 0; k < 300; ++k) {
            Eigen::VectorXd y = reducedM * x;
            lamMax = x.dot(y);
            const double c = y.norm();
            if (c < 1e-300) break;
            y /= c;
            x = y;
        }
        // 最小特征值（反幂法: shift 0 求 M^{-1} 最大 = 1/lamMin）
        Eigen::LLT<Eigen::MatrixXd> llt(reducedM);
        Eigen::VectorXd x2 = Eigen::VectorXd::Random(p);
        x2.normalize();
        double invLamMin = 0.0;
        for (int k = 0; k < 300; ++k) {
            Eigen::VectorXd y = llt.solve(x2);
            invLamMin = x2.dot(y);
            const double c = y.norm();
            if (c < 1e-300) break;
            y /= c;
            x2 = y;
        }
        std::fprintf(stderr, "[subspace] M λmax≈%.16e λmin≈%.16e\n",
                     lamMax, 1.0 / invLamMin);
        std::fprintf(stderr, "[subspace] M 对称误差=%.3e\n",
                     (reducedM - reducedM.transpose()).cwiseAbs().maxCoeff());
        // K 正定性: LLT
        Eigen::LLT<Eigen::MatrixXd> lk(reducedK);
        std::fprintf(stderr, "[subspace] K LLT info=%s\n",
                     lk.info() == Eigen::Success ? "OK" : "FAIL");
        std::fprintf(stderr, "[subspace] M cond≈%.3e\n", lamMax * invLamMin);
    }
    if (optN.verbosity > 0)
        std::cout << "[amls] 子空间规模 p=" << sol.reducedSize << "\n";
    if (sol.reducedSize > optN.denseProjectedMax && optN.verbosity > 0)
        std::cout << "[amls] 警告: p=" << sol.reducedSize
                  << " 超过 denseProjectedMax=" << optN.denseProjectedMax
                  << "（P12 切换点；仍用稠密求解）\n";

    // minFreq=0 时取 (-∞, λ_hi]：刚体（λ=0）与负谱模态一并保留；
    // minFreq>0 时按频段下界过滤。
    const double bandLo = optN.minFreq > 0.0
                              ? std::pow(kTwoPi * optN.minFreq, 2.0)
                              : -1e300;
    const double bandHi = std::pow(kTwoPi * optN.maxFreq, 2.0);
    Vector projectedEigenvalues;
    DenseMatrix projectedVectors;
    if (optN.reducedSolver == "rea") {
        // Abaqus AMS 语义：蒸馏过滤 → 一次子空间迭代 → Ritz → 频段/nev 截断
        // → M_r 正交归一（pme_SmpReducedEigenSolver 的稠密适配）。
        rea::DistillParams dp;
        dp.fmaxHz = optN.maxFreq;
        dp.fminHz = optN.minFreq;
        dp.cutoff2 = optN.cutoff2 > 0.0 ? optN.cutoff2 : optN.gainCoefficient;
        dp.cutoff3 = optN.cutoff3 > 0.0 ? optN.cutoff3 : optN.gainCoefficient;
        dp.shift = optN.globalShift;
        dp.subspaceIteration = optN.subspaceIteration;
        dp.extractOrder = optN.extractOrder;
        const rea::DistillResult dr =
            rea::distillRitz(reducedK, reducedM, dp);
        reducedK.resize(0, 0);
        reducedM.resize(0, 0);
        if (dr.eigenvalues.size() == 0)
            throw std::runtime_error("关注频段内无模态（检查 minFreq/maxFreq）");
        projectedEigenvalues = dr.eigenvalues;
        projectedVectors = dr.eigenvectors;
        if (optN.verbosity > 0)
            std::cout << "[amls] 蒸馏: p=" << dr.inputSize
                      << " → p'=" << dr.distilledSize
                      << " → 模态=" << dr.eigenvalues.size() << "\n";
    } else {
        dense.solveInterval(reducedK, reducedM, bandLo, bandHi,
                            projectedEigenvalues, projectedVectors);
        reducedK.resize(0, 0);
        reducedM.resize(0, 0);
        if (projectedEigenvalues.size() == 0)
            throw std::runtime_error("关注频段内无模态（检查 minFreq/maxFreq）");
        if (optN.extractOrder > 0 &&
            projectedEigenvalues.size() > optN.extractOrder) {
            projectedEigenvalues =
                projectedEigenvalues.head(optN.extractOrder).eval();
            projectedVectors =
                projectedVectors.leftCols(optN.extractOrder).eval();
        }
    }
    const auto t3 = Clock::now();
    sol.timing["reduced"] = seconds(t2, t3);

    // ---- step4 恢复（可选）----
    if (optN.recoverVectors && projectedVectors.cols() > 0) {
        reportStage("backtransform");
        const int64_t want = projectedVectors.cols();
        // 子空间行偏移（按节点 id 序，与 saver 组装一致）。
        std::vector<int64_t> subOffset(static_cast<size_t>(nTreeNodes), 0);
        int64_t acc = 0;
        for (int64_t i = 0; i < nTreeNodes; ++i) {
            EigenPairSaver::View view;
            if (!saver.getEigenPairView(i, view))
                throw std::runtime_error("恢复失败：节点特征对缺失");
            subOffset[static_cast<size_t>(i)] = acc;
            acc += static_cast<int64_t>(view.eigenvalues()->size());
        }

        // 重排坐标 dof 偏移（节点 dofs = 连续升序段）。
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

        int64_t maxDepth = 0;
        for (int64_t i = 0; i < nTreeNodes; ++i)
            maxDepth = std::max(
                maxDepth, static_cast<int64_t>(tree.ancestors(i).size()));
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
                        if (opsColDenseProd(
                                *psi,
                                zn[static_cast<size_t>(a)], prod))
                            zi += prod;
                    }
                    zn[static_cast<size_t>(i)] = std::move(zi);
                    // 本节点的 Ψ 行块与特征对仅此处消费，用后即放。
                    for (const int64_t a : tree.ancestors(i))
                        gauss.releasePsi(i, a);
                    saver.release(i);
                },
                optN.numThreads);
        }

        // 拼装 + 反排列（P A Pᵀ 坐标 → 原坐标）。
        DenseMatrix recovered(n, want);
        for (int64_t i = 0; i < nTreeNodes; ++i) {
            const int64_t ni = static_cast<int64_t>(ob.indices(i).size());
            if (ni == 0) continue;
            recovered.middleRows(dofOffset[static_cast<size_t>(i)], ni) =
                zn[static_cast<size_t>(i)];
        }
        zn.clear();
        const auto t4 = Clock::now();
        sol.timing["backtransform"] = seconds(t3, t4);

        DenseMatrix eigenvectors = std::move(recovered);
        Vector reorderedColumn(n);
        for (int64_t c = 0; c < want; ++c) {
            reorderedColumn = eigenvectors.col(c);
            const double* source = reorderedColumn.data();
            double* target = eigenvectors.col(c).data();
            for (int64_t j = 0; j < n; ++j)
                target[part.perm[static_cast<size_t>(j)]] = source[j];
        }
        const auto tNormalizeEnd = Clock::now();
        sol.timing["normalize"] = seconds(t4, tNormalizeEnd);
        sol.eigenvectors = std::move(eigenvectors);
    } else {
        sol.timing["backtransform"] = 0.0;
    }

    sol.eigenvalues = std::move(projectedEigenvalues);
    trace::dump();
    return sol;
}

} // namespace amls
