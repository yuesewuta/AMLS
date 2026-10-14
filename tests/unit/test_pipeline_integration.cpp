// 集成管线子空间同余验证：真实剖分树（ND + bisect）上，显式构造变换
// T（Φ + 约束传播），检验 TᵀKT ≈ Λ 对角（约束折叠恒等式）与
// TᵀMT == 组装子空间质量（μ 块正确性）；全保留下子空间谱 == 全空间谱。
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "amls/eigen_pair_saver.hpp"
#include "amls/gauss_module.hpp"
#include "amls/node_task_executor.hpp"
#include "amls/original_blocks.hpp"
#include "amls/separator_tree.hpp"
#include "amls/subspace_mass_saver.hpp"
#include "amls/temp_block_manager.hpp"
#include "io/case_generator.hpp"
#include "linalg/col_ops.hpp"
#include "linalg/dense_eigensolver.hpp"
#include "partition/partitioner.hpp"

using namespace amls;
using DM = DefaultTraits::DenseMatrix;

namespace {

DefaultTraits::SparseMatrix permuteSym(
    const DefaultTraits::SparseMatrix& A,
    const std::vector<int64_t>& perm) {
    using Index = DefaultTraits::Index;
    std::vector<Index> inv(static_cast<size_t>(A.rows()));
    for (Index j = 0; j < static_cast<Index>(perm.size()); ++j)
        inv[static_cast<size_t>(perm[static_cast<size_t>(j)])] = j;
    std::vector<Eigen::Triplet<double, Index>> trips;
    trips.reserve(static_cast<size_t>(A.nonZeros()));
    for (Index c = 0; c < A.outerSize(); ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
            trips.emplace_back(inv[static_cast<size_t>(it.row())],
                               inv[static_cast<size_t>(it.col())], it.value());
    DefaultTraits::SparseMatrix B(A.rows(), A.cols());
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}

SeparatorTree remap(const SeparatorTree& tree,
                    const std::vector<int64_t>& perm) {
    std::vector<int64_t> inv(perm.size());
    for (size_t j = 0; j < perm.size(); ++j)
        inv[static_cast<size_t>(perm[j])] = static_cast<int64_t>(j);
    SeparatorTree out;
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        std::vector<int64_t> idx;
        for (int64_t old : tree.node(i).indices)
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

SeparatorTree renumberPost(const SeparatorTree& tree) {
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
        for (int64_t c : ch) kids.push_back(map[static_cast<size_t>(c)]);
        out.setChildren(map[static_cast<size_t>(old)], kids);
        for (int64_t c : kids)
            out.setParent(c, map[static_cast<size_t>(old)]);
    }
    out.relabelRoot();
    return out;
}

struct PipeResult {
    DM subK, subM;              // 组装输出
    DM kRef, mRef;              // TᵀKT / TᵀMT
    SeparatorTree tree;         // 剖分树（结构性质检查用）
    std::vector<int64_t> subOff;    // 节点 → 子空间段偏移
    std::vector<int64_t> nodeOff;   // 节点 → dof 段偏移（重排坐标）
    DM T;                           // 显式变换矩阵（诊断用）
    DM Kd, Md;                      // 重排后稠密 pencil（诊断用）
    int64_t n = 0, p = 0;
};

PipeResult runPipe(const PencilPair& pair, const std::string& method,
                   int64_t leafSize, double lambdaHi) {
    MetisPartitioner<DefaultTraits> partitioner;
    auto part = method == "nd"
                    ? partitioner.nestedDissection(pair.K + pair.M, leafSize)
                    : partitioner.recursiveBisection(pair.K + pair.M, 6,
                                                     leafSize);
    SeparatorTree tree = renumberPost(remap(part.tree, part.perm));
    REQUIRE(tree.validate(pair.K.rows()));

    Pencil<DefaultTraits> reordered;
    reordered.K = permuteSym(pair.K, part.perm);
    reordered.M = permuteSym(pair.M, part.perm);
    const int64_t n = reordered.K.rows();

    OriginalBlocks ob(reordered, tree);
    GaussEliminationModule gauss;
    REQUIRE(gauss.initialize(ob));
    EigenPairSaver saver;
    REQUIRE(saver.initialize(tree.nNodes(), true));
    SubspaceMassSaver massSaver;
    REQUIRE(massSaver.initialize(tree));
    TempBlockManager temp;
    REQUIRE(temp.initialize(tree, &gauss, &saver, &massSaver));
    NodeTaskStateStore states(tree.nNodes());

    LapackDenseEigensolver<DefaultTraits> dense;
    NodeTaskExecutor::LocalEigenConfig cfg;
    cfg.dense = &dense;
    cfg.blocks = &ob;
    cfg.lambdaHi = lambdaHi;
    NodeTaskExecutor exec(tree, gauss, temp, saver, states, cfg);
    ThreadPool nodePool(4), branchPool(4);
    REQUIRE(exec.runNodeOrder(nodePool, branchPool));

    PipeResult out;
    out.n = n;
    out.tree = tree;
    REQUIRE(saver.assembleSubspaceStiffness(out.subK));
    REQUIRE(massSaver.assembleSubspaceMass(saver, out.subM));
    const int64_t p = out.subK.rows();
    out.p = p;

    // dof / 子空间偏移
    std::vector<int64_t> dofOff(static_cast<size_t>(tree.nNodes()), 0);
    std::vector<int64_t> subOff(static_cast<size_t>(tree.nNodes()), 0);
    int64_t acc = 0;
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const auto& idx = ob.indices(i);
        REQUIRE(!idx.empty());
        dofOff[static_cast<size_t>(i)] = idx.front();
        for (size_t r = 0; r < idx.size(); ++r)
            REQUIRE(idx[r] == idx.front() + static_cast<int64_t>(r));
        subOff[static_cast<size_t>(i)] = acc;
    }
    out.nodeOff = dofOff;
    acc = 0;
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        EigenPairSaver::View v;
        REQUIRE(saver.getEigenPairView(i, v));
        subOff[static_cast<size_t>(i)] = acc;
        acc += static_cast<int64_t>(v.eigenvalues()->size());
    }
    out.subOff = subOff;

    // 显式 T：X_i = Φ_i z_i + Σ_{a∈anc(i)} Ψ_{i,a} X_a（与 solver.cpp 恢复
    // 完全同式）。Ψ_{child,anc} 把祖先 dof 映到孩子 dof（totalCols=祖先规模）。
    // 必须对**每个祖先对**都传播——叶子可能直接耦合到非父辈分隔符
    // （K_{leaf,root}≠0），只沿父子边传播会漏项（grid bisect 即此情形）。
    // 自顶向下处理：轮到节点 d 时其全部祖先行已定稿（拓扑序保证）。
    DM T = DM::Zero(n, p);
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        EigenPairSaver::View v;
        REQUIRE(saver.getEigenPairView(i, v));
        const DM& phi = *v.eigenvectors();
        const int64_t m = static_cast<int64_t>(v.eigenvalues()->size());
        for (int64_t k = 0; k < m; ++k)
            T.middleRows(dofOff[static_cast<size_t>(i)], phi.rows())
                .col(subOff[static_cast<size_t>(i)] + k) = phi.col(k);
    }
    std::vector<int64_t> topDown = tree.postOrder();
    std::reverse(topDown.begin(), topDown.end());
    for (int64_t d : topDown) {
        const int64_t nD = static_cast<int64_t>(
            ob.indices(static_cast<size_t>(d)).size());
        for (const int64_t a : tree.ancestors(d)) {
            const ColCompMat* psi = gauss.getPsi(d, a);
            if (psi == nullptr) FAIL("getPsi(" << d << "," << a << ") 为空");
            const int64_t nA = static_cast<int64_t>(
                ob.indices(static_cast<size_t>(a)).size());
            DM contrib;
            if (!opsColDenseProd(*psi,
                                 T.middleRows(dofOff[static_cast<size_t>(a)],
                                              nA),
                                 contrib))
                FAIL("opsColDenseProd psi(" << d << "," << a << ") 失败");
            T.middleRows(dofOff[static_cast<size_t>(d)], contrib.rows()) +=
                contrib;
        }
    }

    // 参考同余
    DM Kd(n, n), Md(n, n);
    Kd = DM(reordered.K);
    Md = DM(reordered.M);
    out.kRef = T.transpose() * Kd * T;
    out.T = T;
    out.mRef = T.transpose() * Md * T;
    out.Kd = Kd;
    out.Md = Md;
    return out;
}

} // namespace

TEST_CASE("管线集成:path:200 ND 树子空间同余", "[pipeline][integration]") {
    auto pair = pathGraph(200);
    PipeResult r = runPipe(pair, "nd", 50, 1e300);
    // K 同余 = Λ 对角（非对角恒 0）
    for (int64_t i = 0; i < r.p; ++i)
        for (int64_t j = 0; j < r.p; ++j) {
            if (i == j) continue;
            INFO("K offdiag (" << i << "," << j << ")="
                               << r.kRef(i, j));
            CHECK(std::abs(r.kRef(i, j)) <= 1e-9 * std::max(1.0, r.subK(i, i)));
        }
    double kDiagErr = 0.0;
    for (int64_t i = 0; i < r.p; ++i)
        kDiagErr = std::max(kDiagErr,
                            std::abs(r.kRef(i, i) - r.subK(i, i)));
    CHECK(kDiagErr <= 1e-9);
    // M 同余
    double mErr = 0.0;
    for (int64_t i = 0; i < r.p; ++i)
        for (int64_t j = 0; j < r.p; ++j)
            mErr = std::max(mErr, std::abs(r.mRef(i, j) - r.subM(i, j)));
    INFO("M 同余最大误差=" << mErr);
    CHECK(mErr <= 1e-9);
}

TEST_CASE("管线集成:path:2000 单自由度分隔节点子空间同余", "[pipeline][integration]") {
    // 回归 path:2000 fuser=1 场景的树形：两叶 ~1000 + 单 dof 分隔节点。
    auto pair = pathGraph(2000);
    PipeResult r = runPipe(pair, "nd", 500, 1e300);
    double kOffMax = 0.0, kDiagErr = 0.0, mErr = 0.0;
    for (int64_t i = 0; i < r.p; ++i) {
        kDiagErr = std::max(kDiagErr,
                            std::abs(r.kRef(i, i) - r.subK(i, i)));
        for (int64_t j = 0; j < r.p; ++j) {
            if (i != j) kOffMax = std::max(kOffMax, std::abs(r.kRef(i, j)));
            mErr = std::max(mErr, std::abs(r.mRef(i, j) - r.subM(i, j)));
        }
    }
    INFO("K 对角误差=" << kDiagErr << " K 非对角最大=" << kOffMax);
    INFO("M 同余最大误差=" << mErr);
    // 定位最大 K 误差位置与所属节点
    int64_t bi = -1, bj = -1;
    double bVal = 0.0;
    bool bDiag = false;
    for (int64_t i = 0; i < r.p; ++i)
        for (int64_t j = 0; j < r.p; ++j) {
            const double e = i == j
                                 ? std::abs(r.kRef(i, j) - r.subK(i, j))
                                 : std::abs(r.kRef(i, j));
            if (e > bVal) { bVal = e; bi = i; bj = j; bDiag = i == j; }
        }
    INFO("K 最大误差 " << bVal << " @(" << bi << "," << bj << ") diag="
                       << bDiag << " kRef=" << r.kRef(bi, bj)
                       << " subK=" << r.subK(bi, bj));
    CHECK(bVal <= 1e-9);
}

TEST_CASE("管线集成:约束恒等式抽查 1dof 分隔", "[pipeline][identity]") {
    // K̂_ii·Ψ_{i,a} + K̃_{i,a} == 0 逐节点祖先边抽查（path:2000 ND，1 dof 分隔）。
    auto pair = pathGraph(2000);
    MetisPartitioner<DefaultTraits> partitioner;
    auto part = partitioner.nestedDissection(pair.K + pair.M, 500);
    SeparatorTree tree = renumberPost(remap(part.tree, part.perm));
    Pencil<DefaultTraits> reordered;
    reordered.K = permuteSym(pair.K, part.perm);
    reordered.M = permuteSym(pair.M, part.perm);

    OriginalBlocks ob(reordered, tree);
    GaussEliminationModule gauss;
    REQUIRE(gauss.initialize(ob));
    EigenPairSaver saver;
    REQUIRE(saver.initialize(tree.nNodes(), true));
    SubspaceMassSaver massSaver;
    REQUIRE(massSaver.initialize(tree));
    TempBlockManager temp;
    REQUIRE(temp.initialize(tree, &gauss, &saver, &massSaver));
    NodeTaskStateStore states(tree.nNodes());
    LapackDenseEigensolver<DefaultTraits> dense;
    NodeTaskExecutor::LocalEigenConfig cfg;
    cfg.dense = &dense;
    cfg.blocks = &ob;
    cfg.lambdaHi = 1e300;
    NodeTaskExecutor exec(tree, gauss, temp, saver, states, cfg);
    ThreadPool nodePool(4), branchPool(4);
    REQUIRE(exec.runNodeOrder(nodePool, branchPool));

    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        for (int64_t a : tree.ancestors(i)) {
            const ColCompMat* psi = gauss.getPsi(i, a);
            REQUIRE(psi != nullptr);
            const DM Kh = DM(ob.diagK(i));  // 叶子无后代，K̂=原始对角
            const DM Kcouple = DM(ob.subK(i, a));
            REQUIRE(Kcouple.cols() == psi->totalCols());
            REQUIRE(Kcouple.rows() == Kh.rows());
            DM psiD = DM::Zero(Kh.rows(), psi->totalCols());
            for (int64_t c = 0; c < psi->totalCols(); ++c) {
                const int64_t sc = psi->storageCol(c);
                if (sc >= 0) psiD.col(c) = psi->dense().col(sc);
            }
            DM resid = Kh * psiD + Kcouple;
            INFO("边(" << i << "," << a << ") 恒等残差=" << resid.norm()
                       << " Kh=" << Kh.rows() << "x" << Kh.cols());
            CHECK(resid.norm() <= 1e-10 * std::max(1.0, Kh.norm()));
            // 与暴力 -K̂⁻¹K̃ 对拍
            Eigen::LLT<DM> llt(Kh);
            DM psiTrue = -llt.solve(Kcouple);
            DM dpsi = psiD - psiTrue;
            // 分层定位：直接 sytrf/sytrs 路径（绕过 col-comp）
            DM lu = Kh;
            std::vector<int32_t> ipiv;
            REQUIRE(sytrfFactor(lu, ipiv) == 0);
            DM b = Kcouple;
            REQUIRE(sytrsSolve(static_cast<int32_t>(lu.rows()), lu, ipiv,
                               b.data(), 1) == 0);
            DM dLapack = -b - psiTrue;
            DM resid2 = Kh * b - Kcouple;  // sytrs 解 b=K̂⁻¹K̃，残差对 b 检验
            INFO("边(" << i << "," << a << ") 暴力Ψ差范数=" << dpsi.norm()
                       << " Ψ范数=" << psiD.norm()
                       << " 暴力范数=" << psiTrue.norm()
                       << " | lapack直解差=" << dLapack.norm()
                       << " lapack残差=" << resid2.norm()
                       << " Kh范数=" << Kh.norm()
                       << " Kcouple范数=" << Kcouple.norm());
            CHECK(dpsi.norm() <= 1e-9);
            CHECK(dLapack.norm() <= 1e-9);
            CHECK(resid2.norm() <= 1e-10 * std::max(1.0, Kh.norm()));
        }
    }
}

TEST_CASE("管线集成:结构性质 P3/P8 + M 同余（grid bisect 全保留）", "[pipeline][structure]") {
    // 原 SubspaceCondensation 结构性质测试移植到新管线：
    // P3: 每节点局部特征值升序；P8: 全保留 ⇒ M̃ 对角 = I；
    // M̃ 同余 subM == TᵀMT（此前只在 path 图上验证）。
    // 注：旧架构 P4「M̃ 非对角块只出现在祖先对之间」是旧基底的性质——
    // PKU 列压缩基底中 T 的列在祖先 dof 段上有累积分量，兄弟列经分隔符
    // 质量耦合产生真实非零块（同基底变换不改变谱，投影谱用例覆盖）。
    auto pair = gridGraph2d(8, 8);
    PipeResult r = runPipe(pair, "bisect", 20, 1e300);
    const auto& tree = r.tree;
    // P8: 对角恒 1（PKU 约定，assembleSubspaceMass 保证）
    for (int64_t i = 0; i < r.p; ++i)
        CHECK(std::abs(r.subM(i, i) - 1.0) <= 1e-12);
    // P3: 节点段内 subK 对角 = 局部特征值，升序
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const int64_t o = r.subOff[static_cast<size_t>(i)];
        const int64_t k = static_cast<int64_t>(tree.node(i).indices.size());
        for (int64_t j = 1; j < k; ++j)
            CHECK(r.subK(o + j, o + j) >= r.subK(o + j - 1, o + j - 1));
    }
    // M̃ 同余：subM == TᵀMT
    DM mErr = (r.subM - r.mRef).cwiseAbs();
    DM kErr = (r.subK - r.kRef).cwiseAbs();
    CHECK(mErr.maxCoeff() <= 1e-9);
    CHECK(kErr.maxCoeff() <= 1e-9);
    // 独立参照：全 K 稠密 Schur 补构造根精确静态响应 B = -K_bb⁻¹·K_bS
    // （S=根 dof 段），验证 (a) T 折叠 == 精确响应、(b) Schur pencil 特征
    // 值 == subK 根段、(c) 精确响应下 Φ 的 M-归一化成立。
    {
        const int64_t nNodes = tree.nNodes();
        const int64_t root = nNodes - 1;   // postorder ⇒ 根在末位
        const int64_t o6 = r.nodeOff[static_cast<size_t>(root)];
        const int64_t k6 = static_cast<int64_t>(
            tree.node(root).indices.size());
        const int64_t nb = r.n - k6;
        std::vector<int64_t> bRows;
        bRows.reserve(static_cast<size_t>(nb));
        for (int64_t i = 0; i < r.n; ++i)
            if (i < o6 || i >= o6 + k6) bRows.push_back(i);
        DM Kbb(nb, nb), Kbr(nb, k6);
        for (int64_t a = 0; a < nb; ++a) {
            for (int64_t c = 0; c < nb; ++c)
                Kbb(a, c) = r.Kd(bRows[static_cast<size_t>(a)],
                                 bRows[static_cast<size_t>(c)]);
            for (int64_t c = 0; c < k6; ++c)
                Kbr(a, c) = r.Kd(bRows[static_cast<size_t>(a)], o6 + c);
        }
        Eigen::LLT<DM> llt(Kbb);
        REQUIRE(llt.info() == Eigen::Success);
        const DM B = -llt.solve(Kbr);
        // (a) 折叠检查：r.T 的根列响应段（b 行）== B·Φ_T
        const DM phiT = r.T.middleRows(o6, k6).middleCols(
            r.subOff[static_cast<size_t>(root)], k6);
        const DM respT = r.T.middleCols(r.subOff[static_cast<size_t>(root)],
                                        k6);
        DM dFold(nb, k6);
        for (int64_t a = 0; a < nb; ++a)
            dFold.row(a) = respT.row(bRows[static_cast<size_t>(a)]);
        DM foldErr = dFold - B * phiT;
        CHECK(foldErr.norm() <=
              1e-10 * std::max(1.0, dFold.norm()));
        // (b)(c) 精确 Schur pencil
        DM Tex = DM::Zero(r.n, k6);
        for (int64_t a = 0; a < nb; ++a)
            Tex.row(bRows[static_cast<size_t>(a)]) = B.row(a);
        for (int64_t c = 0; c < k6; ++c) Tex(o6 + c, c) = 1.0;
        DM kEx = Tex.transpose() * r.Kd * Tex;
        DM mEx = Tex.transpose() * r.Md * Tex;
        Eigen::GeneralizedSelfAdjointEigenSolver<DM> esEx(kEx, mEx);
        REQUIRE(esEx.info() == Eigen::Success);
        double lamErr = 0.0;
        for (int64_t c = 0; c < k6; ++c)
            lamErr = std::max(lamErr,
                std::abs(esEx.eigenvalues()[static_cast<int>(c)] -
                         r.subK(r.subOff[static_cast<size_t>(root)] + c,
                                r.subOff[static_cast<size_t>(root)] + c)));
        CHECK(lamErr <=
              1e-9 * std::max(1.0, kEx.diagonal().cwiseAbs().maxCoeff()));
        DM mNorm = phiT.transpose() * mEx * phiT;
        CHECK((mNorm - DM::Identity(k6, k6)).cwiseAbs().maxCoeff() <= 1e-9);
    }
}

TEST_CASE("管线集成:全模态投影谱 == 闭式谱（grid）", "[pipeline][projection]") {
    // 全保留装配子空间 (subK, subM) 的广义特征谱精确再现闭式谱。
    auto pair = gridGraph2d(10, 10);
    PipeResult r = runPipe(pair, "bisect", 20, 1e300);
    REQUIRE(r.p == pair.K.rows());   // 全保留 ⇒ p = n
    Eigen::GeneralizedSelfAdjointEigenSolver<DM> es(r.subK, r.subM);
    REQUIRE(es.info() == Eigen::Success);
    const auto ref = gridGraph2dSpectrum(10, 10);
    for (int64_t j = 0; j < 10; ++j)
        REQUIRE(std::abs(es.eigenvalues()[static_cast<int>(j)] - ref[j]) <=
                1e-10 * std::max(1.0, std::abs(ref[j])));
}
