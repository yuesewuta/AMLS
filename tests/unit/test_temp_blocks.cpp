// TempBlockManager + EigenPairSaver + SubspaceMassSaver 暴力对拍单元测试。
// 稠密参考按同一缩聚数学独立复算 μ / S 块，与列压缩模块产物逐项对拍；
// 另含完整模态下的子空间全谱等价检验与 release / 就绪门控语义。
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "amls/eigen_pair_saver.hpp"
#include "amls/gauss_module.hpp"
#include "amls/original_blocks.hpp"
#include "amls/subspace_mass_saver.hpp"
#include "amls/temp_block_manager.hpp"

using namespace amls;
using Dense = DefaultTraits::DenseMatrix;
using SM = DefaultTraits::SparseMatrix;

namespace {

/// 手搓树：3 叶 + 2 分割（与 test_gauss_module 同构）。
SeparatorTree makeTree() {
    SeparatorTree tree;
    tree.addNode(NodeRole::Leaf, {0, 1});        // 0
    tree.addNode(NodeRole::Leaf, {2, 3});        // 1
    tree.addNode(NodeRole::Separator, {4});      // 2（孩子 0,1）
    tree.addNode(NodeRole::Leaf, {5, 6});        // 3
    tree.addNode(NodeRole::Root, {7});           // 4（孩子 2,3）
    tree.setChildren(2, 0, 1);
    tree.setChildren(4, 2, 3);
    tree.setParent(0, 2);
    tree.setParent(1, 2);
    tree.setParent(2, 4);
    tree.setParent(3, 4);
    return tree;
}

/// 深链：0→2→4→6 三跳 μ 传播；每层分割节点至少 2 孩子（validate 约束）。
SeparatorTree makeChainTree() {
    SeparatorTree tree;
    tree.addNode(NodeRole::Leaf, {0, 1});        // 0
    tree.addNode(NodeRole::Leaf, {2});           // 1
    tree.addNode(NodeRole::Separator, {3});      // 2（孩子 0,1）
    tree.addNode(NodeRole::Leaf, {4});           // 3
    tree.addNode(NodeRole::Separator, {5});      // 4（孩子 2,3）
    tree.addNode(NodeRole::Leaf, {6, 7});        // 5
    tree.addNode(NodeRole::Root, {8});           // 6（孩子 4,5）
    tree.setChildren(2, 0, 1);
    tree.setChildren(4, 2, 3);
    tree.setChildren(6, 4, 5);
    tree.setParent(0, 2);
    tree.setParent(1, 2);
    tree.setParent(2, 4);
    tree.setParent(3, 4);
    tree.setParent(4, 6);
    tree.setParent(5, 6);
    return tree;
}

/// 按树允许结构生成随机对称稠密矩阵（对角块 + (后代,祖先) 对，块内全稠密
/// 保证列压缩布局覆盖全部 n_a 列）。
Dense buildStructured(const SeparatorTree& tree, int64_t n, double scale,
                      uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    Dense a = Dense::Zero(n, n);
    auto fill = [&](int64_t i, int64_t j) {
        for (int64_t p : tree.node(i).indices)
            for (int64_t q : tree.node(j).indices) {
                const double v = dist(rng);
                a(p, q) = v;
                a(q, p) = v;
            }
    };
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        fill(i, i);
        for (int64_t a : tree.ancestors(i)) fill(i, a);
    }
    // 对角占优保证 K̂ 局部可分解（对称正定）且 M 正定
    a.diagonal().array() += scale * static_cast<double>(n) * 2.0;
    return a;
}

SM toSparse(const Dense& d) {
    SM s = d.sparseView();
    s.makeCompressed();
    return s;
}

/// 稠密暴力参考：同一消去数学，全对称稠密块表逐步复算（与 test_gauss_module
/// 的 DenseRef 同数学；树中后代 id 恒小于祖先 id，方向块按 (min,max) 存无歧义）。
struct DenseRef {
    std::map<std::pair<int64_t, int64_t>, Dense> K, M, psi, mtemp;

    Dense& at(std::map<std::pair<int64_t, int64_t>, Dense>& t, int64_t i,
              int64_t j) {
        return t[{std::min(i, j), std::max(i, j)}];
    }
    const Dense& at(const std::map<std::pair<int64_t, int64_t>, Dense>& t,
                    int64_t i, int64_t j) const {
        return t.at({std::min(i, j), std::max(i, j)});
    }

    DenseRef(const Pencil<DefaultTraits>& pencil, const SeparatorTree& tree) {
        const int64_t N = tree.nNodes();
        for (int64_t i = 0; i < N; ++i) {
            at(K, i, i) = denseBlock(pencil.K, tree.node(i).indices,
                                     tree.node(i).indices);
            at(M, i, i) = denseBlock(pencil.M, tree.node(i).indices,
                                     tree.node(i).indices);
            for (int64_t a : tree.ancestors(i)) {
                at(K, i, a) = denseBlock(pencil.K, tree.node(i).indices,
                                         tree.node(a).indices);
                at(M, i, a) = denseBlock(pencil.M, tree.node(i).indices,
                                         tree.node(a).indices);
            }
        }
        for (int64_t i : tree.postOrder()) {
            const std::vector<int64_t> anc = tree.ancestors(i);
            for (int64_t j : anc)
                at(psi, i, j) = -at(K, i, i).llt().solve(at(K, i, j));
            for (size_t p = 0; p < anc.size(); ++p)
                for (size_t q = p; q < anc.size(); ++q) {
                    const int64_t j = anc[p], k = anc[q];
                    at(K, j, k) += at(psi, i, j).transpose() * at(K, i, k);
                    if (j != k) at(K, k, j) = at(K, j, k).transpose();
                }
            for (int64_t j : anc)
                at(mtemp, i, j) =
                    at(M, i, j) + at(M, i, i) * at(psi, i, j);
            for (size_t p = 0; p < anc.size(); ++p)
                for (size_t q = p; q < anc.size(); ++q) {
                    const int64_t j = anc[p], k = anc[q];
                    at(M, j, k) += at(psi, i, j).transpose() * at(M, i, k) +
                                   at(mtemp, i, j).transpose() * at(psi, i, k);
                    if (j != k) at(M, k, j) = at(M, j, k).transpose();
                }
        }
    }

    static Dense denseBlock(const SM& A, const std::vector<int64_t>& rows,
                            const std::vector<int64_t>& cols) {
        Dense out = Dense::Zero(static_cast<int64_t>(rows.size()),
                                static_cast<int64_t>(cols.size()));
        for (size_t p = 0; p < rows.size(); ++p)
            for (size_t q = 0; q < cols.size(); ++q)
                out(static_cast<int64_t>(p), static_cast<int64_t>(q)) =
                    A.coeff(rows[p], cols[q]);
        return out;
    }
};

/// μ 的稠密暴力递推：初始化 μ_{d,a} = Φ_dᵀ·M_temp_{d,a}（全部祖先对），
/// 再按 postOrder 逐中间节点传播 μ_{d,a} += μ_{d,i}·Ψ_{i,a}（d ∈ D_i）。
/// 注意不能拿 ref.M 的终值当参考——μ 传播只发生在 (d,a) 祖先方向块上。
std::map<std::pair<int64_t, int64_t>, Dense> denseMu(
    const SeparatorTree& tree, const DenseRef& ref,
    const std::vector<Dense>& phis) {
    std::map<std::pair<int64_t, int64_t>, Dense> mu;
    const int64_t N = tree.nNodes();
    for (int64_t d = 0; d < N; ++d)
        for (int64_t a : tree.ancestors(d))
            mu[{d, a}] =
                phis[static_cast<size_t>(d)].transpose() *
                ref.at(ref.mtemp, d, a);
    for (int64_t i : tree.postOrder())
        for (int64_t d : tree.descendants(i))
            for (int64_t a : tree.ancestors(i))
                mu.at({d, a}) +=
                    mu.at({d, i}) * ref.at(ref.psi, i, a);
    return mu;
}

/// 每节点随机稠密 Φ（rows = 节点自由度，cols = 指定模态数；非正交）。
std::vector<Dense> randomPhis(const SeparatorTree& tree,
                              const std::vector<int64_t>& modeCounts,
                              uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Dense> phis(static_cast<size_t>(tree.nNodes()));
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const int64_t ni = static_cast<int64_t>(tree.node(i).indices.size());
        const int64_t mi = modeCounts[static_cast<size_t>(i)];
        Dense phi(ni, mi);
        for (int64_t p = 0; p < ni; ++p)
            for (int64_t q = 0; q < mi; ++q) phi(p, q) = dist(rng);
        phis[static_cast<size_t>(i)] = phi;
    }
    return phis;
}

void expectNear(const Dense& out, const Dense& expect, double tol,
                const std::string& what) {
    REQUIRE(out.rows() == expect.rows());
    REQUIRE(out.cols() == expect.cols());
    if (out.size() == 0) return;
    const double diff = (out - expect).cwiseAbs().maxCoeff();
    const double scale = expect.cwiseAbs().maxCoeff();
    INFO(what << " maxdiff=" << diff);
    REQUIRE(diff <= tol * std::max(1.0, scale));
}

/// μ 逐块对拍（模块 Active 块 vs 稠密参考）。
void expectMuEqual(const TempBlockManager& tm,
                   const std::map<std::pair<int64_t, int64_t>, Dense>& muRef,
                   const SeparatorTree& tree, double tol) {
    for (int64_t d = 0; d < tree.nNodes(); ++d)
        for (int64_t a : tree.ancestors(d)) {
            const ColCompMat* blk = tm.getMu(d, a);
            REQUIRE(blk != nullptr);
            Dense out;
            blk->inflate(out);
            expectNear(out, muRef.at({d, a}), tol,
                       "mu (" + std::to_string(d) + "," + std::to_string(a) +
                           ")");
        }
}

/// 全链路搭建：pencil 完备后构造（消去四 stage 在构造期跑完），特征对由
/// 调用方 submit（随机 Φ 或全谱分解均可），再 runCondensation。
struct FullPipe {
    SeparatorTree tree;
    std::unique_ptr<OriginalBlocks> ob;
    std::unique_ptr<GaussEliminationModule> gm;
    EigenPairSaver saver;
    SubspaceMassSaver massSaver;
    std::unique_ptr<TempBlockManager> tm;
    std::unique_ptr<DenseRef> ref;

    FullPipe(const SeparatorTree& t, const Pencil<DefaultTraits>& pencil)
        : tree(t) {
        ob = std::make_unique<OriginalBlocks>(pencil, tree);
        gm = std::make_unique<GaussEliminationModule>();
        REQUIRE(gm->initialize(*ob));
        ref = std::make_unique<DenseRef>(pencil, tree);
        for (int64_t i : tree.postOrder()) {
            REQUIRE(gm->computeConstraint(i));
            REQUIRE(gm->accumulateReducedStiffness(i));
            REQUIRE(gm->computeMtemp(i));
            REQUIRE(gm->accumulateReducedMass(i));
        }
    }

    void submit(const std::vector<Dense>& phis,
                const std::vector<std::vector<double>>& lams) {
        REQUIRE(saver.initialize(tree.nNodes(), true));
        for (int64_t i = 0; i < tree.nNodes(); ++i)
            REQUIRE(saver.setEigenPair(
                i, std::vector<double>(lams[static_cast<size_t>(i)]),
                Dense(phis[static_cast<size_t>(i)])));
        REQUIRE(massSaver.initialize(tree));
        tm = std::make_unique<TempBlockManager>();
        REQUIRE(tm->initialize(tree, gm.get(), &saver, &massSaver));
    }

    /// 节点序串行缩聚；skipRight 用于就绪门控测试（该节点不发 rightMultiply）。
    void runCondensation(int64_t skipRight = -1) {
        for (int64_t i : tree.postOrder()) {
            REQUIRE(tm->create(i));
            REQUIRE(tm->update(i));
            if (i != skipRight) REQUIRE(tm->rightMultiply(i));
        }
    }
};

/// 子空间质量组装期望：非对角块 = μ_{d,a}·Φ_a，对角恒 1（PKU 约定）。
Dense expectedSubspaceMass(
    const SeparatorTree& tree,
    const std::map<std::pair<int64_t, int64_t>, Dense>& mu,
    const std::vector<Dense>& phis) {
    const int64_t N = tree.nNodes();
    std::vector<int64_t> off(static_cast<size_t>(N), 0);
    int64_t total = 0;
    for (int64_t i = 0; i < N; ++i) {
        off[static_cast<size_t>(i)] = total;
        total += phis[static_cast<size_t>(i)].cols();
    }
    Dense out = Dense::Zero(total, total);
    for (int64_t i = 0; i < total; ++i) out(i, i) = 1.0;
    for (const auto& [key, muDa] : mu) {
        const int64_t d = key.first, a = key.second;
        const Dense block = muDa * phis[static_cast<size_t>(a)];
        REQUIRE(block.rows() == phis[static_cast<size_t>(d)].cols());
        REQUIRE(block.cols() == phis[static_cast<size_t>(a)].cols());
        out.block(off[static_cast<size_t>(d)], off[static_cast<size_t>(a)],
                  block.rows(), block.cols()) = block;
        out.block(off[static_cast<size_t>(a)], off[static_cast<size_t>(d)],
                  block.cols(), block.rows()) = block.transpose();
    }
    return out;
}

} // namespace

TEST_CASE("TempBlockManager μ 创建与多跳传播对拍", "[temp_blocks]") {
    const double tol = 1e-9;
    {
        // 三层树：node1 为 0 模态叶，锻炼零块 create/update/rightMultiply 链路
        SeparatorTree tree = makeTree();
        const int64_t n = 8;
        Pencil<DefaultTraits> pencil;
        pencil.K = toSparse(buildStructured(tree, n, 1.0, 101));
        pencil.M = toSparse(buildStructured(tree, n, 0.5, 202));

        FullPipe pipe(tree, pencil);
        const std::vector<int64_t> modes{1, 0, 1, 1, 1};
        std::vector<Dense> phis = randomPhis(tree, modes, 77);
        std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()));
        for (int64_t i = 0; i < tree.nNodes(); ++i) {
            const int64_t m = modes[static_cast<size_t>(i)];
            for (int64_t q = 0; q < m; ++q)
                lams[static_cast<size_t>(i)].push_back(1.0 + 0.5 * q + 0.1 * i);
        }
        pipe.submit(phis, lams);
        pipe.runCondensation();

        expectMuEqual(*pipe.tm, denseMu(tree, *pipe.ref, phis), tree, tol);
    }
    {
        // 深链树：3 跳 μ 传播（预留列规划 + 多级 update 链），两个 0 模态叶
        SeparatorTree tree = makeChainTree();
        const int64_t n = 9;
        Pencil<DefaultTraits> pencil;
        pencil.K = toSparse(buildStructured(tree, n, 1.0, 303));
        pencil.M = toSparse(buildStructured(tree, n, 0.5, 404));

        FullPipe pipe(tree, pencil);
        const std::vector<int64_t> modes{1, 0, 1, 1, 1, 0, 1};
        std::vector<Dense> phis = randomPhis(tree, modes, 88);
        std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()));
        for (int64_t i = 0; i < tree.nNodes(); ++i) {
            const int64_t m = modes[static_cast<size_t>(i)];
            for (int64_t q = 0; q < m; ++q)
                lams[static_cast<size_t>(i)].push_back(2.0 + 0.5 * q + 0.1 * i);
        }
        pipe.submit(phis, lams);
        pipe.runCondensation();

        expectMuEqual(*pipe.tm, denseMu(tree, *pipe.ref, phis), tree, tol);
    }
}

TEST_CASE("TempBlockManager 右乘与子空间组装对拍", "[temp_blocks]") {
    SeparatorTree tree = makeTree();
    const int64_t n = 8;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(buildStructured(tree, n, 1.0, 505));
    pencil.M = toSparse(buildStructured(tree, n, 0.5, 606));

    FullPipe pipe(tree, pencil);
    const std::vector<int64_t> modes{1, 0, 1, 1, 1};
    std::vector<Dense> phis = randomPhis(tree, modes, 99);
    std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()));
    for (int64_t i = 0; i < tree.nNodes(); ++i) {
        const int64_t m = modes[static_cast<size_t>(i)];
        for (int64_t q = 0; q < m; ++q)
            lams[static_cast<size_t>(i)].push_back(1.5 + 0.7 * q + 0.2 * i);
    }
    pipe.submit(phis, lams);
    pipe.runCondensation();

    // 子空间质量：非对角 = μ·Φ，对角恒 1
    Dense ms;
    REQUIRE(pipe.massSaver.assembleSubspaceMass(pipe.saver, ms));
    expectNear(ms,
               expectedSubspaceMass(tree, denseMu(tree, *pipe.ref, phis), phis),
               1e-9, "subspace mass");

    // 子空间刚度：特征值按节点序铺全局对角线
    Dense ks;
    REQUIRE(pipe.saver.assembleSubspaceStiffness(ks));
    int64_t idx = 0;
    for (int64_t i = 0; i < tree.nNodes(); ++i)
        for (double lam : lams[static_cast<size_t>(i)]) {
            REQUIRE(ks(idx, idx) == Catch::Approx(lam).margin(1e-12));
            ++idx;
        }
    REQUIRE(idx == ks.rows());
}

TEST_CASE("TempBlockManager 全谱等价（完整模态）", "[temp_blocks]") {
    // 完整模态下子空间张成全空间：K̃_ij + K̃_ii·Ψ_ij ≡ 0 使约化刚度严格块
    // 对角 = Λ，缩聚质量与全局 M̃ 同余 —— 子空间谱须与全空间广义谱一致。
    auto runCase = [](const SeparatorTree& tree, int64_t n, uint32_t seed) {
        Pencil<DefaultTraits> pencil;
        pencil.K = toSparse(buildStructured(tree, n, 1.0, seed));
        pencil.M = toSparse(buildStructured(tree, n, 0.5, seed + 1));

        FullPipe pipe(tree, pencil);
        // 每节点对约化对角块做广义特征分解（全模态，Φ M-正交归一）
        std::vector<Dense> phis(static_cast<size_t>(tree.nNodes()));
        std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()));
        for (int64_t i = 0; i < tree.nNodes(); ++i) {
            const Dense* kd = pipe.gm->getReducedDiagK(i);
            const Dense* md = pipe.gm->getReducedDiagM(i);
            REQUIRE(kd != nullptr);
            REQUIRE(md != nullptr);
            Eigen::GeneralizedSelfAdjointEigenSolver<Dense> es(*kd, *md);
            REQUIRE(es.info() == Eigen::Success);
            const auto& vals = es.eigenvalues();
            lams[static_cast<size_t>(i)].assign(vals.data(),
                                                vals.data() + vals.size());
            phis[static_cast<size_t>(i)] = es.eigenvectors();
        }
        pipe.submit(phis, lams);
        pipe.runCondensation();

        Dense ks, ms;
        REQUIRE(pipe.saver.assembleSubspaceStiffness(ks));
        REQUIRE(pipe.massSaver.assembleSubspaceMass(pipe.saver, ms));
        Eigen::GeneralizedSelfAdjointEigenSolver<Dense> esSub(ks, ms);
        REQUIRE(esSub.info() == Eigen::Success);

        // 全空间广义特征问题
        Dense kf(pencil.K);
        Dense mf(pencil.M);
        Eigen::GeneralizedSelfAdjointEigenSolver<Dense> esFull(kf, mf);
        REQUIRE(esFull.info() == Eigen::Success);

        REQUIRE(esSub.eigenvalues().size() == esFull.eigenvalues().size());
        for (int64_t i = 0; i < esFull.eigenvalues().size(); ++i) {
            const double sub = esSub.eigenvalues()(i);
            const double full = esFull.eigenvalues()(i);
            INFO("eigenvalue " << i << ": sub=" << sub << " full=" << full);
            REQUIRE(std::abs(sub - full) <=
                    1e-8 * std::max(1.0, std::abs(full)));
        }
    };
    runCase(makeTree(), 8, 707);
    runCase(makeChainTree(), 9, 808);
}

TEST_CASE("TempBlockManager 就绪门控与 release 语义", "[temp_blocks]") {
    SeparatorTree tree = makeTree();
    const int64_t n = 8;
    Pencil<DefaultTraits> pencil;
    pencil.K = toSparse(buildStructured(tree, n, 1.0, 909));
    pencil.M = toSparse(buildStructured(tree, n, 0.5, 910));

    // —— 就绪门控 ——
    {
        FullPipe pipe(tree, pencil);
        const std::vector<int64_t> modes{1, 1, 1, 1, 1};
        std::vector<Dense> phis = randomPhis(tree, modes, 111);
        std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()),
                                              {1.0});
        pipe.submit(phis, lams);

        REQUIRE(pipe.tm->create(0));
        REQUIRE_FALSE(pipe.tm->create(0));  // 块已创建，重复 create 被拒
        // 左块 (1,2) 未创建：update(2) 全或无拒绝
        REQUIRE_FALSE(pipe.tm->update(2));
        REQUIRE(pipe.tm->create(1));
        REQUIRE(pipe.tm->update(2));
    }
    // —— release 拒绝与整树释放 ——
    {
        FullPipe pipe(tree, pencil);
        const std::vector<int64_t> modes{1, 1, 1, 1, 1};
        std::vector<Dense> phis = randomPhis(tree, modes, 112);
        std::vector<std::vector<double>> lams(static_cast<size_t>(tree.nNodes()),
                                              {1.0});
        pipe.submit(phis, lams);
        // node2 缺 rightMultiply：release(2) 拒绝
        pipe.runCondensation(/*skipRight*/ 2);
        REQUIRE_FALSE(pipe.tm->release(2));
        REQUIRE(pipe.tm->rightMultiply(2));
        REQUIRE(pipe.tm->release(2));
        // (·,2) 块已释放；其余块不受影响
        REQUIRE(pipe.tm->getMu(0, 2) == nullptr);
        REQUIRE(pipe.tm->getMu(1, 2) == nullptr);
        REQUIRE(pipe.tm->getMu(2, 4) != nullptr);
        REQUIRE(pipe.tm->getMu(0, 4) != nullptr);
        // 重复 release 幂等
        REQUIRE(pipe.tm->release(2));

        // 根整树释放
        REQUIRE(pipe.tm->release(4));
        for (int64_t d = 0; d < tree.nNodes(); ++d)
            for (int64_t a : tree.ancestors(d))
                REQUIRE(pipe.tm->getMu(d, a) == nullptr);
        REQUIRE(pipe.tm->release(4));  // 幂等
    }
}
