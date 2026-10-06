// step 2b 实现：子空间缩聚（N7 局部谱 → N8 μ̃ 种子 → N9 折叠 / N10 传播）。
// 全部块乘走 blockops 的 accumulateProduct / accumulateTransposedProduct
//（OpenBLAS dgemm 后端），转置经 CblasTrans 标志隐式完成。
#include "amls/subspace_condensation.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "amls/thread_pool.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"

namespace amls {

SubspaceCondensation::Result
SubspaceCondensation::run(const ISpectralTruncation<DefaultTraits>& truncation,
                          int numThreads) {
    Result res;
    res.keptModes.assign(static_cast<size_t>(N_), 0);
    res.eigenvalues.resize(static_cast<size_t>(N_));
    res.eigenvectors.resize(static_cast<size_t>(N_));
    res.reducedOffset.assign(static_cast<size_t>(N_), 0);
    stageTiming_.clear();

    const int64_t nLevels = ge_.maxLevel() + 1;
    beginPass();
    for (int64_t l = 0; l < nLevels; ++l)
        spectralLevel(truncation, l, numThreads, res);
    for (int64_t l = 0; l < nLevels; ++l)
        projectLevel(l, numThreads, res);
    finalize(res);
    return res;
}

void SubspaceCondensation::beginPass() {
    store_.prepare(
        ge_.tree(),
        [this](int64_t a, int64_t i) -> const ColMask& {
            return ge_.symbolicRowK(a, i);
        },
        [this](int64_t a, int64_t i) -> const ColMask& {
            return ge_.symbolicRowM(a, i);
        },
        [this](int64_t node) { return ge_.size(node); });
}

// ---------------------------------------------------------------------------
// N7：局部谱 + 截断选择
// ---------------------------------------------------------------------------
void SubspaceCondensation::solveLocalEigenproblem(
    int64_t i, const ISpectralTruncation<DefaultTraits>& truncation,
    Result& res) {
    trace::Scope s("cond", "N7", i);
    if (trace::enabled())
        s.e.aux2 = ge_.level(i);
    const bool leaf = ge_.isLeaf(i);
    // 特征求解覆盖输入 → 取副本
    DenseMatrix Kh = leaf ? DenseMatrix(ge_.origKDiag(i))
                          : ge_.kDiag(i).data;
    DenseMatrix Mh = leaf ? DenseMatrix(ge_.origMDiag(i))
                          : ge_.mDiag(i).data;
    Vector w;
    DenseMatrix v;
    double rlo, rhi;
    DefaultTraits::Index il, iu;
    if (truncation.intervalHint(rlo, rhi))
        eig_.solveLocalInterval(Kh, Mh, rlo, rhi, w, v);
    else if (truncation.indexHint(il, iu))
        eig_.solveLocalIndex(Kh, Mh, il, iu, w, v);
    else
        eig_.solveLocalAll(Kh, Mh, w, v);
    // 截断选择（本节点独立）
    const int64_t kmax = static_cast<int64_t>(w.size());
    const int64_t ki = truncation.select(w, kmax);
    if (std::getenv("AMLS_DUMP_K") && !leaf) {
        std::fprintf(stderr,
                     "[dumpspec] i=%lld lv=%lld n=%lld Kh=%zux%zu "
                     "Mh=%zux%zu kmax=%lld ki=%lld Kh00=%g Mh00=%g\n",
                     static_cast<long long>(i),
                     static_cast<long long>(ge_.level(i)),
                     static_cast<long long>(ge_.size(i)),
                     Kh.rows(), Kh.cols(), Mh.rows(), Mh.cols(),
                     static_cast<long long>(kmax),
                     static_cast<long long>(ki),
                     Kh.rows() > 0 && Kh.cols() > 0 ? Kh(0, 0) : -1.0,
                     Mh.rows() > 0 && Mh.cols() > 0 ? Mh(0, 0) : -1.0);
    }
    res.keptModes[static_cast<size_t>(i)] = ki;
    if (ki > 0) {
        res.eigenvalues[static_cast<size_t>(i)] = w.head(ki);
        res.eigenvectors[static_cast<size_t>(i)] = v.leftCols(ki);
    }
}

void SubspaceCondensation::spectralLevel(
    const ISpectralTruncation<DefaultTraits>& truncation, int64_t l,
    int numThreads, Result& res) {
    const std::vector<int64_t>& nodes = ge_.levelNodes(l);
    const auto t0 = std::chrono::steady_clock::now();
    // 每个节点相互独立：输入 (K̂_ii, M̂_ii) 在消去后固定，输出写入各自槽位。
    // 并发 LAPACK 调用把 OpenBLAS 线程数压到 1，避免 T×T 过度订阅。
    const int savedThreads = openblas_get_num_threads();
    blas::setNumThreads(1);
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            solveLocalEigenproblem(nodes[static_cast<size_t>(t)], truncation,
                                   res);
        },
        numThreads);
    blas::setNumThreads(savedThreads);
    stageTiming_["spectral"] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
}

// ---------------------------------------------------------------------------
// N8：μ̃ 种子
// ---------------------------------------------------------------------------
void SubspaceCondensation::seedMassCoupling(int64_t i, const Result& res) {
    if (res.keptModes[static_cast<size_t>(i)] <= 0)
        return;
    for (const int64_t a : ge_.tree().ancestors(i))
        store_.seed(i, a, res.eigenvectors[static_cast<size_t>(i)],
                    ge_.temporaryMass(i, a));
}

// ---------------------------------------------------------------------------
// N9 折叠 + N10 传播（列 i 定稿 → 沿路径上传 → 释放 μ̃ 列 i）
// ---------------------------------------------------------------------------
void SubspaceCondensation::foldAndPropagate(int64_t i, Result& res) {
    const std::vector<int64_t>& cols = store_.columnEntries(i);
    if (!cols.empty()) {
        // N9：m(c,i) = μ̃(c,i)·eigenvectors_i（k_c×k_i），转置为降阶块
        //（k_i×k_c，rowNode=祖先 i，colNode=后代 c）
        const DenseMatrix& eigenvectorsI =
            res.eigenvectors[static_cast<size_t>(i)];
        for (const int64_t c : cols) {
            const DenseMatrix folded = store_.fold(c, i, eigenvectorsI);
            if (folded.rows() <= 0 || folded.cols() <= 0)
                continue;
            std::lock_guard<std::mutex> lk(outMutex_);
            res.reducedMassBlocks.push_back(
                {i, c, DenseMatrix(folded.transpose())});
        }
        // N10：μ̃(c,a) += μ̃(c,i)·u(i,a)（keptModes_i=0 也照常——链不可断）
        for (const int64_t a : ge_.tree().ancestors(i)) {
            const ColComp& u = ge_.constraintModes(i, a);
            if (u.empty())
                continue;
            for (const int64_t c : cols)
                store_.propagate(c, i, a, u);
        }
    }
    store_.releaseColumn(i);
}

void SubspaceCondensation::projectLevel(int64_t l, int numThreads,
                                        Result& res) {
    const std::vector<int64_t>& nodes = ge_.levelNodes(l);
    const auto t0 = std::chrono::steady_clock::now();
    // 层并行：同层节点的 μ̃ 槽位读写集合互不相交（(c,·) 仅由 c 的子树路径
    // 上的节点触碰），层间 barrier 保证后代列已定稿。
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            const int64_t i = nodes[static_cast<size_t>(t)];
            trace::Scope s("cond", "N8N9N10", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            seedMassCoupling(i, res);
            foldAndPropagate(i, res);
        },
        numThreads);
    stageTiming_["projection"] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
}

void SubspaceCondensation::finalize(Result& res) {
    int64_t acc = 0;
    for (int64_t i = 0; i < N_; ++i) {
        res.reducedOffset[static_cast<size_t>(i)] = acc;
        acc += res.keptModes[static_cast<size_t>(i)];
    }
    res.reducedSize = acc;
    if (std::getenv("AMLS_DUMP_K")) {
        std::string line = "[dumpk]";
        for (int64_t i = 0; i < N_; ++i) {
            line += " i=" + std::to_string(i) + "(lv" +
                    std::to_string(ge_.level(i)) + ",n" +
                    std::to_string(ge_.size(i)) + ",k" +
                    std::to_string(res.keptModes[static_cast<size_t>(i)]) +
                    ")";
        }
        std::fprintf(stderr, "%s p=%lld\n", line.c_str(),
                     static_cast<long long>(res.reducedSize));
    }
}

// ---------------------------------------------------------------------------
// 统一 DAG：N7–N10 与 GE 任务同一 runner（优先级共享 rank*16 节点主序：
// GE 占 +0..+6，缩聚占 +7..+12）
// ---------------------------------------------------------------------------
void SubspaceCondensation::buildDag(
    DagRunner& dag, Result& res,
    const ISpectralTruncation<DefaultTraits>& truncation,
    const GaussianElimination::EliminationDagTasks& geInfo,
    bool releaseData) {
    struct NodeTasks {
        DagTask* n7 = nullptr;
        DagTask* n8 = nullptr;
        DagTask* fold = nullptr;
    };
    std::vector<NodeTasks> nt(static_cast<size_t>(N_));
    const std::vector<int64_t>& order = ge_.postOrder();
    std::vector<int64_t> rank(static_cast<size_t>(N_));
    for (int64_t r = 0; r < static_cast<int64_t>(order.size()); ++r)
        rank[static_cast<size_t>(order[static_cast<size_t>(r)])] = r;

    for (const int64_t i : order) {
        const int64_t pr = rank[static_cast<size_t>(i)] * 16;
        auto& t = nt[static_cast<size_t>(i)];

        // ---- N7(i)：局部谱 + 截断选择 ← N4/N5 汇合哨兵（叶经哨兵 ← N1）
        t.n7 = &dag.emplace([this, &truncation, &res, i] {
            solveLocalEigenproblem(i, truncation, res);
        });
        t.n7->prio = pr + 7;
        dag.depend(*t.n7, *geInfo.updatesDone[static_cast<size_t>(i)]);

        // ---- N8(i)：μ̃ 种子 ← N7(i), N3(i)（Mtemp 就绪）
        t.n8 = &dag.emplace([this, &res, i] {
            trace::Scope s("cond", "N8", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            seedMassCoupling(i, res);
        });
        t.n8->prio = pr + 8;
        dag.depend(*t.n8, *t.n7);
        dag.depend(*t.n8, *geInfo.n3[static_cast<size_t>(i)]);

        // ---- fold(i) = N9+N10 ← N1(i)（u 就绪）, N7(i), N8(i)，
        //      fold(孩子)（μ̃(c,i) 沿路径的传播链定稿），N8(全部后代)（种子）。
        t.fold = &dag.emplace([this, &res, i, releaseData] {
            trace::Scope s("cond", "N9N10", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            foldAndPropagate(i, res);
            // 恢复向量不需要时，eigenvectors 在折叠后同步释放
            if (releaseData)
                res.eigenvectors[static_cast<size_t>(i)].resize(0, 0);
        });
        t.fold->prio = pr + 9;
        dag.depend(*t.fold, *geInfo.n1[static_cast<size_t>(i)]);
        dag.depend(*t.fold, *t.n7);
        dag.depend(*t.fold, *t.n8);
        for (const int64_t c : ge_.tree().node(i).children)
            dag.depend(*t.fold, *nt[static_cast<size_t>(c)].fold);
        for (const int64_t d : ge_.tree().descendants(i))
            dag.depend(*t.fold, *nt[static_cast<size_t>(d)].n8);

        // ---- freeMtemp(i) ← N5 汇合哨兵, N8(i)（Mtemp 两个消费者都完成）
        DagTask* freeMtemp = &dag.emplace([this, i] {
            trace::Scope s("cond", "freeMtemp", i);
            ge_.releaseTemporaryMass(i);
        });
        freeMtemp->prio = pr + 10;
        dag.depend(*freeMtemp, *geInfo.updatesDone[static_cast<size_t>(i)]);
        dag.depend(*freeMtemp, *t.n8);

        // ---- freeDiag(i) ← N1(i)（分解消费）, N7(i)（局部谱消费）
        DagTask* freeDiag = &dag.emplace([this, i] {
            trace::Scope s("cond", "freeDiag", i);
            ge_.releaseDiagonalBlocks(i);
        });
        freeDiag->prio = pr + 11;
        dag.depend(*freeDiag, *geInfo.n1[static_cast<size_t>(i)]);
        dag.depend(*freeDiag, *t.n7);

        // ---- freeU(i) ← fold(i)（u 的最后消费者；恢复向量时不建）
        if (releaseData) {
            DagTask* freeU = &dag.emplace([this, i] {
                trace::Scope s("cond", "freeU", i);
                ge_.releaseConstraintModes(i);
            });
            freeU->prio = pr + 12;
            dag.depend(*freeU, *t.fold);
        }
    }
}

// ---------------------------------------------------------------------------
// step 4：纯 GEMM 回代 z_i = eigenvectors_i·q_i + Σ_a u(i,a)·z_a
//（u = Ψ = −K̂⁻¹K 已含负号；无需因子与耦合块）
// ---------------------------------------------------------------------------
SubspaceCondensation::DenseMatrix
SubspaceCondensation::backtransform(const Result& r,
                                    const DenseMatrix& Q) const {
    const int64_t m = Q.cols();
    int64_t n = 0;
    for (int64_t i = 0; i < N_; ++i)
        n += ge_.size(i);

    std::vector<DenseMatrix> z(static_cast<size_t>(N_));
    const std::vector<int64_t>& order = ge_.postOrder();

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t i  = *it;
        const int64_t ni = ge_.size(i);
        const int64_t ki = r.keptModes[static_cast<size_t>(i)];

        DenseMatrix zi = DenseMatrix::Zero(ni, m);
        if (ki > 0)
            zi.noalias() +=
                r.eigenvectors[static_cast<size_t>(i)] *
                Q.block(r.reducedOffset[static_cast<size_t>(i)], 0, ki, m);
        // Σ_a u(i,a)·z_a：u 列压缩（cols = a 空间非零列，驱动 z_a 行 gather）
        for (const int64_t a : ge_.tree().ancestors(i))
            blockops::accumulateProduct(zi, ge_.constraintModes(i, a),
                                        z[static_cast<size_t>(a)]);
        z[static_cast<size_t>(i)] = std::move(zi);
    }

    DenseMatrix out(n, m);
    for (int64_t i = 0; i < N_; ++i)
        out.block(ge_.offset(i), 0, ge_.size(i), m) =
            z[static_cast<size_t>(i)];
    return out;
}

} // namespace amls
