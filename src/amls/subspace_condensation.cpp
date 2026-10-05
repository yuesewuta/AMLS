// step 2b 实现：子空间缩聚（并行局部谱 + 后序投影装配）。
// 右/左投影全部经 OpenBLAS gemm；转置由 CblasTrans 标志隐式完成。
#include "amls/subspace_condensation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include <Eigen/Sparse>

#include "amls/thread_pool.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"

namespace amls {

SubspaceCondensation::Result
SubspaceCondensation::run(const ISpectralTruncation<DefaultTraits>& truncation,
                          int numThreads) {
    Result res;
    res.k.assign(static_cast<size_t>(N_), 0);
    res.mu.resize(static_cast<size_t>(N_));
    res.Z.resize(static_cast<size_t>(N_));
    res.projOffset.assign(static_cast<size_t>(N_), 0);
    stageTiming_.clear();

    const int nLevels = static_cast<int>(ge_.maxLevel()) + 1;
    beginPass();
    for (int l = 0; l < nLevels; ++l)
        spectralLevel(truncation, l, numThreads, res);
    for (int l = 0; l < nLevels; ++l)
        projectLevel(l, numThreads, res);
    finalize(res);
    return res;
}

void SubspaceCondensation::beginPass() {
    mbar_.clear();
    mbarRow_.clear();
    wAll_.assign(static_cast<size_t>(N_), Vector());
    vAll_.assign(static_cast<size_t>(N_), DenseMatrix());
}

void SubspaceCondensation::spectralLevel(
    const ISpectralTruncation<DefaultTraits>& truncation, int64_t l,
    int numThreads, Result& res) {
    const std::vector<int64_t>& nodes = ge_.levelNodes(l);
    const auto tI0 = std::chrono::steady_clock::now();
    // 每个节点相互独立：输入 (K̂_ii, M̂_ii) 在消去后固定，输出写入各自槽位。
    // 并发 LAPACK/SIL 调用把 OpenBLAS 线程数压到 1，避免 T×T 过度订阅。
    const int savedThreads = openblas_get_num_threads();
    blas::setNumThreads(1);
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            const int64_t i = nodes[static_cast<size_t>(t)];
            trace::Scope s("cond", "SPEC", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            const bool isLeaf = ge_.isLeaf(i);
            // 特征求解覆盖输入 → 取副本
            DenseMatrix Kh = isLeaf ? DenseMatrix(ge_.origKDiag(i))
                                    : ge_.kDiag(i).data;
            DenseMatrix Mh = isLeaf ? DenseMatrix(ge_.origMDiag(i))
                                    : ge_.mDiag(i).data;
            double rlo, rhi;
            DefaultTraits::Index il, iu;
            if (truncation.intervalHint(rlo, rhi))
                eig_.solveLocalInterval(Kh, Mh, rlo, rhi,
                                        wAll_[static_cast<size_t>(i)],
                                        vAll_[static_cast<size_t>(i)]);
            else if (truncation.indexHint(il, iu))
                eig_.solveLocalIndex(Kh, Mh, il, iu,
                                     wAll_[static_cast<size_t>(i)],
                                     vAll_[static_cast<size_t>(i)]);
            else
                eig_.solveLocalAll(Kh, Mh, wAll_[static_cast<size_t>(i)],
                                   vAll_[static_cast<size_t>(i)]);
        },
        numThreads);
    blas::setNumThreads(savedThreads);
    // 截断选择（本层各节点独立）
    for (int64_t i : nodes) {
        const Vector& w = wAll_[static_cast<size_t>(i)];
        const int64_t kmax = static_cast<int64_t>(w.size());
        const int64_t ki = truncation.select(w, kmax);
        res.k[static_cast<size_t>(i)] = ki;
        if (ki > 0) {
            res.mu[static_cast<size_t>(i)] = w.head(ki);
            res.Z[static_cast<size_t>(i)] =
                vAll_[static_cast<size_t>(i)].leftCols(ki);
        }
        wAll_[static_cast<size_t>(i)].resize(0);
        vAll_[static_cast<size_t>(i)].resize(0, 0);
    }
    stageTiming_["spectral"] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tI0)
            .count();
}

void SubspaceCondensation::projectLevel(int64_t l, int numThreads,
                                        Result& res) {
    const std::vector<int64_t>& nodes = ge_.levelNodes(l);
    const auto tII0 = std::chrono::steady_clock::now();
    // 层并行：同层节点的 mbar 块读写集合互不相交（块 (a,kk) 仅由 kk 与 a
    // 之间的中间节点及 a 本人触碰），层间 barrier 保证后代块已定稿；
    // 仅 unordered_map/vector 的结构性操作经 mapMutex 短锁保护。
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            projectNode_(nodes[static_cast<size_t>(t)], res);
        },
        numThreads);
    stageTiming_["projection"] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tII0)
            .count();
}

void SubspaceCondensation::finalize(Result& res) {
    int64_t acc = 0;
    for (int64_t i = 0; i < N_; ++i) {
        res.projOffset[static_cast<size_t>(i)] = acc;
        acc += res.k[static_cast<size_t>(i)];
    }
    res.p = acc;
    if (std::getenv("AMLS_DUMP_K")) {
        std::string line = "[dumpk]";
        for (int64_t i = 0; i < N_; ++i) {
            line += " i=" + std::to_string(i) + "(lv" +
                    std::to_string(ge_.level(i)) + ",n" +
                    std::to_string(ge_.size(i)) + ",k" +
                    std::to_string(res.k[static_cast<size_t>(i)]) + ")";
        }
        std::fprintf(stderr, "%s p=%lld\n", line.c_str(),
                     static_cast<long long>(res.p));
    }
}

void SubspaceCondensation::buildDag(
    DagRunner& dag, Result& res,
    const ISpectralTruncation<DefaultTraits>& truncation,
    const GaussianElimination::GeDagTasks& geInfo, bool releaseData) {
    struct NodeTasks {
        DagTask *spec = nullptr, *f = nullptr, *g = nullptr, *h = nullptr,
               *rel = nullptr;
    };
    std::vector<NodeTasks> nt(static_cast<size_t>(N_));
    const std::vector<int64_t>& order = ge_.postOrder();

    // W（Mtemp = K̂⁻¹M̂）释放移交：f(i) 的 WMz = W·Z_i 复用后，
    // 由本路径的 rel(i) 调 releaseNodeW（GE 的 DONE/relW 不再释放）。
    ge_.deferWRelease();

    // 优先级与 GE 同一节点主序：SPEC=rank*8+7，f=+8，g/h=+9，rel=+10。
    std::vector<int64_t> rank(static_cast<size_t>(N_));
    for (int64_t r = 0; r < static_cast<int64_t>(order.size()); ++r)
        rank[static_cast<size_t>(order[static_cast<size_t>(r)])] = r;

    // 静态后代表（依赖枚举）：g/h 需要"全部后代 j 的 f(j)/g(j) 完成"。
    // 行块 (i,kk) 由 f(kk) 创建、g(中间节点) 累加定稿——过近似覆盖全集，
    // 成本≈0：GE 内部边已把 geDone(kk) 排在 geDone(i) 前，COND 任务随挂随走。
    std::vector<std::vector<int64_t>> descs(static_cast<size_t>(N_));
    for (const int64_t i : order)
        for (int64_t a : ge_.tree().ancestors(i))
            descs[static_cast<size_t>(a)].push_back(i);

    for (const int64_t i : order) {
        const int64_t pr = rank[static_cast<size_t>(i)] * 8;
        auto& t = nt[static_cast<size_t>(i)];

        // ---- SPEC(i)：局部谱 + 截断选择（写 res.k/mu/Z）----
        t.spec = &dag.emplace([this, &truncation, &res, i] {
            trace::Scope s("cond", "SPEC", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            const bool isLeaf = ge_.isLeaf(i);
            // 特征求解覆盖输入 → 取副本
            DenseMatrix Kh = isLeaf ? DenseMatrix(ge_.origKDiag(i))
                                    : ge_.kDiag(i).data;
            DenseMatrix Mh = isLeaf ? DenseMatrix(ge_.origMDiag(i))
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
            if (std::getenv("AMLS_DUMP_K") && !isLeaf) {
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
            res.k[static_cast<size_t>(i)] = ki;
            if (ki > 0) {
                res.mu[static_cast<size_t>(i)] = w.head(ki);
                res.Z[static_cast<size_t>(i)] = v.leftCols(ki);
            }
        });
        t.spec->prio = pr + 7;
        dag.depend(*t.spec, *geInfo.done[static_cast<size_t>(i)]);

        // ---- f(i)：右投影初始化（式 16/17a；写 mbar(a,i) 行块）----
        t.f = &dag.emplace([this, &res, i] {
            if (res.k[static_cast<size_t>(i)] <= 0)
                return;
            trace::Scope s("cond", "PROJ.init", i);
            if (trace::enabled()) {
                s.e.aux = res.k[static_cast<size_t>(i)];
                s.e.aux2 = ge_.level(i);
            }
            const bool isLeaf = ge_.isLeaf(i);
            const auto& Zi = res.Z[static_cast<size_t>(i)];
            // Mtemp 复用：K̂⁻¹(M̂·Z_i) = W·Z_i（结合律，同 GE 的 T3）。
            // 统一 DAG 路径 W 存活 → 一次 gemm 代替"M̂·Z_i gemm + 回代"；
            // 旧路径 ElimData 已销毁（nodeW 返回 nullptr）→ 回退原路径。
            DenseMatrix WMz;
            if (const DenseMatrix* W = ge_.nodeW(i)) {
                ge_.ops().gemm(CblasNoTrans, CblasNoTrans, 1.0, *W, Zi, 0.0,
                               WMz);
            } else {
                DenseMatrix Mz = isLeaf
                                     ? DenseMatrix(ge_.origMDiag(i)) * Zi
                                     : ge_.mDiag(i).data * Zi;
                ge_.solveFactor(i, Mz, WMz);
            }
            for (int64_t a : ge_.tree().ancestors(i)) {
                DenseMatrix proj;
                if (isLeaf)
                    proj = DenseMatrix(ge_.origMCoupling(a, i)) * Zi;
                else
                    ge_.ops().gemm(CblasNoTrans, CblasNoTrans, 1.0,
                                   ge_.mCoupling(a, i), Zi, 0.0, proj);
                if (isLeaf)
                    proj.noalias() -=
                        DenseMatrix(ge_.origKCoupling(a, i)) * WMz;
                else
                    ge_.ops().subMul(proj, ge_.kCoupling(a, i), WMz);
                std::lock_guard<std::mutex> lk(mapMutex_);
                mbar_.emplace(key_(a, i), std::move(proj));
                mbarRow_[a].push_back(i);
            }
        });
        t.f->prio = pr + 8;
        dag.depend(*t.f, *t.spec);

        // ---- g(i)：中间节点列更新（式 17c；读行 i 块、写祖先行块）----
        t.g = &dag.emplace([this, i] {
            trace::Scope s("cond", "PROJ.mid", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            const std::vector<int64_t> anc = ge_.tree().ancestors(i);
            if (anc.empty())
                return;   // 根：同余更新无接收方
            std::vector<int64_t> rowKK;
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                const auto it = mbarRow_.find(i);
                if (it != mbarRow_.end())
                    rowKK = it->second;
            }
            if (rowKK.empty())
                return;
            if (trace::enabled())
                s.e.aux = static_cast<int64_t>(rowKK.size());
            const bool isLeaf = ge_.isLeaf(i);
            const int64_t ni = ge_.size(i);
            int64_t total = 0;
            for (int64_t kk : rowKK) {
                std::lock_guard<std::mutex> lk(mapMutex_);
                total += mbar_.at(key_(i, kk)).cols();
            }
            DenseMatrix RHS(ni, total);
            int64_t off = 0;
            for (int64_t kk : rowKK) {
                const DenseMatrix* b;
                {
                    std::lock_guard<std::mutex> lk(mapMutex_);
                    b = &mbar_.at(key_(i, kk));
                }
                RHS.block(0, off, ni, b->cols()) = *b;
                off += b->cols();
            }
            DenseMatrix Vs;
            ge_.solveFactor(i, RHS, Vs);
            // 块 (a,kk)：本任务独占（g 链串行化同块写入），查表加锁、算术解锁
            off = 0;
            for (int64_t kk : rowKK) {
                int64_t kcols;
                {
                    std::lock_guard<std::mutex> lk(mapMutex_);
                    kcols = mbar_.at(key_(i, kk)).cols();
                }
                for (int64_t a : anc) {
                    DenseMatrix* tgt;
                    {
                        std::lock_guard<std::mutex> lk(mapMutex_);
                        tgt = &mbar_.at(key_(a, kk));
                    }
                    if (isLeaf)
                        ge_.ops().subMul(*tgt, ge_.origKCoupling(a, i),
                                         Vs.block(0, off, ni, kcols));
                    else
                        ge_.ops().subMul(*tgt, ge_.kCoupling(a, i),
                                         Vs.block(0, off, ni, kcols));
                }
                off += kcols;
            }
        });
        t.g->prio = pr + 9;
        dag.depend(*t.g, *geInfo.done[static_cast<size_t>(i)]);
        for (int64_t j : descs[static_cast<size_t>(i)]) {
            dag.depend(*t.g, *nt[static_cast<size_t>(j)].g);
            dag.depend(*t.g, *nt[static_cast<size_t>(j)].f);
        }

        // ---- h(i)：左投影（式 17b；M̃_{i,kk} = Z_iᵀ·行块）----
        t.h = &dag.emplace([this, &res, i] {
            const int64_t ki = res.k[static_cast<size_t>(i)];
            if (ki <= 0)
                return;
            trace::Scope s("cond", "PROJ.left", i);
            if (trace::enabled())
                s.e.aux2 = ge_.level(i);
            std::vector<int64_t> rowKK;
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                const auto it = mbarRow_.find(i);
                if (it != mbarRow_.end())
                    rowKK = it->second;
            }
            if (trace::enabled())
                s.e.aux = static_cast<int64_t>(rowKK.size());
            for (int64_t kk : rowKK) {
                DenseMatrix* blk;
                {
                    std::lock_guard<std::mutex> lk(mapMutex_);
                    blk = &mbar_.at(key_(i, kk));
                }
                DenseMatrix out;
                blas::gemm0(CblasTrans, CblasNoTrans,
                            res.Z[static_cast<size_t>(i)], *blk, out);
                std::lock_guard<std::mutex> lk(outMutex_);
                res.mprojBlocks.push_back({i, kk, std::move(out)});
            }
        });
        t.h->prio = pr + 9;
        dag.depend(*t.h, *t.spec);
        for (int64_t j : descs[static_cast<size_t>(i)]) {
            dag.depend(*t.h, *nt[static_cast<size_t>(j)].g);
            dag.depend(*t.h, *nt[static_cast<size_t>(j)].f);
        }

        // ---- rel(i)：行块释放 + GE 数据缩容（f/g/h 之后）----
        t.rel = &dag.emplace([this, &res, i, releaseData] {
            trace::Scope s("cond", "REL", i);
            std::vector<int64_t> rowKK;
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                const auto it = mbarRow_.find(i);
                if (it != mbarRow_.end())
                    rowKK = it->second;
            }
            // mbar 投影中间块：恢复向量不需要，无条件释放。
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                for (int64_t kk : rowKK)
                    mbar_.erase(key_(i, kk));
                mbarRow_.erase(i);
            }
            // Mtemp（W = K̂⁻¹M̂）：缩聚 f 已用完（rel ← f），释放。
            ge_.releaseNodeW(i);
            // GE 块（diag/耦合）与 Z：恢复向量时的 backtransform 仍需要
            // （旧路径 if(!keepFactors) releaseNode 同理），只在
            // releaseData（= !keepFactors）时缩容。
            if (releaseData)
                ge_.shrinkNodeData(i, true);
            if (releaseData)
                res.Z[static_cast<size_t>(i)].resize(0, 0);
        });
        t.rel->prio = pr + 10;
        dag.depend(*t.rel, *t.f);
        dag.depend(*t.rel, *t.g);
        dag.depend(*t.rel, *t.h);
    }
}

void SubspaceCondensation::projectNode_(int64_t i, Result& res) {
    trace::Scope sAll("cond", "PROJ", i);
    if (trace::enabled())
        sAll.e.aux2 = ge_.level(i);
    const bool isLeaf = ge_.isLeaf(i);
    const std::vector<int64_t> anc = ge_.tree().ancestors(i);
    const int64_t ki = res.k[static_cast<size_t>(i)];

    // 本节点的后代列（ mb row i ），在层 barrier 后已定稿
    std::vector<int64_t> rowKK;
    {
        std::lock_guard<std::mutex> lk(mapMutex_);
        const auto it = mbarRow_.find(i);
        if (it != mbarRow_.end())
            rowKK = it->second;
    }

    if (!anc.empty()) {
        // 中间节点列更新：V_k = K̂_ii⁻¹ M̄_{i,kk}；全部后代列合并右端一次回代
        if (!rowKK.empty()) {
            trace::Scope sMid("cond", "PROJ.mid", i);
            if (trace::enabled()) {
                sMid.e.aux = static_cast<int64_t>(rowKK.size());
                sMid.e.aux2 = ge_.level(i);
            }
            const int64_t ni = ge_.size(i);
            int64_t total = 0;
            for (int64_t kk : rowKK) {
                std::lock_guard<std::mutex> lk(mapMutex_);
                total += mbar_.at(key_(i, kk)).cols();
            }
            DenseMatrix RHS(ni, total);
            int64_t off = 0;
            for (int64_t kk : rowKK) {
                const DenseMatrix* b;
                {
                    std::lock_guard<std::mutex> lk(mapMutex_);
                    b = &mbar_.at(key_(i, kk));
                }
                RHS.block(0, off, ni, b->cols()) = *b;
                off += b->cols();
            }
            DenseMatrix Vs;
            ge_.solveFactor(i, RHS, Vs);
            // 块 (a,kk) 为本层内本节点独占：查表加锁取地址（node-based
            // 容器引用稳定），解锁后做算术。
            off = 0;
            for (int64_t kk : rowKK) {
                trace::Scope sKK("cond", "PROJ.mid.kk", i);
                int64_t kcols;
                {
                    std::lock_guard<std::mutex> lk(mapMutex_);
                    kcols = mbar_.at(key_(i, kk)).cols();
                }
                if (trace::enabled()) {
                    sKK.e.aux = kk;
                    sKK.e.aux2 = kcols;
                }
                for (int64_t a : anc) {
                    DenseMatrix* tgt;
                    {
                        std::lock_guard<std::mutex> lk(mapMutex_);
                        tgt = &mbar_.at(key_(a, kk));
                    }
                    if (isLeaf)
                        ge_.ops().subMul(*tgt, ge_.origKCoupling(a, i),
                                         Vs.block(0, off, ni, kcols));
                    else
                        ge_.ops().subMul(*tgt, ge_.kCoupling(a, i),
                                         Vs.block(0, off, ni, kcols));
                }
                off += kcols;
            }
        }

        // 右投影初始化（结合律重排：先乘 Z_i 再回代，见 run 头注）
        if (ki > 0) {
            trace::Scope sInit("cond", "PROJ.init", i);
            if (trace::enabled()) {
                sInit.e.aux = ki;
                sInit.e.aux2 = ge_.level(i);
            }
            const auto& Zi = res.Z[static_cast<size_t>(i)];
            // 同 DAG 路径：Mtemp 复用（旧路径 nodeW 恒 nullptr → 回退）
            DenseMatrix WMz;
            if (const DenseMatrix* W = ge_.nodeW(i)) {
                ge_.ops().gemm(CblasNoTrans, CblasNoTrans, 1.0, *W, Zi, 0.0,
                               WMz);
            } else {
                DenseMatrix Mz = isLeaf
                                     ? DenseMatrix(ge_.origMDiag(i)) * Zi
                                     : ge_.mDiag(i).data * Zi;
                ge_.solveFactor(i, Mz, WMz);
            }
            for (int64_t a : anc) {
                DenseMatrix proj;
                if (isLeaf)
                    proj = DenseMatrix(ge_.origMCoupling(a, i)) * Zi;
                else
                    ge_.ops().gemm(CblasNoTrans, CblasNoTrans, 1.0,
                                   ge_.mCoupling(a, i), Zi, 0.0, proj);
                if (isLeaf)
                    proj.noalias() -=
                        DenseMatrix(ge_.origKCoupling(a, i)) * WMz;
                else
                    ge_.ops().subMul(proj, ge_.kCoupling(a, i), WMz);
                std::lock_guard<std::mutex> lk(mapMutex_);
                mbar_.emplace(key_(a, i), std::move(proj));
                mbarRow_[a].push_back(i);
            }
        }
    }

    // 左投影：M̃_{i,kk} = Z_iᵀ·M̄_{i,kk}；随后释放本行块
    if (ki > 0) {
        trace::Scope sLeft("cond", "PROJ.left", i);
        if (trace::enabled()) {
            sLeft.e.aux = static_cast<int64_t>(rowKK.size());
            sLeft.e.aux2 = ge_.level(i);
        }
        for (int64_t kk : rowKK) {
            DenseMatrix* blk;
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                blk = &mbar_.at(key_(i, kk));
            }
            DenseMatrix out;
            blas::gemm0(CblasTrans, CblasNoTrans,
                        res.Z[static_cast<size_t>(i)], *blk, out);
            std::lock_guard<std::mutex> lk(outMutex_);
            res.mprojBlocks.push_back({i, kk, std::move(out)});
        }
    }
    {
        std::lock_guard<std::mutex> lk(mapMutex_);
        for (int64_t kk : rowKK)
            mbar_.erase(key_(i, kk));
        mbarRow_.erase(i);
    }
}

SubspaceCondensation::DenseMatrix
SubspaceCondensation::backtransform(const Result& r,
                                    const DenseMatrix& Q) const {    const int64_t m = Q.cols();
    int64_t n = 0;
    for (int64_t i = 0; i < N_; ++i)
        n += ge_.size(i);

    std::vector<DenseMatrix> z(static_cast<size_t>(N_));
    const std::vector<int64_t>& order = ge_.postOrder();

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int64_t i     = *it;
        const int64_t ni    = ge_.size(i);
        const bool isLeaf   = ge_.isLeaf(i);
        const int64_t ki    = r.k[static_cast<size_t>(i)];

        DenseMatrix yi;
        if (ki > 0)
            yi = r.Z[static_cast<size_t>(i)] *
                 Q.block(r.projOffset[static_cast<size_t>(i)], 0, ki, m);
        else
            yi = DenseMatrix::Zero(ni, m);

        DenseMatrix rhs = DenseMatrix::Zero(ni, m);
        for (int64_t a : ge_.tree().ancestors(i)) {
            if (isLeaf) {
                rhs.noalias() +=
                    DenseMatrix(ge_.origKCoupling(a, i)).transpose().eval() *
                    z[static_cast<size_t>(a)];
            } else {
                // K_{i,a}·z_a = (a,i) 块ᵀ·z_a：存储行空间积 + 行散布
                const ColComp& B = ge_.kCoupling(a, i);
                if (B.empty()) continue;
                DenseMatrix tmp(B.storedCols(), m);
                ge_.ops().gemm(CblasTrans, CblasNoTrans, 1.0, B,
                               z[static_cast<size_t>(a)], 0.0, tmp);
                if (B.ident) {
                    rhs.noalias() += tmp;
                } else {
                    for (int64_t j = 0; j < static_cast<int64_t>(B.cols.size());
                         ++j)
                        rhs.row(B.cols[static_cast<size_t>(j)]) +=
                            tmp.row(j);
                }
            }
        }
        DenseMatrix dz;
        ge_.solveFactor(i, rhs, dz);
        z[static_cast<size_t>(i)] = yi - dz;
    }

    DenseMatrix out(n, m);
    for (int64_t i = 0; i < N_; ++i)
        out.block(ge_.offset(i), 0, ge_.size(i), m) =
            z[static_cast<size_t>(i)];
    return out;
}

} // namespace amls
