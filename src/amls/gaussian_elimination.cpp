// step 2a 实现：高斯消去（块分解 → 祖先方向回代求解 → Schur / 同余更新）。
// 数值推导同 Gao et al. TOMS 2008 §2–§3；矩阵乘一律走 OpenBLAS gemm，
// 转置经 CblasTrans 标志隐式完成（不物化转置副本）。
#include "amls/gaussian_elimination.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <Eigen/Sparse>

#include "amls/thread_pool.hpp"
#include "core/dag.hpp"
#include "core/logging.hpp"
#include "core/trace.hpp"
#include "linalg/blas.hpp"

namespace amls {

namespace {

/// 稠密块显著非零元占比（|v| > 1e-10·max|v|）。
inline double significantFrac(
    const Eigen::Ref<const DefaultTraits::DenseMatrix>& A) {
    if (A.size() == 0)
        return 0.0;
    double mx = 0.0;
    for (int64_t j = 0; j < A.size(); ++j)
        mx = std::max(mx, std::abs(A.data()[j]));
    if (mx == 0.0)
        return 0.0;
    int64_t c = 0;
    for (int64_t j = 0; j < A.size(); ++j)
        if (std::abs(A.data()[j]) > 1e-10 * mx)
            ++c;
    return static_cast<double>(c) / static_cast<double>(A.size());
}

/// 稠密块显著非零行数（|v| > 1e-10·max|v|）。
inline int64_t significantRows(
    const Eigen::Ref<const DefaultTraits::DenseMatrix>& A) {
    if (A.size() == 0)
        return 0;
    double mx = 0.0;
    for (int64_t j = 0; j < A.size(); ++j)
        mx = std::max(mx, std::abs(A.data()[j]));
    if (mx == 0.0)
        return 0;
    int64_t c = 0;
    for (int64_t r = 0; r < A.rows(); ++r)
        for (int64_t j = 0; j < A.cols(); ++j)
            if (std::abs(A(r, j)) > 1e-10 * mx) { ++c; break; }
    return c;
}

/// 稀疏块的非零列索引（升序）。
inline std::vector<int32_t> nonzeroColIdx(
    const DefaultTraits::SparseMatrix& A) {
    std::vector<int32_t> idx;
    for (int64_t j = 0; j < A.cols(); ++j)
        if (A.col(j).nonZeros() > 0)
            idx.push_back(static_cast<int32_t>(j));
    return idx;
}

/// 稠密块的精确非零列索引（升序）。右端零列经可逆回代仍为零列，
/// 只需解非零列（X_a 非零列 = K_{i,a} 非零列的数学保证）。
inline std::vector<int32_t> nonzeroColIdx(
    const Eigen::Ref<const DefaultTraits::DenseMatrix>& A) {
    std::vector<int32_t> idx;
    for (int64_t j = 0; j < A.cols(); ++j)
        for (int64_t r = 0; r < A.rows(); ++r)
            if (A(r, j) != 0.0) {
                idx.push_back(static_cast<int32_t>(j));
                break;
            }
    return idx;
}

/// 列 gather：B(:,c) = A(:,idx[c])。
inline DefaultTraits::DenseMatrix gatherCols(
    const DefaultTraits::DenseMatrix& A, const std::vector<int32_t>& idx) {
    DefaultTraits::DenseMatrix B(A.rows(), static_cast<int64_t>(idx.size()));
    for (int64_t c = 0; c < static_cast<int64_t>(idx.size()); ++c)
        B.col(c) = A.col(idx[static_cast<size_t>(c)]);
    return B;
}

/// 列散布：full(:,idx[c]) = S(:,c)（full 预先置零）。
inline void scatterCols(DefaultTraits::DenseMatrix& full,
                        const DefaultTraits::DenseMatrix& S,
                        const std::vector<int32_t>& idx) {
    for (int64_t c = 0; c < static_cast<int64_t>(idx.size()); ++c)
        full.col(idx[static_cast<size_t>(c)]) = S.col(c);
}

/// 稀疏块的非零行位图。
inline ColMask rowMaskOf(const DefaultTraits::SparseMatrix& A) {
    ColMask m;
    m.init(A.rows());
    for (int64_t c = 0; c < A.cols(); ++c)
        for (typename DefaultTraits::SparseMatrix::InnerIterator it(A,
                 static_cast<DefaultTraits::Index>(c));
             it; ++it)
            m.set(it.row());
    return m;
}

/// 稀疏块的非零列位图。
inline ColMask colMaskOf(const DefaultTraits::SparseMatrix& A) {
    ColMask m;
    m.init(A.cols());
    for (int64_t c = 0; c < A.cols(); ++c)
        if (A.col(c).nonZeros() > 0) m.set(c);
    return m;
}

} // namespace

GaussianElimination::GaussianElimination(
    const Pencil<DefaultTraits>& reordered, const SeparatorTree& tree)
    : pencil_(reordered), tree_(tree) {
    N_ = tree_.nNodes();
    sparseFactors_.resize(static_cast<size_t>(N_));
    denseFactors_.resize(static_cast<size_t>(N_));
    precompute_();
    // 树层级分组（后序保证孩子先于父亲）
    std::vector<int64_t> level(static_cast<size_t>(N_), 0);
    for (const int64_t i : order_) {
        const auto& nd = tree_.node(i);
        if (!nd.children.empty()) {
            int64_t l = 0;
            for (const int64_t c : nd.children)
                l = std::max(l, level[static_cast<size_t>(c)] + 1);
            level[static_cast<size_t>(i)] = l;
            maxLevel_ = std::max(maxLevel_, l);
        }
    }
    byLevel_.resize(static_cast<size_t>(maxLevel_) + 1);
    for (const int64_t i : order_)
        byLevel_[static_cast<size_t>(level[static_cast<size_t>(i)])].push_back(i);
    level_ = std::move(level);

    // 动态块表容量预算：运行期（并行消去）绝不允许 rehash——
    // 工作线程持有指向表元素的引用做累加，rehash 会使其悬垂。
    // 键集上界来自符号结构：每节点读 (a,i) 列块 |anc| 个 + 写祖先对块
    // C(|anc|,2) 个；diag 表键 = 节点 id。留 2x 余量。
    int64_t coupKeys = 16;
    for (const int64_t i : order_) {
        const int64_t a = static_cast<int64_t>(tree_.ancestors(i).size());
        coupKeys += a + a * (a - 1) / 2;
    }
    kCoupD_.reserve(static_cast<size_t>(coupKeys));
    mCoupD_.reserve(static_cast<size_t>(coupKeys));
    kDiag_.reserve(static_cast<size_t>(N_) + 1);
    mDiag_.reserve(static_cast<size_t>(N_) + 1);
}

DefaultTraits::SparseMatrix GaussianElimination::sparseBlock_(
    const SparseMatrix& A, int64_t r0, int64_t r1, int64_t c0,
    int64_t c1) const {
    const int64_t m = r1 - r0, n = c1 - c0;
    std::vector<Eigen::Triplet<double, Index>> trips;
    trips.reserve(static_cast<size_t>(m));
    for (Index c = c0; c < c1; ++c) {
        for (typename SparseMatrix::InnerIterator it(A, c); it; ++it) {
            const Index r = it.row();
            if (r >= r0 && r < r1)
                trips.emplace_back(r - r0, c - c0, it.value());
        }
    }
    SparseMatrix B(m, n);
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}

void GaussianElimination::precompute_() {
    off_.assign(static_cast<size_t>(N_), 0);
    size_.assign(static_cast<size_t>(N_), 0);

    order_ = tree_.postOrder();
    int64_t acc = 0;
    for (const int64_t id : order_) {
        off_[static_cast<size_t>(id)]  = acc;
        size_[static_cast<size_t>(id)] =
            static_cast<int64_t>(tree_.node(id).indices.size());
        acc += size_[static_cast<size_t>(id)];
    }

    // 对角块
    for (int64_t id = 0; id < N_; ++id) {
        const int64_t o = off_[static_cast<size_t>(id)];
        const int64_t n = size_[static_cast<size_t>(id)];
        origKDiag_[id] = sparseBlock_(pencil_.K, o, o + n, o, o + n);
        origMDiag_[id] = sparseBlock_(pencil_.M, o, o + n, o, o + n);
    }
    // 祖先-后代耦合原块（所有祖先对都建条目：零非零但维度正确，
    // 保证“无直接边”的块返回正确尺寸的零矩阵）
    for (int64_t id = 0; id < N_; ++id) {
        const int64_t oi = off_[static_cast<size_t>(id)];
        const int64_t ni = size_[static_cast<size_t>(id)];
        for (int64_t a : tree_.ancestors(id)) {
            const int64_t oa = off_[static_cast<size_t>(a)];
            const int64_t na = size_[static_cast<size_t>(a)];
            const int64_t k  = key_(a, id);
            origKCoup_[k] = sparseBlock_(pencil_.K, oa, oa + na, oi, oi + ni);
            origMCoup_[k] = sparseBlock_(pencil_.M, oa, oa + na, oi, oi + ni);
        }
    }

    // 符号填充 pass：精确预算全部耦合块的终态列模式（表出生即终态）
    symbolicFill_();
}

void GaussianElimination::symbolicFill_() {
    // rowK/rowM：块 (a,i) 消去完成时刻的非零行模式（postOrder 中处理
    // i 时即终态）；colK/colM：块 (a,i) 终态非零列模式 = 出生表。
    std::unordered_map<int64_t, ColMask> rowK, rowM, colK, colM;
    for (int64_t id = 0; id < N_; ++id)
        for (int64_t a : tree_.ancestors(id)) {
            const int64_t k = key_(a, id);
            rowK.emplace(k, rowMaskOf(origK_(a, id)));
            colK.emplace(k, colMaskOf(origK_(a, id)));
            rowM.emplace(k, rowMaskOf(origM_(a, id)));
            colM.emplace(k, colMaskOf(origM_(a, id)));
        }
    for (const int64_t i : order_) {
        const std::vector<int64_t> anc = tree_.ancestors(i);
        for (size_t u = 1; u < anc.size(); ++u) {      // hi：更高祖先
            const int64_t hi = anc[u];
            const ColMask& rHi  = rowK.at(key_(hi, i));
            const ColMask& rMhi = rowM.at(key_(hi, i));
            if (rHi.count() == 0) continue;   // K_{hi,i}≡0，无积
            for (size_t v = 0; v < u; ++v) {  // lo：更低祖先
                const int64_t lo = anc[v];
                const int64_t k = key_(hi, lo);
                const ColMask& rKlo = rowK.at(key_(lo, i));
                if (rKlo.count() != 0) {
                    // K̄_{hi,lo} −= K_{hi,i}·X：列来自 X 的非零列
                    // = K_{i,lo} 的非零列 = K_{lo,i} 的非零行
                    colK[k].merge(rKlo);
                    rowK[k].merge(rHi);   // 积的行 ⊆ K_{hi,i} 的行
                    // M̄ 的 T2=(K_{lo,i}Y_hi)ᵀ 与 T3=K_{hi,i}·W·X_lo 的列
                    // 同为 K 行驱动（X/W 不扩展零列结构）
                    colM[k].merge(rKlo);
                    rowM[k].merge(rHi);
                    if (rMhi.count() != 0)
                        rowM[k].merge(rMhi);  // T2 的 hi 空间行由 M 行驱动
                }
                const ColMask& rMlo = rowM.at(key_(lo, i));
                if (rMlo.count() != 0) {
                    // T1 = K_{hi,i}·Y_lo 的列由 M 行驱动
                    colM[k].merge(rMlo);
                    rowM[k].merge(rHi);
                }
            }
        }
    }
    // 仅保留将成为出生表的列模式（行模式已完成使命）
    for (auto& [k, m] : colK)
        if (m.count() != 0) kColFinal_[k] = std::move(m);
    for (auto& [k, m] : colM)
        if (m.count() != 0) mColFinal_[k] = std::move(m);
}

const GaussianElimination::SparseMatrix&
GaussianElimination::origKCoupling(int64_t a, int64_t lo) const {
    return origK_(a, lo);
}
const GaussianElimination::SparseMatrix&
GaussianElimination::origMCoupling(int64_t a, int64_t lo) const {
    return origM_(a, lo);
}

const GaussianElimination::SparseMatrix& GaussianElimination::origK_(
    int64_t a, int64_t lo) const {
    const int64_t k = key_(a, lo);
    static const SparseMatrix empty;
    const auto it = origKCoup_.find(k);
    return it == origKCoup_.end() ? empty : it->second;
}

const GaussianElimination::SparseMatrix& GaussianElimination::origM_(
    int64_t a, int64_t lo) const {
    const int64_t k = key_(a, lo);
    static const SparseMatrix empty;
    const auto it = origMCoup_.find(k);
    return it == origMCoup_.end() ? empty : it->second;
}

const GaussianElimination::ColComp& GaussianElimination::kCoup_(
    int64_t a, int64_t lo) const {
    const int64_t k = key_(a, lo);
    const auto it = kCoupD_.find(k);
    if (it != kCoupD_.end()) return it->second;
    // 诞生空表块（从未收过更新 ⇒ 旧语义的 Zero(n_a×n_lo) 块）
    std::lock_guard<std::mutex> lk(structMutex_);
    const auto it2 = kCoupD_.find(k);
    if (it2 != kCoupD_.end()) return it2->second;
    ColComp blk;
    blk.full = size_[static_cast<size_t>(lo)];
    blk.ident = false;
    blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(a)], 0);
    auto res = kCoupD_.emplace(k, std::move(blk));
    return res.first->second;
}

const GaussianElimination::ColComp& GaussianElimination::mCoup_(
    int64_t a, int64_t lo) const {
    const int64_t k = key_(a, lo);
    const auto it = mCoupD_.find(k);
    if (it != mCoupD_.end()) return it->second;
    std::lock_guard<std::mutex> lk(structMutex_);
    const auto it2 = mCoupD_.find(k);
    if (it2 != mCoupD_.end()) return it2->second;
    ColComp blk;
    blk.full = size_[static_cast<size_t>(lo)];
    blk.ident = false;
    blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(a)], 0);
    auto res = mCoupD_.emplace(k, std::move(blk));
    return res.first->second;
}

/// 诞生时把原稀疏块的非零列散布进 ColComp（表已含这些列——符号 pass
/// 初始化已并入原块列模式；lower_bound 找不到属防御路径，跳过并计数）。
static void injectOrig_(ColComp& blk, const DefaultTraits::SparseMatrix& orig,
                        std::atomic<int64_t>& nMiss) {
    for (int64_t j = 0; j < orig.cols(); ++j) {
        if (orig.col(j).nonZeros() == 0) continue;
        if (blk.ident) {
            blk.data.col(j) = DefaultTraits::DenseMatrix(orig.col(j));
            continue;
        }
        auto it = std::lower_bound(blk.cols.begin(), blk.cols.end(),
                                   static_cast<int32_t>(j));
        if (it == blk.cols.end() || *it != static_cast<int32_t>(j)) {
            ++nMiss;   // 符号 pass 漏列（不应发生）
            continue;
        }
        blk.data.col(it - blk.cols.begin()) =
            DefaultTraits::DenseMatrix(orig.col(j));
    }
}

ColComp& GaussianElimination::kCoupTarget_(int64_t hi, int64_t lo) {
    const int64_t k = key_(hi, lo);
    // 热路径：表已 reserve（构造时按符号结构预算），无锁查找返回引用；
    // 引用在运行期稳定（无 rehash、无同键 erase——erase 只在趟间
    // releaseNode/resetBlocks，彼时无并发消去）。仅创建走 structMutex_。
    if (auto it = kCoupD_.find(k); it != kCoupD_.end())
        return it->second;
    std::lock_guard<std::mutex> lk(structMutex_);
    if (auto it = kCoupD_.find(k); it != kCoupD_.end())
        return it->second;
    // 诞生：表 = 符号填充 pass 的终态列模式（出生即终态）
    ColComp blk;
    blk.full = size_[static_cast<size_t>(lo)];
    auto fi = kColFinal_.find(k);
    if (fi == kColFinal_.end() || fi->second.count() == 0) {
        blk.ident = false;
        blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(hi)], 0);
    } else {
        blk.ident = fi->second.count() == blk.full;
        blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(hi)],
                                     fi->second.count());
        if (!blk.ident) blk.cols = fi->second.indices();
        // 注意：不能 erase(fi)——趟 A/B（twoPass）会重跑整个流水线，
        // 表需保留供二次物化（GE 可重入，releaseNode 已清空全部块）。
        injectOrig_(blk, origK_(hi, lo), nMiss_);
    }
    auto res = kCoupD_.emplace(k, std::move(blk));
    return res.first->second;
}

ColComp& GaussianElimination::mCoupTarget_(int64_t hi, int64_t lo) {
    const int64_t k = key_(hi, lo);
    if (auto it = mCoupD_.find(k); it != mCoupD_.end())
        return it->second;
    std::lock_guard<std::mutex> lk(structMutex_);
    if (auto it = mCoupD_.find(k); it != mCoupD_.end())
        return it->second;
    ColComp blk;
    blk.full = size_[static_cast<size_t>(lo)];
    auto fi = mColFinal_.find(k);
    if (fi == mColFinal_.end() || fi->second.count() == 0) {
        blk.ident = false;
        blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(hi)], 0);
    } else {
        blk.ident = fi->second.count() == blk.full;
        blk.data = DenseMatrix::Zero(size_[static_cast<size_t>(hi)],
                                     fi->second.count());
        if (!blk.ident) blk.cols = fi->second.indices();
        // 表保留供 twoPass 重跑（同 kCoupTarget_）
        injectOrig_(blk, origM_(hi, lo), nMiss_);
    }
    auto res = mCoupD_.emplace(k, std::move(blk));
    return res.first->second;
}

ColComp& GaussianElimination::kDiagTarget_(int64_t a) {
    if (auto it = kDiag_.find(a); it != kDiag_.end())
        return it->second;
    std::lock_guard<std::mutex> lk(structMutex_);
    if (auto it = kDiag_.find(a); it != kDiag_.end())
        return it->second;
    ColComp blk;
    blk.full = size_[static_cast<size_t>(a)];
    blk.ident = true;
    blk.data = DenseMatrix(origKDiag_.at(a));
    auto res = kDiag_.emplace(a, std::move(blk));
    return res.first->second;
}

ColComp& GaussianElimination::mDiagTarget_(int64_t a) {
    if (auto it = mDiag_.find(a); it != mDiag_.end())
        return it->second;
    std::lock_guard<std::mutex> lk(structMutex_);
    if (auto it = mDiag_.find(a); it != mDiag_.end())
        return it->second;
    ColComp blk;
    blk.full = size_[static_cast<size_t>(a)];
    blk.ident = true;
    blk.data = DenseMatrix(origMDiag_.at(a));
    auto res = mDiag_.emplace(a, std::move(blk));
    return res.first->second;
}

bool GaussianElimination::isLeaf(int64_t i) const {
    return tree_.node(i).role == NodeRole::Leaf;
}

std::mutex& GaussianElimination::keyLock_(int64_t key) {
    {
        std::shared_lock<std::shared_mutex> lk(keyLocksMutex_);
        const auto it = keyLocks_.find(key);
        if (it != keyLocks_.end())
            return *it->second;
    }
    std::lock_guard<std::shared_mutex> lk(keyLocksMutex_);
    return *keyLocks_.emplace(key, std::make_unique<std::mutex>())
                .first->second;
}

void GaussianElimination::solveFactor(int64_t i, const DenseMatrix& B,
                                      DenseMatrix& X) const {
    if (sparseFactors_[static_cast<size_t>(i)])
        sparseFactors_[static_cast<size_t>(i)]->solveWithFactor(B, X);
    else
        denseFactors_[static_cast<size_t>(i)]->solveWithFactor(B, X);
}

void GaussianElimination::factorNode_(int64_t i, ElimData& d) {
    using Clock = std::chrono::steady_clock;
    const bool leaf = isLeaf(i);
    const char* who = leaf ? ".leaf" : ".sep";
    const auto t0 = Clock::now();

    // ---- 1. K̂_ii 因式分解 ----
    if (leaf) {
        sparseFactors_[static_cast<size_t>(i)] =
            std::make_unique<EigenSimplicialLDLTSolver<DefaultTraits>>();
        sparseFactors_[static_cast<size_t>(i)]->factorize(origKDiag_.at(i));
    } else {
        denseFactors_[static_cast<size_t>(i)] =
            std::make_unique<DenseLDLTSolver<DefaultTraits>>();
        if (static const char* dump = std::getenv("AMLS_DUMP_KDIAG");
            dump != nullptr) {
            const std::string f =
                std::string("kdiag_") + dump + "_" +
                std::to_string(i) + ".bin";
            std::ofstream os(f, std::ios::binary);
            const DenseMatrix& kd = kDiagTarget_(i).data;
            const int64_t r = kd.rows();
            os.write(reinterpret_cast<const char*>(&r), sizeof(r));
            os.write(reinterpret_cast<const char*>(kd.data()),
                     static_cast<std::streamsize>(r * r) *
                         static_cast<std::streamsize>(sizeof(double)));
        }
        try {
            denseFactors_[static_cast<size_t>(i)]->factorizeDense(
                kDiagTarget_(i).data);
        } catch (const std::exception& e) {
            AMLS_LOG_ERROR("[ge] 节点 {} (n={}, leaf={}) LDL 失败: {}", i,
                           size_[static_cast<size_t>(i)], leaf, e.what());
            throw;
        }
    }
    const auto t1 = Clock::now();
    {
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["factor"] += dt;
        timing_[std::string("factor") + who] += dt;
    }

    if (tree_.ancestors(i).empty())
        return;   // 根节点：消去即完成（同余更新无接收方，W 亦不需要）

    // M̂_ii 缓存（T3 结合律的右乘因子；D/E 任务及 makeWX_ 读）。
    // 必须在全部后代完成后执行——DAG 中由节点就绪条件（孩子 DONE）保证。
    d.W = leaf ? DenseMatrix(origMDiag_.at(i)) : mDiagTarget_(i).data;
}

void GaussianElimination::solveXYAnc_(int64_t i, ElimData& d, int64_t a,
                                      bool leaf) {
    using Clock = std::chrono::steady_clock;
    const int64_t na = size_[static_cast<size_t>(a)];
    auto tp0 = Clock::now();
    auto& prof = phaseProf();
    // 右端零列跳过：X_a（Y_a）的非零列 = K_{i,a}（M_{i,a}）的非零列
    // （K̂⁻¹ 可逆保零列）。叶右端直接由稀疏块非零行构造窄右端（免
    // expandT 全物化）；分割节点右端 expandT 后 scan 非零列再 gather。
    auto buildSub = [&](const SparseMatrix& A, bool isK) -> bool {
        // A = 块 (a,i)（na×m）：右端 K_{i,a} = Aᵀ（m×na）的非零列
        // = A 的非零行。直接构造窄右端 Bsub(:,p) = Aᵀ(:,idx[p])。
        ColMask mk;
        mk.init(na);
        for (int64_t c = 0; c < A.cols(); ++c)
            for (typename SparseMatrix::InnerIterator it(
                     A, static_cast<Index>(c));
                 it; ++it)
                mk.set(it.row());
        auto idx = mk.indices();
        auto& colsMap = isK ? d.xCols : d.yCols;
        auto& out = isK ? d.X[a] : d.Y[a];
        colsMap[a] = idx;
        out = DenseMatrix::Zero(size_[static_cast<size_t>(i)], na);
        if (idx.empty()) return true;   // 全零右端 → 全零解
        DenseMatrix Bsub =
            DenseMatrix::Zero(size_[static_cast<size_t>(i)],
                              static_cast<int64_t>(idx.size()));
        std::vector<int64_t> pos(static_cast<size_t>(na), -1);
        for (int64_t p = 0; p < static_cast<int64_t>(idx.size()); ++p)
            pos[static_cast<size_t>(idx[static_cast<size_t>(p)])] = p;
        for (int64_t c = 0; c < A.cols(); ++c)
            for (typename SparseMatrix::InnerIterator it(
                     A, static_cast<Index>(c));
                 it; ++it)
                Bsub(c, pos[static_cast<size_t>(it.row())]) = it.value();
        DenseMatrix Xsub;
        solveFactor(i, Bsub, Xsub);
        scatterCols(out, Xsub, idx);
        return true;
    };
    auto buildSubDense = [&](const ColComp& Cc, bool isK) {
        // Cc = (a,i) 列压缩块：右端 = expandT（m×na）后 scan 非零列。
        const DenseMatrix R = ops().expandT(Cc);
        auto idx = nonzeroColIdx(R);
        auto& colsMap = isK ? d.xCols : d.yCols;
        auto& out = isK ? d.X[a] : d.Y[a];
        colsMap[a] = idx;
        out = DenseMatrix::Zero(R.rows(), na);
        if (idx.empty()) return;
        DenseMatrix Xsub;
        solveFactor(i, gatherCols(R, idx), Xsub);
        scatterCols(out, Xsub, idx);
    };
    if (leaf) {
        buildSub(origK_(a, i), true);
        buildSub(origM_(a, i), false);
    } else {
        buildSubDense(kCoup_(a, i), true);
        buildSubDense(mCoup_(a, i), false);
    }
    profAdd(prof.solveRhs,
            std::chrono::duration<double>(Clock::now() - tp0).count());

    // ---- 块统计：耦合块列稀疏性 + 解的实际密度 ----
    if (stats) {
        GeBlockStat bs;
        bs.node = i;
        bs.leaf = leaf;
        bs.level = level_[static_cast<size_t>(i)];
        bs.anc  = a;
        bs.rows = na;
        bs.cols = size_[static_cast<size_t>(i)];
        if (leaf) {
            bs.nonzeroCols = static_cast<int64_t>(nonzeroColIdx(origK_(a, i)).size());
            bs.kFrac = significantFrac(DenseMatrix(origK_(a, i)));
        } else {
            const ColComp& kb = kCoup_(a, i);
            bs.nonzeroCols = kb.storedCols();
            bs.kFrac = significantFrac(kb.data);
        }
        bs.xNZRows = significantRows(d.X.at(a));
        bs.yNZRows = significantRows(d.Y.at(a));
        bs.xFrac = significantFrac(d.X.at(a));
        bs.yFrac = significantFrac(d.Y.at(a));
        std::lock_guard<std::mutex> lk(statsMutex_);
        stats->blocks.push_back(bs);
    }
}

void GaussianElimination::recordNodeStat_(int64_t i) {
    GeNodeStat ns;
    ns.node  = i;
    ns.n     = size_[static_cast<size_t>(i)];
    ns.depth = static_cast<int64_t>(tree_.ancestors(i).size());
    ns.leaf  = isLeaf(i);
    if (ns.leaf) {
        const SparseMatrix& Kd = origKDiag_.at(i);
        ns.kDiagFrac =
            Kd.nonZeros() > 0
                ? static_cast<double>(Kd.nonZeros()) /
                      (static_cast<double>(ns.n) * ns.n)
                : 0.0;
    } else {
        ns.kDiagFrac = significantFrac(kDiagTarget_(i).data);
    }
    std::lock_guard<std::mutex> lk(statsMutex_);
    stats->nodes.push_back(std::move(ns));
}

void GaussianElimination::solveXY_(int64_t i, ElimData& d) {
    using Clock = std::chrono::steady_clock;
    const bool leaf = isLeaf(i);
    const char* who = leaf ? ".leaf" : ".sep";
    const std::vector<int64_t> anc = tree_.ancestors(i);
    if (anc.empty())
        return;
    const auto t1 = Clock::now();

    // ---- 2+3. W/X/Y 回代 ----
    // W = K̂_ii⁻¹ M̂_ii（P5/P11），X_a = K̂_ii⁻¹ K_{i,a}，Y_a = K̂_ii⁻¹ M_{i,a}
    // （K_{i,a} = K_{a,i}ᵀ，经算子 expandT 物化：回代右端需实体矩阵）。
    // 注：合并宽右端单次求解实测更慢（Eigen 宽稠密右端三角回代缓存不友好），
    // 保持分祖先求解（DAG 路径下同——每祖先一个 B 任务）。
    d.X.reserve(anc.size() * 2);
    d.Y.reserve(anc.size() * 2);
    for (int64_t a : anc)
        solveXYAnc_(i, d, a, leaf);
    if (stats)
        recordNodeStat_(i);
    {
        const double dt = std::chrono::duration<double>(Clock::now() - t1).count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["solve"] += dt;
        timing_[std::string("solve") + who] += dt;
    }
}

void GaussianElimination::computeNodeA_(int64_t i, ElimData& d) {
    factorNode_(i, d);
    solveXY_(i, d);
}

const GaussianElimination::DenseMatrix&
GaussianElimination::makeWX_(int64_t i, ElimData& d, int64_t a) const {
    // T3 结合律：K_{a,i}·(K̂⁻¹M̂)·X_a = K_{a,i}·K̂⁻¹·(M̂·X_a)——
    // wx 只对 X_a 的非零列先乘 M̂ 再回代（不物化 W，避免 m×m 全回代）。
    // 结果按祖先缓存：C 任务（DAG）算一次，E 的 diag/coup 多处复用。
    auto it = d.WX.find(a);
    if (it != d.WX.end() && it->second.has_value())
        return *it->second;
    const DenseMatrix& Xa = d.X.at(a);
    DenseMatrix wx = DenseMatrix::Zero(Xa.rows(), Xa.cols());
    const auto itc = d.xCols.find(a);
    if (itc != d.xCols.end() && !itc->second.empty()) {
        DenseMatrix MXg, Wg;
        ops().gemm(CblasNoTrans, CblasNoTrans, 1.0, d.W,
                   gatherCols(Xa, itc->second), 0.0, MXg);
        solveFactor(i, MXg, Wg);
        scatterCols(wx, Wg, itc->second);
    }
    if (it != d.WX.end()) {
        it->second = std::move(wx);
        return *it->second;
    }
    return *d.WX.emplace(a, std::move(wx)).first->second;
}

void GaussianElimination::schurKDiagTask_(int64_t i, ElimData& d, int64_t a,
                                          bool leaf) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    // 尾部大任务（观测 3.2s 级）在就绪队列空闸下临时放开 BLAS 多线程。
    const double gf =
        2.0 * size_[static_cast<size_t>(a)] * size_[static_cast<size_t>(i)] *
        d.X.at(a).cols() / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(a));
    ColComp& tgt = kDiagTarget_(a);
    if (leaf)
        ops().subMul(tgt.data, origK_(a, i), d.X.at(a));
    else
        ops().subMul(tgt, kCoup_(a, i), d.X.at(a));
    profAdd(prof.schurKdiag,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
}

void GaussianElimination::schurKCoupTask_(int64_t i, ElimData& d, int64_t hi,
                                          int64_t lo, bool leaf) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    const double gf =
        2.0 * size_[static_cast<size_t>(hi)] * size_[static_cast<size_t>(i)] *
        d.X.at(lo).cols() / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(key_(hi, lo)));
    ColComp& tgt = kCoupTarget_(hi, lo);
    if (leaf)
        ops().subMul(tgt, origK_(hi, i), d.X.at(lo));
    else
        ops().subMul(tgt, kCoup_(hi, i), d.X.at(lo));
    profAdd(prof.schurKcoup,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
}

void GaussianElimination::congDiagTask_(int64_t i, ElimData& d, int64_t a,
                                        bool leaf) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    double kw = 0;
    const auto itw = d.WX.find(a);
    if (itw != d.WX.end() && itw->second)
        kw = static_cast<double>(itw->second->cols());
    const double gf =
        2.0 * size_[static_cast<size_t>(a)] * size_[static_cast<size_t>(i)] *
        (2.0 * d.Y.at(a).cols() + kw) / 1e9;
    auto mb = blas::maybeBudget(gf);
    const DenseMatrix& wx = makeWX_(i, d, a);
    profAdd(prof.congDiagWX,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
    const auto tp1 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(keyLock_(a));
    ColComp& md = mDiagTarget_(a);
    if (leaf) {
        ops().subMul(md, origK_(a, i), d.Y.at(a));
        ops().subTransMul(md, origK_(a, i), d.Y.at(a));
        ops().addMul(md, origK_(a, i), wx);
    } else {
        const ColComp& B = kCoup_(a, i);
        ops().subMul(md, B, d.Y.at(a));
        ops().subTransMul(md, B, d.Y.at(a));
        ops().addMul(md, B, wx);
    }
    profAdd(prof.congDiagT,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp1).count());
}

void GaussianElimination::congCoupTask_(int64_t i, ElimData& d, int64_t hi,
                                        int64_t lo, bool leaf) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    double kw = 0;
    const auto itw = d.WX.find(lo);
    if (itw != d.WX.end() && itw->second)
        kw = static_cast<double>(itw->second->cols());
    const double gf =
        2.0 * size_[static_cast<size_t>(i)] *
        (size_[static_cast<size_t>(hi)] * (d.Y.at(lo).cols() + kw) +
         size_[static_cast<size_t>(lo)] * d.Y.at(hi).cols()) /
        1e9;
    auto mb = blas::maybeBudget(gf);
    const DenseMatrix& wx = makeWX_(i, d, lo);
    std::lock_guard<std::mutex> lk(keyLock_(key_(hi, lo)));
    ColComp& tgt = mCoupTarget_(hi, lo);
    if (leaf) {
        ops().subMul(tgt, origK_(hi, i), d.Y.at(lo));
        ops().subTransMul(tgt, origK_(lo, i), d.Y.at(hi));
        ops().addMul(tgt, origK_(hi, i), wx);
    } else {
        const ColComp& Bhi = kCoup_(hi, i);
        ops().subMul(tgt, Bhi, d.Y.at(lo));
        ops().subTransMul(tgt, kCoup_(lo, i), d.Y.at(hi));
        ops().addMul(tgt, Bhi, wx);
    }
    profAdd(prof.congCoup,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
}

void GaussianElimination::applyNodeB_(int64_t i, ElimData& d) {
    using Clock = std::chrono::steady_clock;
    const bool leaf = isLeaf(i);
    const char* who = leaf ? ".leaf" : ".sep";
    const std::vector<int64_t> anc = tree_.ancestors(i);
    if (anc.empty())
        return;
    const Clock::time_point t2 = Clock::now();
    Clock::time_point t3;

    // ---- 4. K 单边 Schur 更新（P6；列压缩：求和维只走乘子存储列，
    //      右端行 gather 由乘子列映射驱动——算子内部完成）----
    for (int64_t a : anc)
        schurKDiagTask_(i, d, a, leaf);
    for (size_t p = 0; p < anc.size(); ++p)
        for (size_t q = p + 1; q < anc.size(); ++q)
            schurKCoupTask_(i, d, anc[q], anc[p], leaf);
    t3 = Clock::now();
    {
        const double dt = std::chrono::duration<double>(t3 - t2).count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["schurK"] += dt;
        timing_[std::string("schurK") + who] += dt;
    }

    // ---- 5. M 双边同余更新（式 2 的 M⁽ⁱ⁾ 递归）----
    // 对角：M̄_{a,a} −= K_{a,i}Y_a + (K_{a,i}Y_a)ᵀ − K_{a,i}·W·X_a
    // 三项全部直写 md（beta=1 累加），无 t/t2 中间量、无转置拷贝。
    for (int64_t a : anc)
        congDiagTask_(i, d, a, leaf);
    // 耦合：M̄_{hi,lo} −= K_{hi,i}Y_lo + (K_{lo,i}Y_hi)ᵀ − K_{hi,i}·W·X_lo
    // 三项均经算子限制在接收块存储列上；W·X_lo 与 hi 无关，按 lo 提公因子
    //（makeWX_ 缓存保证每个 lo 只算一次）。
    for (size_t p = 0; p < anc.size(); ++p)
        for (size_t q = p + 1; q < anc.size(); ++q)
            congCoupTask_(i, d, anc[q], anc[p], leaf);
    {
        const double dt = std::chrono::duration<double>(Clock::now() - t3)
                              .count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["congrM"] += dt;
        timing_[std::string("congrM") + who] += dt;
    }
}

void GaussianElimination::eliminateNode_(int64_t i) {
    ElimData d;
    computeNodeA_(i, d);
    applyNodeB_(i, d);
}

void GaussianElimination::run(int numThreads) {
    if (numThreads <= 1) {
        for (const int64_t i : order_)
            eliminateNode_(i);
        return;
    }
    // 层并行：level=0（叶）→ 根。层间 barrier 后本层节点读到的后代列块/
    // 对角块均已定稿；同层节点的写集合（祖先 diag/anc 对块）与读集合
    // （自身列块 (a,i)）互不相交，写-写冲突由 applyNodeB_ 内条带锁处理。
    for (int64_t l = 0; l <= maxLevel_; ++l)
        runLevel(l, numThreads);
}

void GaussianElimination::runLevel(int64_t l, int numThreads) {
    const std::vector<int64_t>& nodes = byLevel_[static_cast<size_t>(l)];
    const auto tl0 = std::chrono::steady_clock::now();
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            ElimData d;
            computeNodeA_(nodes[static_cast<size_t>(t)], d);
            applyNodeB_(nodes[static_cast<size_t>(t)], d);
        },
        numThreads);
    const double dt = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - tl0)
                          .count();
    std::lock_guard<std::mutex> lk(statsMutex_);
    timing_["par.level" + std::to_string(l)] = dt;
    timing_["par.levelN" + std::to_string(l)] =
        static_cast<double>(nodes.size());
    timing_["par.total"] += dt;
}

void GaussianElimination::runDag(int numThreads) {
    using Clock = std::chrono::steady_clock;
    const auto tg0 = Clock::now();
    DagRunner dag(numThreads);
    GeDagTasks info;
    buildDag(dag, info);
    for (const int64_t i : order_)
        if (tree_.node(i).children.empty())
            dag.ready(*info.f[static_cast<size_t>(i)]);
    dag.waitIdle();

    const double dt = std::chrono::duration<double>(Clock::now() - tg0).count();
    {
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["par.total"] += dt;
        timing_["dag.total"] = dt;
    }
}

void GaussianElimination::buildDag(DagRunner& dag, GeDagTasks& out) {
    out.f.assign(static_cast<size_t>(N_), nullptr);
    out.done.assign(static_cast<size_t>(N_), nullptr);

    // ---- 节点任务句柄（后序建图：孩子先于父）----
    struct NodeTasks {
        DagTask* F = nullptr;      // 分解 + M̂ 缓存
        DagTask* done = nullptr;   // 节点完成哨兵（→ F(parent) 边）
        std::vector<DagTask*> B, C, Dd, Ed;  // per 祖先
        std::vector<DagTask*> Dc, Ec;        // per 祖先对（展平 p<q）
        // 按消费者链精确释放（X/Y/WX 全宽稠密是内存大头：3·n_i·Σn_a·8B；
        // 等 DONE 才放会在首波肥叶同时存活时 OOM——250k 实测 2s 内 13GB）。
        std::vector<DagTask*> relX, relY, relWX;  // per 祖先
        DagTask* relW = nullptr;
    };
    std::vector<NodeTasks> nt(static_cast<size_t>(N_));
    std::vector<std::shared_ptr<ElimData>> data(static_cast<size_t>(N_));
    trace::mark("meta", "ge.begin", N_, maxLevel_);

    // 优先级 = 后序 rank*8 + phase（F0 B1 C2 D3 E4 rel5 DONE6）。
    // 必须是节点主序：FIFO 就绪队列会让全部叶子的 B 先于任何一个 C 执行，
    // X/Y(叶×根) 全量同时存活（250k 实测 ~16GB OOM，trace 4292 事件无一 C）；
    // 节点主序保证"先开链的节点先释放"，存活 X/Y 收缩为滑动窗口。
    std::vector<int64_t> rank(static_cast<size_t>(N_));
    for (int64_t r = 0; r < static_cast<int64_t>(order_.size()); ++r)
        rank[static_cast<size_t>(order_[static_cast<size_t>(r)])] = r;

    for (const int64_t i : order_) {
        const int64_t pr = rank[static_cast<size_t>(i)] * 8;
        const std::vector<int64_t> anc = tree_.ancestors(i);
        const bool leaf = isLeaf(i);
        const int64_t A = static_cast<int64_t>(anc.size());
        const int64_t lv = level_[static_cast<size_t>(i)];
        const int64_t ni = size_[static_cast<size_t>(i)];
        auto d = std::make_shared<ElimData>();
        data[static_cast<size_t>(i)] = d;
        // 预填全部键：任务阶段多线程只做值写入，unordered_map 无结构
        // 变更（并发插入会损坏桶结构——回归测试 out_of_range 的根因）。
        d->X.reserve(static_cast<size_t>(A) * 2);
        d->Y.reserve(static_cast<size_t>(A) * 2);
        d->xCols.reserve(static_cast<size_t>(A) * 2);
        d->yCols.reserve(static_cast<size_t>(A) * 2);
        d->WX.reserve(static_cast<size_t>(A) * 2);
        for (int64_t a : anc) {
            d->X.emplace(a, DenseMatrix());
            d->Y.emplace(a, DenseMatrix());
            d->xCols.emplace(a, std::vector<int32_t>());
            d->yCols.emplace(a, std::vector<int32_t>());
            d->WX.emplace(a, std::optional<DenseMatrix>());
        }
        auto& t = nt[static_cast<size_t>(i)];

        t.F = &dag.emplace([this, i, d, ni, lv] {
            trace::Scope s("ge", "F", i);
            if (trace::enabled()) { s.e.aux = ni; s.e.aux2 = lv; }
            if (std::getenv("AMLS_DUMP_K"))
                std::fprintf(stderr, "[dumpge] F i=%lld runs\n",
                             static_cast<long long>(i));
            factorNode_(i, *d);
            if (stats)
                recordNodeStat_(i);
        });
        t.F->prio = pr;   // phase 0
        // 节点就绪 = 直接孩子 DONE（跨层并行：子树先完成的父节点先跑）。
        for (const int64_t c : tree_.node(i).children)
            dag.depend(*t.F, *nt[static_cast<size_t>(c)].done);

        t.B.resize(static_cast<size_t>(A));
        t.C.resize(static_cast<size_t>(A));
        t.Dd.resize(static_cast<size_t>(A));
        t.Ed.resize(static_cast<size_t>(A));
        for (int64_t u = 0; u < A; ++u) {
            const int64_t a = anc[static_cast<size_t>(u)];
            auto& Bu = t.B[static_cast<size_t>(u)];
            auto& Cu = t.C[static_cast<size_t>(u)];
            auto& Du = t.Dd[static_cast<size_t>(u)];
            auto& Eu = t.Ed[static_cast<size_t>(u)];
            // B（式 7a/8a）：X_a / Y_a 回代。
            Bu = &dag.emplace([this, i, a, leaf, d, lv] {
                trace::Scope s("ge", "B", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                solveXYAnc_(i, *d, a, leaf);
            });
            Bu->prio = pr + 1;   // phase 1
            dag.depend(*Bu, *t.F);
            // C（式 8c/7c）：WX_a = K̂⁻¹(M̂·X_a)——算过即缓存，E 不重复算。
            Cu = &dag.emplace([this, i, a, d, lv] {
                trace::Scope s("ge", "C", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                (void)makeWX_(i, *d, a);
            });
            Cu->prio = pr + 2;   // phase 2
            dag.depend(*Cu, *Bu);
            // D（式 7d/8d）：K Schur 更新。
            Du = &dag.emplace([this, i, a, leaf, d, lv] {
                trace::Scope s("ge", "D.diag", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                schurKDiagTask_(i, *d, a, leaf);
            });
            Du->prio = pr + 3;   // phase 3
            dag.depend(*Du, *Bu);
            Eu = &dag.emplace([this, i, a, leaf, d, lv] {
                trace::Scope s("ge", "E.diag", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                congDiagTask_(i, *d, a, leaf);
            });
            Eu->prio = pr + 4;   // phase 4
            dag.depend(*Eu, *Bu);   // Y_a
            dag.depend(*Eu, *Cu);   // WX_a
        }
        const int64_t P = A * (A - 1) / 2;
        t.Dc.resize(static_cast<size_t>(P));
        t.Ec.resize(static_cast<size_t>(P));
        t.relX.resize(static_cast<size_t>(A));
        t.relY.resize(static_cast<size_t>(A));
        t.relWX.resize(static_cast<size_t>(A));
        int64_t pi = 0;
        for (int64_t q = 1; q < A; ++q)
            for (int64_t p = 0; p < q; ++p, ++pi) {
                const int64_t hi = anc[static_cast<size_t>(q)];
                const int64_t lo = anc[static_cast<size_t>(p)];
                t.Dc[static_cast<size_t>(pi)] =
                    &dag.emplace([this, i, hi, lo, leaf, d, lv] {
                        trace::Scope s("ge", "D.coup", i);
                        if (trace::enabled()) { s.e.aux = hi * N_ + lo; s.e.aux2 = lv; }
                        schurKCoupTask_(i, *d, hi, lo, leaf);
                    });
                t.Dc[static_cast<size_t>(pi)]->prio = pr + 3;
                dag.depend(*t.Dc[static_cast<size_t>(pi)], *t.B[static_cast<size_t>(p)]);
                t.Ec[static_cast<size_t>(pi)] =
                    &dag.emplace([this, i, hi, lo, leaf, d, lv] {
                        trace::Scope s("ge", "E.coup", i);
                        if (trace::enabled()) { s.e.aux = hi * N_ + lo; s.e.aux2 = lv; }
                        congCoupTask_(i, *d, hi, lo, leaf);
                    });
                t.Ec[static_cast<size_t>(pi)]->prio = pr + 4;
                dag.depend(*t.Ec[static_cast<size_t>(pi)], *t.B[static_cast<size_t>(p)]);  // Y_lo
                dag.depend(*t.Ec[static_cast<size_t>(pi)], *t.B[static_cast<size_t>(q)]);  // Y_hi
                dag.depend(*t.Ec[static_cast<size_t>(pi)], *t.C[static_cast<size_t>(p)]);  // WX_lo
            }

        // ---- 按消费者链精确释放（不阻塞 DONE，只缩短存活期）----
        // X_a ← D.diag(u) + D.coup(p==u)；且 C_u 的 makeWX gather X_a。
        // Y_a ← E.diag(u) + E.coup(p==u | q==u)。WX_a ← E.diag(u) + E.coup(p==u)。
        // W   ← 全部 C。释放后无二次使用（缓存命中路径都已在其消费者之前）。
        for (int64_t u = 0; u < A; ++u) {
            const int64_t a = anc[static_cast<size_t>(u)];
            t.relX[static_cast<size_t>(u)] =
                &dag.emplace([d, a] {
                    std::lock_guard<std::mutex> lk(d->relMut);
                    d->X.erase(a);
                });
            t.relX[static_cast<size_t>(u)]->prio = pr + 5;   // phase 5
            t.relY[static_cast<size_t>(u)] =
                &dag.emplace([d, a] {
                    std::lock_guard<std::mutex> lk(d->relMut);
                    d->Y.erase(a);
                });
            t.relY[static_cast<size_t>(u)]->prio = pr + 5;
            t.relWX[static_cast<size_t>(u)] =
                &dag.emplace([d, a] {
                    std::lock_guard<std::mutex> lk(d->relMut);
                    d->WX.erase(a);
                });
            t.relWX[static_cast<size_t>(u)]->prio = pr + 5;
        }
        t.relW = &dag.emplace([d] { d->W.resize(0, 0); });
        t.relW->prio = pr + 5;
        // 根节点 A=0 时下方循环全空 → relW 无依赖、又不是 ready 种子
        //（solver 只 ready 叶子 F）→ 悬挂；而 DONE 无条件等 relW，连带
        // done(根) 悬挂 → spec(根) 永不执行（res.k[根]=0，path:800 实测
        // p 少 1）。补 relW ← F：W 是 F 分解的产物缓存，F 后释放即安全。
        dag.depend(*t.relW, *t.F);
        for (int64_t u = 0; u < A; ++u) {
            auto& rX = *t.relX[static_cast<size_t>(u)];
            auto& rY = *t.relY[static_cast<size_t>(u)];
            auto& rW = *t.relWX[static_cast<size_t>(u)];
            dag.depend(rX, *t.C[static_cast<size_t>(u)]);
            dag.depend(rX, *t.Dd[static_cast<size_t>(u)]);
            dag.depend(rY, *t.Ed[static_cast<size_t>(u)]);
            dag.depend(rW, *t.Ed[static_cast<size_t>(u)]);
            dag.depend(*t.relW, *t.C[static_cast<size_t>(u)]);
        }
        for (int64_t pi2 = 0; pi2 < P; ++pi2) {
            // p 索引在 pair 展平中的位置：pi2 = q(q-1)/2 + p
            int64_t q2 = 1, base = 0;
            while (base + q2 <= pi2) { base += q2; ++q2; }
            const int64_t p2 = pi2 - base;
            dag.depend(*t.relX[static_cast<size_t>(p2)], *t.Dc[static_cast<size_t>(pi2)]);
            dag.depend(*t.relY[static_cast<size_t>(p2)], *t.Ec[static_cast<size_t>(pi2)]);
            dag.depend(*t.relY[static_cast<size_t>(q2)], *t.Ec[static_cast<size_t>(pi2)]);
            dag.depend(*t.relWX[static_cast<size_t>(p2)], *t.Ec[static_cast<size_t>(pi2)]);
        }

        // DONE：全部 D/E 完成后节点 i 消去收尾；顺带释放 X/Y/WX 工作内存
        //（与旧路径 eliminateNode_ 返回即析构 ElimData 等价）。
        // AMLS_FREE_FACTORS：打桩探针专用——节点消去完成后立即释放其 LDL
        // 因子（数值上破坏缩聚投影，仅用于验证"452 个叶因子常驻 = 250k OOM"
        // 假设；正式解法是缩聚统一 DAG 内释放）。
        static const bool probeFreeFactors = std::getenv("AMLS_FREE_FACTORS") != nullptr;
        t.done = &dag.emplace([this, d, i, lv, probeFreeFactors, &dag] {
            trace::Scope s("ge", "DONE", i);
            if (trace::enabled()) { s.e.aux2 = lv; }
            if (probeFreeFactors) {
                std::lock_guard<std::mutex> lk(structMutex_);
                sparseFactors_[static_cast<size_t>(i)].reset();
                denseFactors_[static_cast<size_t>(i)].reset();
            }
            {
                std::lock_guard<std::mutex> lk(d->relMut);
                d->X.clear();
                d->Y.clear();
                d->WX.clear();
            }
            d->W.resize(0, 0);
            // 负载自适应：每完成一个节点判定一次饥饿（就绪空 + 活跃少），
            // 连续 3 个节点成立则升任务内并行度档（只升不降）。
            const int before = dag.taskParallelism();
            dag.noteNodeDone();
            const int after = dag.taskParallelism();
            if (after > before)
                trace::mark("dag", "loadlevel", after);
        });
        t.done->prio = pr + 6;   // phase 6
        // 根节点 A=0 时上面循环全空 → done 无任何依赖，而 DagRunner
        // 只从显式 ready 的种子出发（solver ready 叶子 F），无依赖的
        // done 永远不会触发——统一 DAG 里 spec(root) 等不到 done(root)，
        // res.k[root] 停留初始 0（path:800 实测 p 少 1、根模态丢失）。
        // 补 done ← F：对 A>0 是冗余边（D/E 传递已等 F），对根是救活链。
        dag.depend(*t.done, *t.F);
        for (int64_t u = 0; u < A; ++u) {
            dag.depend(*t.done, *t.Dd[static_cast<size_t>(u)]);
            dag.depend(*t.done, *t.Ed[static_cast<size_t>(u)]);
        }
        for (int64_t pi2 = 0; pi2 < P; ++pi2) {
            dag.depend(*t.done, *t.Dc[static_cast<size_t>(pi2)]);
            dag.depend(*t.done, *t.Ec[static_cast<size_t>(pi2)]);
        }
        // DONE 与 relX/relY/relWX/relW 之间无数据依赖边但共享 d->X/Y/WX/W：
        // 二者常在同一完成事件后同时就绪，被不同 worker 并行弹出 →
        // erase/clear/resize 并发改同一 unordered_map → 堆破坏（随机 AV，
        // 46k/12 线程 ~40% 复现）。显式补序：释放任务先于 DONE。
        for (int64_t u = 0; u < A; ++u) {
            dag.depend(*t.done, *t.relX[static_cast<size_t>(u)]);
            dag.depend(*t.done, *t.relY[static_cast<size_t>(u)]);
            dag.depend(*t.done, *t.relWX[static_cast<size_t>(u)]);
        }
        dag.depend(*t.done, *t.relW);
    }

    // ---- 输出任务句柄（ready/waitIdle 由调用方负责）----
    for (const int64_t i : order_) {
        out.f[static_cast<size_t>(i)] = nt[static_cast<size_t>(i)].F;
        out.done[static_cast<size_t>(i)] = nt[static_cast<size_t>(i)].done;
    }
}

void GaussianElimination::releaseNode(int64_t i) {
    std::lock_guard<std::mutex> lk(structMutex_);
    kDiag_.erase(i);
    mDiag_.erase(i);
    for (int64_t a : tree_.ancestors(i)) {
        const int64_t k = key_(a, i);
        kCoupD_.erase(k);
        mCoupD_.erase(k);
    }
}

void GaussianElimination::shrinkNodeData(int64_t i, bool releaseFactor) const {
    // 与 releaseNode 不同：不 erase 表项（桶链不动，可与运行期无锁 find
    // 并发），只把已存在表项的稠密数据清零释放。调用方保证此刻无人再
    // 读节点 i 的块（DAG 依赖序）；其它线程的 find 可能查到这些空壳，
    // 但按依赖设计不会使用。
    static auto shrink = [](ColComp& blk) {
        blk.data.resize(0, 0);
        blk.cols.clear();
        blk.ident = false;
    };
    if (auto it = kDiag_.find(i); it != kDiag_.end())
        shrink(it->second);
    if (auto it = mDiag_.find(i); it != mDiag_.end())
        shrink(it->second);
    for (int64_t a : tree_.ancestors(i)) {
        const int64_t k = key_(a, i);
        if (auto it = kCoupD_.find(k); it != kCoupD_.end())
            shrink(it->second);
        if (auto it = mCoupD_.find(k); it != mCoupD_.end())
            shrink(it->second);
    }
    if (releaseFactor) {
        sparseFactors_[static_cast<size_t>(i)].reset();
        denseFactors_[static_cast<size_t>(i)].reset();
    }
}

void GaussianElimination::resetBlocks() {
    std::lock_guard<std::mutex> lk(structMutex_);
    decltype(kDiag_)().swap(kDiag_);
    decltype(mDiag_)().swap(mDiag_);
    decltype(kCoupD_)().swap(kCoupD_);
    decltype(mCoupD_)().swap(mCoupD_);
    // swap 后桶计数清零——重跑趟需恢复 reserve（运行期引用稳定性，
    // 见构造器注释；容量上界与构造时相同，按符号结构重算代价高，
    // 直接沿用节点/祖先结构的上界）。
    int64_t coupKeys = 16;
    for (const int64_t i : order_) {
        const int64_t a = static_cast<int64_t>(tree_.ancestors(i).size());
        coupKeys += a + a * (a - 1) / 2;
    }
    kCoupD_.reserve(static_cast<size_t>(coupKeys));
    mCoupD_.reserve(static_cast<size_t>(coupKeys));
    kDiag_.reserve(static_cast<size_t>(N_) + 1);
    mDiag_.reserve(static_cast<size_t>(N_) + 1);
}

double GaussianElimination::denseBytes() const {
    std::lock_guard<std::mutex> lk(structMutex_);
    double b = 0.0;
    auto sum = [&](const std::unordered_map<int64_t, ColComp>& tab) {
        for (const auto& [_, m] : tab) {
            b += m.data.size() * sizeof(double);
            b += static_cast<double>(m.cols.size()) * sizeof(int32_t);
        }
    };
    sum(kDiag_);
    sum(mDiag_);
    sum(kCoupD_);
    sum(mCoupD_);
    return b;
}

void GaussianElimination::zeroColStats(int64_t& zeroCols, int64_t& totalCols,
                                       int64_t& zeroBlocks,
                                       int64_t& totalBlocks) const {
    std::lock_guard<std::mutex> lk(structMutex_);
    zeroCols = totalCols = zeroBlocks = totalBlocks = 0;
    auto scan = [&](const std::unordered_map<int64_t, ColComp>& tab) {
        for (const auto& [_, m] : tab) {
            ++totalBlocks;
            if (m.ident) {
                int64_t zc = 0;
                for (int64_t j = 0; j < m.data.cols(); ++j) {
                    ++totalCols;
                    bool z = true;
                    for (int64_t r = 0; r < m.data.rows() && z; ++r)
                        if (m.data(r, j) != 0.0) z = false;
                    if (z) { ++zeroCols; ++zc; }
                }
                if (zc == m.data.cols()) ++zeroBlocks;
            } else {
                totalCols += m.full;
                int64_t zc = m.full - m.storedCols();  // 未存储列必为零
                for (int64_t j = 0; j < m.data.cols(); ++j) {
                    bool z = true;
                    for (int64_t r = 0; r < m.data.rows() && z; ++r)
                        if (m.data(r, j) != 0.0) z = false;
                    if (z) ++zc;
                }
                zeroCols += zc;
                if (zc == m.full) ++zeroBlocks;
            }
        }
    };
    scan(kDiag_);
    scan(mDiag_);
    scan(kCoupD_);
    scan(mCoupD_);
}

} // namespace amls
