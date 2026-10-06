// step 2a 实现：高斯消去（N1 约束模态回代 → N3 Mtemp → N4 K Schur / N5 M 更新）。
// 全部块乘走 blockops 的 accumulateProduct / accumulateTransposedProduct
// （OpenBLAS dgemm 后端），转置经 CblasTrans 标志隐式完成。
#include "amls/gaussian_elimination.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

using Dense = DefaultTraits::DenseMatrix;
using Sparse = DefaultTraits::SparseMatrix;

/// 稠密块显著非零元占比（|v| > 1e-10·max|v|）。
inline double significantFrac(const Eigen::Ref<const Dense>& A) {
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
inline int64_t significantRows(const Eigen::Ref<const Dense>& A) {
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
inline std::vector<int32_t> nonzeroColIdx(const Sparse& A) {
    std::vector<int32_t> idx;
    for (int64_t j = 0; j < A.cols(); ++j)
        if (A.col(j).nonZeros() > 0)
            idx.push_back(static_cast<int32_t>(j));
    return idx;
}

/// 稠密块的精确非零列索引（升序）。右端零列经可逆回代仍为零列，
/// 只需解非零列（u(i,a) 非零列 = K_{i,a} 非零列的数学保证）。
inline std::vector<int32_t> nonzeroColIdx(const Eigen::Ref<const Dense>& A) {
    std::vector<int32_t> idx;
    for (int64_t j = 0; j < A.cols(); ++j)
        for (int64_t r = 0; r < A.rows(); ++r)
            if (A(r, j) != 0.0) {
                idx.push_back(static_cast<int32_t>(j));
                break;
            }
    return idx;
}

/// 稀疏块的非零行位图。
inline ColMask rowMaskOf(const Sparse& A) {
    ColMask m;
    m.init(A.rows());
    for (int64_t c = 0; c < A.cols(); ++c)
        for (Sparse::InnerIterator it(A, static_cast<DefaultTraits::Index>(c));
             it; ++it)
            m.set(it.row());
    return m;
}

/// 稀疏块的非零列位图。
inline ColMask colMaskOf(const Sparse& A) {
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
    // u / Mtemp 键预建（运行期只写值，无结构变更）
    constraintModes_.reserve(static_cast<size_t>(coupKeys));
    temporaryMass_.reserve(static_cast<size_t>(coupKeys));
    for (const int64_t i : order_)
        for (const int64_t a : tree_.ancestors(i)) {
            constraintModes_.emplace(modeKey_(i, a), ColComp());
            temporaryMass_.emplace(modeKey_(i, a), ColComp());
        }
}

DefaultTraits::SparseMatrix GaussianElimination::sparseBlock_(
    const SparseMatrix& A, int64_t r0, int64_t r1, int64_t c0,
    int64_t c1) const {
    const int64_t m = r1 - r0, n = c1 - c0;
    std::vector<Eigen::Triplet<double, Index>> trips;
    trips.reserve(static_cast<size_t>(m));
    for (Index c = c0; c < c1; ++c) {
        for (SparseMatrix::InnerIterator it(A, c); it; ++it) {
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

    // 符号填充 pass：精确预算全部耦合块的终态列/行模式
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
            for (size_t v = 0; v < u; ++v) {  // lo：更低祖先
                const int64_t lo = anc[v];
                const int64_t k = key_(hi, lo);
                const ColMask& rKlo = rowK.at(key_(lo, i));
                const ColMask& rMlo = rowM.at(key_(lo, i));
                if (rKlo.count() != 0) {
                    if (rHi.count() != 0) {
                        // N4：K̄(hi,lo) += K_{hi,i}·u(i,lo)
                        // 积列 = u(i,lo) 列 = K_{lo,i} 非零行；行 ⊆ K_{hi,i} 行
                        colK[k].merge(rKlo);
                        rowK[k].merge(rHi);
                        // N5 第二项 u(i,hi)ᵀ·Mtemp(i,lo) 的 K 部分列
                        colM[k].merge(rKlo);
                        rowM[k].merge(rHi);
                    }
                    if (rMhi.count() != 0) {
                        // N5 第一项：M̄(hi,lo) += M_{hi,i}·u(i,lo)
                        // 积列 = u(i,lo) 列；行 ⊆ M_{hi,i} 行
                        colM[k].merge(rKlo);
                        rowM[k].merge(rMhi);
                    }
                }
                if (rMlo.count() != 0 && rHi.count() != 0) {
                    // N5 第二项的 M 部分列（Mtemp 列 = rowK ∪ rowM）
                    colM[k].merge(rMlo);
                    rowM[k].merge(rHi);
                }
            }
        }
    }
    // 列模式 = 耦合块出生表；行模式保留供 MassCouplingStore 预算 μ̃ 列模式
    for (auto& [k, m] : colK)
        if (m.count() != 0) kColFinal_[k] = std::move(m);
    for (auto& [k, m] : colM)
        if (m.count() != 0) mColFinal_[k] = std::move(m);
    rowKFinal_ = std::move(rowK);
    rowMFinal_ = std::move(rowM);
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
    // releaseNodeData/resetBlocks，彼时无并发消去）。仅创建走 structMutex_。
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
        // 表需保留供二次物化。
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

void GaussianElimination::solveWithFactor_(int64_t i, const DenseMatrix& rhs,
                                           DenseMatrix& solution) const {
    if (sparseFactors_[static_cast<size_t>(i)])
        sparseFactors_[static_cast<size_t>(i)]->solveWithFactor(rhs, solution);
    else
        denseFactors_[static_cast<size_t>(i)]->solveWithFactor(rhs, solution);
}

// ---------------------------------------------------------------------------
// N1：约束模态 u(i,a) = Ψ(i,a) = −K̂_ii⁻¹·K_{i,a}
// ---------------------------------------------------------------------------
void GaussianElimination::computeConstraintModes(int64_t i) {
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
        // diag 块定稿于全部后代的 N4/N5（依赖边保证）；确保诞生
        //（无后代写入时 = 原始稀疏块的稠密化）。M 对角同理（N7 消费）。
        ColComp& kd = kDiagTarget_(i);
        (void)mDiagTarget_(i);
        if (static const char* dump = std::getenv("AMLS_DUMP_KDIAG");
            dump != nullptr) {
            const std::string f =
                std::string("kdiag_") + dump + "_" +
                std::to_string(i) + ".bin";
            std::ofstream os(f, std::ios::binary);
            const DenseMatrix& kdd = kd.data;
            const int64_t r = kdd.rows();
            os.write(reinterpret_cast<const char*>(&r), sizeof(r));
            os.write(reinterpret_cast<const char*>(kdd.data()),
                     static_cast<std::streamsize>(r * r) *
                         static_cast<std::streamsize>(sizeof(double)));
        }
        // 根节点无祖先：u 无消费者、回代恢复用存储的 u 而非因子——
        // 分解毫无意义，跳过（省掉全图最大一次稠密 LDL）。
        if (!tree_.ancestors(i).empty()) {
            denseFactors_[static_cast<size_t>(i)] =
                std::make_unique<DenseLDLTSolver<DefaultTraits>>();
            try {
                denseFactors_[static_cast<size_t>(i)]->factorizeDense(kd.data);
            } catch (const std::exception& e) {
                AMLS_LOG_ERROR("[ge] 节点 {} (n={}, leaf={}) LDL 失败: {}", i,
                               size_[static_cast<size_t>(i)], leaf, e.what());
                throw;
            }
        }
    }
    const auto t1 = Clock::now();
    {
        const double dt = std::chrono::duration<double>(t1 - t0).count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["factor"] += dt;
        timing_[std::string("factor") + who] += dt;
    }

    const std::vector<int64_t> anc = tree_.ancestors(i);
    if (anc.empty()) {
        if (stats)
            recordNodeStat_(i);
        return;   // 根节点：无 u 需求
    }

    // ---- 2. 逐祖先窄右端回代（右端零列跳过：u 的非零列 = K_{i,a} 非零列）
    // 右端 K_{i,a} = (a,i) 块ᵀ：叶由稀疏块非零行直接构造窄右端（免全物化
    // 转置）；分割节点 expandT 后 scan 非零列再 gather。
    auto& prof = phaseProf();
    for (const int64_t a : anc) {
        const auto tp0 = Clock::now();
        const int64_t na = size_[static_cast<size_t>(a)];
        ColComp& u = constraintModes_.at(modeKey_(i, a));   // 预建键
        u.full = na;
        u.ident = false;
        u.cols.clear();
        u.data.resize(0, 0);
        if (leaf) {
            // 右端 = K_{i,a} 的非零列 = (a,i) 稀疏块的非零行
            const SparseMatrix& block = origK_(a, i);
            ColMask mk;
            mk.init(na);
            for (int64_t c = 0; c < block.cols(); ++c)
                for (SparseMatrix::InnerIterator it(
                         block, static_cast<Index>(c));
                     it; ++it)
                    mk.set(it.row());
            auto idx = mk.indices();
            if (!idx.empty()) {
                DenseMatrix rhs =
                    DenseMatrix::Zero(size_[static_cast<size_t>(i)],
                                      static_cast<int64_t>(idx.size()));
                std::vector<int64_t> pos(static_cast<size_t>(na), -1);
                for (int64_t p = 0; p < static_cast<int64_t>(idx.size()); ++p)
                    pos[static_cast<size_t>(idx[static_cast<size_t>(p)])] = p;
                for (int64_t c = 0; c < block.cols(); ++c)
                    for (SparseMatrix::InnerIterator it(
                             block, static_cast<Index>(c));
                         it; ++it)
                        rhs(c, pos[static_cast<size_t>(it.row())]) = it.value();
                u.cols = std::move(idx);
                solveWithFactor_(i, rhs, u.data);
                u.data *= -1.0;   // Ψ = −K̂⁻¹K
            }
        } else {
            const DenseMatrix transposed =
                blockops::expandTranspose(kCoup_(a, i));
            auto idx = nonzeroColIdx(transposed);
            if (!idx.empty()) {
                DenseMatrix rhs(size_[static_cast<size_t>(i)],
                                static_cast<int64_t>(idx.size()));
                for (int64_t p = 0; p < static_cast<int64_t>(idx.size()); ++p)
                    rhs.col(p) = transposed.col(idx[static_cast<size_t>(p)]);
                u.cols = std::move(idx);
                solveWithFactor_(i, rhs, u.data);
                u.data *= -1.0;   // Ψ = −K̂⁻¹K
            }
        }
        profAdd(prof.solveRhs,
                std::chrono::duration<double>(Clock::now() - tp0).count());

        if (stats) {
            GeBlockStat bs;
            bs.node = i;
            bs.leaf = leaf;
            bs.level = level_[static_cast<size_t>(i)];
            bs.anc  = a;
            bs.rows = na;
            bs.cols = size_[static_cast<size_t>(i)];
            if (leaf) {
                bs.nonzeroCols =
                    static_cast<int64_t>(nonzeroColIdx(origK_(a, i)).size());
                bs.kFrac = significantFrac(DenseMatrix(origK_(a, i)));
            } else {
                const ColComp& kb = kCoup_(a, i);
                bs.nonzeroCols = kb.storedCols();
                bs.kFrac = significantFrac(kb.data);
            }
            bs.uNZRows = significantRows(u.data);
            bs.uFrac = significantFrac(u.data);
            std::lock_guard<std::mutex> lk(statsMutex_);
            stats->blocks.push_back(bs);
        }
    }
    // ---- 3. 因子使命完成（N3–N5、缩聚、回代恢复均不再需要 K̂⁻¹）----
    sparseFactors_[static_cast<size_t>(i)].reset();
    denseFactors_[static_cast<size_t>(i)].reset();
    if (stats)
        recordNodeStat_(i);
    {
        const double dt = std::chrono::duration<double>(Clock::now() - t1)
                              .count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["solve"] += dt;
        timing_[std::string("solve") + who] += dt;
    }
}

// ---------------------------------------------------------------------------
// N3：Mtemp(i,a) = M_{i,a} + M_ii·u(i,a)（列 = u 列 ∪ M 耦合行模式）
// ---------------------------------------------------------------------------
void GaussianElimination::buildTemporaryMass(int64_t i) {
    using Clock = std::chrono::steady_clock;
    const std::vector<int64_t> anc = tree_.ancestors(i);
    if (anc.empty())
        return;
    const bool leaf = isLeaf(i);
    const char* who = leaf ? ".leaf" : ".sep";
    const auto t0 = Clock::now();
    const int64_t ni = size_[static_cast<size_t>(i)];

    for (const int64_t a : anc) {
        const auto tp0 = Clock::now();
        const int64_t na = size_[static_cast<size_t>(a)];
        const ColComp& u = constraintModes_.at(modeKey_(i, a));
        // M_{i,a} 稠密物化（n_i × n_a 全）：叶 = 稀疏转置；分割 = expandT
        DenseMatrix mCouplingT =
            leaf ? blockops::expandTranspose(origM_(a, i))
                 : blockops::expandTranspose(mCoup_(a, i));
        // M_ii·u：M_ii = 叶稀疏原块 / 分割节点消去后稠密对角
        DenseMatrix product;
        if (!u.empty()) {
            if (leaf)
                product = DenseMatrix(origMDiag_.at(i) * u.data);
            else
                blas::gemm0(CblasNoTrans, CblasNoTrans, mDiagTarget_(i).data,
                            u.data, product);
        }
        // 列 = u.cols ∪ M_{i,a} 非零列（升序并集；两集合可重叠——
        // 分割节点的 M 耦合更新列恰落进 u.cols，merge 会产生重复列，
        // 下游按 cols 驱动的 gather/散布会对同一物理列累加两次）
        const auto mCols = nonzeroColIdx(mCouplingT);
        std::vector<int32_t> cols;
        cols.reserve(u.cols.size() + mCols.size());
        std::set_union(u.cols.begin(), u.cols.end(), mCols.begin(),
                       mCols.end(), std::back_inserter(cols));
        ColComp& mt = temporaryMass_.at(modeKey_(i, a));   // 预建键
        mt.full = na;
        mt.ident = false;   // 作为 C 操作数时经 cols 驱动求和维，恒非 ident
        mt.cols = cols;
        mt.data = DenseMatrix::Zero(ni, static_cast<int64_t>(cols.size()));
        for (int64_t p = 0; p < static_cast<int64_t>(cols.size()); ++p) {
            const int32_t col = cols[static_cast<size_t>(p)];
            mt.data.col(p) = mCouplingT.col(col);
            if (!u.empty()) {
                const auto it = std::lower_bound(u.cols.begin(), u.cols.end(),
                                                 col);
                if (it != u.cols.end() && *it == col)
                    mt.data.col(p) += product.col(it - u.cols.begin());
            }
        }
        profAdd(phaseProf().mtempBuild,
                std::chrono::duration<double>(Clock::now() - tp0).count());
    }
    const double dt =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(statsMutex_);
    timing_["mtemp"] += dt;
    timing_[std::string("mtemp") + who] += dt;
}

// ---------------------------------------------------------------------------
// N4：K 单边 Schur 更新（A += B·C 规范形态）
// ---------------------------------------------------------------------------
void GaussianElimination::stiffnessDiagUpdate(int64_t i, int64_t a) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    const ColComp& u = constraintModes_.at(modeKey_(i, a));
    // 尾部大任务在就绪队列空闸下临时放开 BLAS 多线程。
    const double gf =
        2.0 * size_[static_cast<size_t>(a)] * size_[static_cast<size_t>(i)] *
        u.storedCols() / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(a));
    ColComp& tgt = kDiagTarget_(a);
    if (isLeaf(i))
        blockops::accumulateProduct(tgt, origK_(a, i), u);
    else
        blockops::accumulateProduct(tgt, kCoup_(a, i), u);
    profAdd(prof.schurKdiag,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
}

void GaussianElimination::stiffnessCoupUpdate(int64_t i, int64_t hi,
                                              int64_t lo) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    const ColComp& u = constraintModes_.at(modeKey_(i, lo));
    const double gf =
        2.0 * size_[static_cast<size_t>(hi)] * size_[static_cast<size_t>(i)] *
        u.storedCols() / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(key_(hi, lo)));
    ColComp& tgt = kCoupTarget_(hi, lo);
    if (isLeaf(i))
        blockops::accumulateProduct(tgt, origK_(hi, i), u);
    else
        blockops::accumulateProduct(tgt, kCoup_(hi, i), u);
    profAdd(prof.schurKcoup,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tp0).count());
}

// ---------------------------------------------------------------------------
// N5：M 更新（Mtemp 形式：M̄ += M·u + uᵀ·Mtemp，三重积消失）
// ---------------------------------------------------------------------------
void GaussianElimination::massDiagUpdate(int64_t i, int64_t a) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    const ColComp& u = constraintModes_.at(modeKey_(i, a));
    const ColComp& mt = temporaryMass_.at(modeKey_(i, a));
    const double gf =
        2.0 * size_[static_cast<size_t>(i)] *
        (size_[static_cast<size_t>(a)] * u.storedCols() +
         u.storedCols() * mt.storedCols()) / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(a));
    ColComp& tgt = mDiagTarget_(a);
    const auto tp1 = std::chrono::steady_clock::now();
    if (isLeaf(i))
        blockops::accumulateProduct(tgt, origM_(a, i), u);
    else
        blockops::accumulateProduct(tgt, mCoup_(a, i), u);
    const auto tp2 = std::chrono::steady_clock::now();
    blockops::accumulateTransposedProduct(tgt, u, mt);
    const auto tp3 = std::chrono::steady_clock::now();
    profAdd(prof.massDiagTerm1,
            std::chrono::duration<double>(tp2 - tp1).count());
    profAdd(prof.massDiagTerm2,
            std::chrono::duration<double>(tp3 - tp2).count());
}

void GaussianElimination::massCoupUpdate(int64_t i, int64_t hi, int64_t lo) {
    auto& prof = phaseProf();
    const auto tp0 = std::chrono::steady_clock::now();
    const ColComp& uLo = constraintModes_.at(modeKey_(i, lo));
    const ColComp& uHi = constraintModes_.at(modeKey_(i, hi));
    const ColComp& mtLo = temporaryMass_.at(modeKey_(i, lo));
    const double gf =
        2.0 * size_[static_cast<size_t>(i)] *
        (size_[static_cast<size_t>(hi)] * uLo.storedCols() +
         uHi.storedCols() * mtLo.storedCols()) / 1e9;
    auto mb = blas::maybeBudget(gf);
    std::lock_guard<std::mutex> lk(keyLock_(key_(hi, lo)));
    ColComp& tgt = mCoupTarget_(hi, lo);
    const auto tp1 = std::chrono::steady_clock::now();
    if (isLeaf(i))
        blockops::accumulateProduct(tgt, origM_(hi, i), uLo);
    else
        blockops::accumulateProduct(tgt, mCoup_(hi, i), uLo);
    const auto tp2 = std::chrono::steady_clock::now();
    blockops::accumulateTransposedProduct(tgt, uHi, mtLo);
    const auto tp3 = std::chrono::steady_clock::now();
    profAdd(prof.massCoupTerm1,
            std::chrono::duration<double>(tp2 - tp1).count());
    profAdd(prof.massCoupTerm2,
            std::chrono::duration<double>(tp3 - tp2).count());
}

// ---------------------------------------------------------------------------
// 释放（统一 DAG 内"边算边放"：值缩容不动结构，与无锁 find 并发安全）
// ---------------------------------------------------------------------------
void GaussianElimination::releaseCouplingColumn(int64_t i) const {
    for (const int64_t a : tree_.ancestors(i)) {
        const int64_t k = key_(a, i);
        if (auto it = kCoupD_.find(k); it != kCoupD_.end())
            it->second.shrink();
        if (auto it = mCoupD_.find(k); it != mCoupD_.end())
            it->second.shrink();
    }
}

void GaussianElimination::releaseDiagonalBlocks(int64_t i) const {
    if (auto it = kDiag_.find(i); it != kDiag_.end())
        it->second.shrink();
    if (auto it = mDiag_.find(i); it != mDiag_.end())
        it->second.shrink();
}

void GaussianElimination::releaseTemporaryMass(int64_t i) const {
    for (const int64_t a : tree_.ancestors(i))
        if (auto it = temporaryMass_.find(modeKey_(i, a));
            it != temporaryMass_.end())
            it->second.shrink();
}

void GaussianElimination::releaseConstraintModes(int64_t i) const {
    for (const int64_t a : tree_.ancestors(i))
        if (auto it = constraintModes_.find(modeKey_(i, a));
            it != constraintModes_.end())
            it->second.shrink();
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

void GaussianElimination::eliminateNode_(int64_t i) {
    using Clock = std::chrono::steady_clock;
    const std::vector<int64_t> anc = tree_.ancestors(i);
    computeConstraintModes(i);
    buildTemporaryMass(i);
    if (anc.empty())
        return;
    const bool leaf = isLeaf(i);
    const char* who = leaf ? ".leaf" : ".sep";
    auto t2 = Clock::now();
    for (int64_t a : anc)
        stiffnessDiagUpdate(i, a);
    for (size_t p = 0; p < anc.size(); ++p)
        for (size_t q = p + 1; q < anc.size(); ++q)
            stiffnessCoupUpdate(i, anc[q], anc[p]);
    {
        const double dt = std::chrono::duration<double>(Clock::now() - t2)
                              .count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["schurK"] += dt;
        timing_[std::string("schurK") + who] += dt;
    }
    t2 = Clock::now();
    for (int64_t a : anc)
        massDiagUpdate(i, a);
    for (size_t p = 0; p < anc.size(); ++p)
        for (size_t q = p + 1; q < anc.size(); ++q)
            massCoupUpdate(i, anc[q], anc[p]);
    {
        const double dt = std::chrono::duration<double>(Clock::now() - t2)
                              .count();
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["congrM"] += dt;
        timing_[std::string("congrM") + who] += dt;
    }
}

void GaussianElimination::run(int numThreads) {
    if (numThreads <= 1) {
        for (const int64_t i : order_)
            eliminateNode_(i);
        return;
    }
    // 层并行：level=0（叶）→ 根。层间 barrier 后本层节点读到的后代列块/
    // 对角块均已定稿；同层节点的写集合（祖先 diag/anc 对块）与读集合
    // （自身列块 (a,i)）互不相交，写-写冲突由条带锁处理。
    for (int64_t l = 0; l <= maxLevel_; ++l)
        runLevel(l, numThreads);
}

void GaussianElimination::runLevel(int64_t l, int numThreads) {
    const std::vector<int64_t>& nodes = byLevel_[static_cast<size_t>(l)];
    const auto tl0 = std::chrono::steady_clock::now();
    parallelFor(
        0, static_cast<int64_t>(nodes.size()),
        [&](int64_t t) {
            eliminateNode_(nodes[static_cast<size_t>(t)]);
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
    EliminationDagTasks info;
    buildDag(dag, info);
    for (const int64_t i : order_)
        if (tree_.node(i).children.empty())
            dag.ready(*info.n1[static_cast<size_t>(i)]);
    dag.waitIdle();

    const double dt = std::chrono::duration<double>(Clock::now() - tg0).count();
    {
        std::lock_guard<std::mutex> lk(statsMutex_);
        timing_["par.total"] += dt;
        timing_["dag.total"] = dt;
    }
}

void GaussianElimination::buildDag(DagRunner& dag, EliminationDagTasks& out) {
    out.n1.assign(static_cast<size_t>(N_), nullptr);
    out.n3.assign(static_cast<size_t>(N_), nullptr);
    out.updatesDone.assign(static_cast<size_t>(N_), nullptr);

    struct NodeTasks {
        DagTask* n1 = nullptr;
        DagTask* n3 = nullptr;
        DagTask* updatesDone = nullptr;   // N4/N5 汇合哨兵
        DagTask* freeCoup = nullptr;      // N2
        DagTask* done = nullptr;          // 负载自适应探针
        std::vector<DagTask*> n4d, n5d;   // per 祖先
        std::vector<DagTask*> n4c, n5c;   // per 祖先对（展平 p<q）
    };
    std::vector<NodeTasks> nt(static_cast<size_t>(N_));
    trace::mark("meta", "ge.begin", N_, maxLevel_);

    // 优先级 = 后序 rank*16 + phase（节点主序：FIFO 会让全部叶子的后续
    // 任务先于任何一个收尾任务执行，块全量同时存活 OOM；节点主序保证
    // "先开链的节点先释放"，存活块收缩为滑动窗口）。
    std::vector<int64_t> rank(static_cast<size_t>(N_));
    for (int64_t r = 0; r < static_cast<int64_t>(order_.size()); ++r)
        rank[static_cast<size_t>(order_[static_cast<size_t>(r)])] = r;

    for (const int64_t i : order_) {
        const int64_t pr = rank[static_cast<size_t>(i)] * 16;
        const std::vector<int64_t> anc = tree_.ancestors(i);
        const int64_t A = static_cast<int64_t>(anc.size());
        const int64_t lv = level_[static_cast<size_t>(i)];
        const int64_t ni = size_[static_cast<size_t>(i)];
        auto& t = nt[static_cast<size_t>(i)];

        // ---- N1（phase 0）：约束模态回代。就绪 = 直接孩子 N4/N5 完成
        //（跨层并行：子树先完成的父节点先跑；后代更新沿哨兵链传递）。----
        t.n1 = &dag.emplace([this, i, ni, lv] {
            trace::Scope s("ge", "N1", i);
            if (trace::enabled()) { s.e.aux = ni; s.e.aux2 = lv; }
            computeConstraintModes(i);
        });
        t.n1->prio = pr;
        for (const int64_t c : tree_.node(i).children)
            dag.depend(*t.n1, *nt[static_cast<size_t>(c)].updatesDone);

        // ---- N3（phase 1）：Mtemp 构造 ← N1 ----
        t.n3 = &dag.emplace([this, i, lv] {
            trace::Scope s("ge", "N3", i);
            if (trace::enabled()) s.e.aux2 = lv;
            buildTemporaryMass(i);
        });
        t.n3->prio = pr + 1;
        dag.depend(*t.n3, *t.n1);

        // ---- N4（phase 2）/ N5（phase 3）----
        t.n4d.resize(static_cast<size_t>(A));
        t.n5d.resize(static_cast<size_t>(A));
        for (int64_t u = 0; u < A; ++u) {
            const int64_t a = anc[static_cast<size_t>(u)];
            auto& n4 = t.n4d[static_cast<size_t>(u)];
            auto& n5 = t.n5d[static_cast<size_t>(u)];
            n4 = &dag.emplace([this, i, a, lv] {
                trace::Scope s("ge", "N4.diag", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                stiffnessDiagUpdate(i, a);
            });
            n4->prio = pr + 2;
            dag.depend(*n4, *t.n1);
            n5 = &dag.emplace([this, i, a, lv] {
                trace::Scope s("ge", "N5.diag", i);
                if (trace::enabled()) { s.e.aux = a; s.e.aux2 = lv; }
                massDiagUpdate(i, a);
            });
            n5->prio = pr + 3;
            dag.depend(*n5, *t.n1);
            dag.depend(*n5, *t.n3);
        }
        const int64_t P = A * (A - 1) / 2;
        t.n4c.resize(static_cast<size_t>(P));
        t.n5c.resize(static_cast<size_t>(P));
        int64_t pi = 0;
        for (int64_t q = 1; q < A; ++q)
            for (int64_t p = 0; p < q; ++p, ++pi) {
                const int64_t hi = anc[static_cast<size_t>(q)];
                const int64_t lo = anc[static_cast<size_t>(p)];
                t.n4c[static_cast<size_t>(pi)] =
                    &dag.emplace([this, i, hi, lo, lv] {
                        trace::Scope s("ge", "N4.coup", i);
                        if (trace::enabled()) { s.e.aux = hi * N_ + lo; s.e.aux2 = lv; }
                        stiffnessCoupUpdate(i, hi, lo);
                    });
                t.n4c[static_cast<size_t>(pi)]->prio = pr + 2;
                dag.depend(*t.n4c[static_cast<size_t>(pi)], *t.n1);
                t.n5c[static_cast<size_t>(pi)] =
                    &dag.emplace([this, i, hi, lo, lv] {
                        trace::Scope s("ge", "N5.coup", i);
                        if (trace::enabled()) { s.e.aux = hi * N_ + lo; s.e.aux2 = lv; }
                        massCoupUpdate(i, hi, lo);
                    });
                t.n5c[static_cast<size_t>(pi)]->prio = pr + 3;
                dag.depend(*t.n5c[static_cast<size_t>(pi)], *t.n1);
                dag.depend(*t.n5c[static_cast<size_t>(pi)], *t.n3);
            }

        // ---- updatesDone（phase 4）：N4/N5 汇合哨兵；父节点 N1 的锚点。
        // 根节点 A=0 时无 N4/N5 → 补 ← N1 防悬挂（冗余边无代价）。----
        t.updatesDone = &dag.emplace([] {});
        t.updatesDone->prio = pr + 4;
        dag.depend(*t.updatesDone, *t.n1);
        for (int64_t u = 0; u < A; ++u) {
            dag.depend(*t.updatesDone, *t.n4d[static_cast<size_t>(u)]);
            dag.depend(*t.updatesDone, *t.n5d[static_cast<size_t>(u)]);
        }
        for (int64_t p2 = 0; p2 < P; ++p2) {
            dag.depend(*t.updatesDone, *t.n4c[static_cast<size_t>(p2)]);
            dag.depend(*t.updatesDone, *t.n5c[static_cast<size_t>(p2)]);
        }

        // ---- N2（phase 5）：耦合列释放 ← N1/N3/N4/N5 全部（经哨兵）----
        if (A > 0) {
            t.freeCoup = &dag.emplace([this, i] {
                trace::Scope s("ge", "N2.free", i);
                releaseCouplingColumn(i);
            });
            t.freeCoup->prio = pr + 5;
            dag.depend(*t.freeCoup, *t.updatesDone);
            dag.depend(*t.freeCoup, *t.n3);
        }

        // ---- done（phase 6）：负载自适应探针（饥饿升档）----
        t.done = &dag.emplace([&dag, lv] {
            trace::Scope s("ge", "DONE", -1);
            if (trace::enabled()) s.e.aux2 = lv;
            const int before = dag.taskParallelism();
            dag.noteNodeDone();
            const int after = dag.taskParallelism();
            if (after > before)
                trace::mark("dag", "loadlevel", after);
        });
        t.done->prio = pr + 6;
        dag.depend(*t.done, *t.updatesDone);

        out.n1[static_cast<size_t>(i)] = t.n1;
        out.n3[static_cast<size_t>(i)] = t.n3;
        out.updatesDone[static_cast<size_t>(i)] = t.updatesDone;
    }
}

void GaussianElimination::releaseNodeData(int64_t i,
                                          bool keepConstraintModes) {
    std::lock_guard<std::mutex> lk(structMutex_);
    kDiag_.erase(i);
    mDiag_.erase(i);
    for (const int64_t a : tree_.ancestors(i)) {
        const int64_t k = key_(a, i);
        kCoupD_.erase(k);
        mCoupD_.erase(k);
        if (auto it = temporaryMass_.find(modeKey_(i, a));
            it != temporaryMass_.end())
            it->second.shrink();
        if (!keepConstraintModes)
            if (auto it = constraintModes_.find(modeKey_(i, a));
                it != constraintModes_.end())
                it->second.shrink();
    }
}

void GaussianElimination::resetBlocks() {
    std::lock_guard<std::mutex> lk(structMutex_);
    decltype(kDiag_)().swap(kDiag_);
    decltype(mDiag_)().swap(mDiag_);
    decltype(kCoupD_)().swap(kCoupD_);
    decltype(mCoupD_)().swap(mCoupD_);
    // swap 后桶计数清零——重跑趟需恢复 reserve（运行期引用稳定性，
    // 见构造器注释）。
    int64_t coupKeys = 16;
    for (const int64_t i : order_) {
        const int64_t a = static_cast<int64_t>(tree_.ancestors(i).size());
        coupKeys += a + a * (a - 1) / 2;
    }
    kCoupD_.reserve(static_cast<size_t>(coupKeys));
    mCoupD_.reserve(static_cast<size_t>(coupKeys));
    kDiag_.reserve(static_cast<size_t>(N_) + 1);
    mDiag_.reserve(static_cast<size_t>(N_) + 1);
    // u / Mtemp：键保留（预建语义），值缩容
    for (auto& [_, v] : constraintModes_)
        v.shrink();
    for (auto& [_, v] : temporaryMass_)
        v.shrink();
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
    sum(constraintModes_);
    sum(temporaryMass_);
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
