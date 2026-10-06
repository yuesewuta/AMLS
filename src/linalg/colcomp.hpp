#pragma once
// 列压缩块（ColComp）与块乘积累加算子层。
//
// 设计要点：
//   * ColComp = { cols(升序局部列号), data, full, ident }。ident 为真时
//     cols 为空、data 即全列稠密（diag 块、近根全耦合块的退化情形），
//     算子对 ident 走原始 BLAS 零间接开销路径。
//   * 全部块更新化简为两种规范形态（依赖关系拆解表 N4/N5/N8/N9/N10）：
//       accumulateProduct            A += B·C
//       accumulateTransposedProduct  A += Bᵀ·C
//     B 的存储列 = 求和维（驱动 C 的行 gather）；接收块压缩时积只在
//     A.cols ∩ 积列集 上计算（零列整列跳过，dgemm 的 n 维按交集缩小）。
//   * 乘法方向约定：B 为乘子（其 cols 限制求和维并同时驱动右端行 gather），
//     A 为接收块（压缩接收时积只在其存储列上计算）。
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include <openblas/cblas.h>

#include "core/types.hpp"
#include "linalg/blas.hpp"

namespace amls {

/// 算子层并行度（thread_local）：DagRunner 在每个任务入口按当前负载
/// 档位设置，任务内的列分组并行 dgemm 读取它。任务中途升档不影响已
/// 运行任务（入口快照）；不在 DAG worker 里的调用恒为 1（串行）。
inline int& opsTlsThreads() {
    static thread_local int value = 1;
    return value;
}

/// 原始指针列分组并行 dgemm（tb=CblasNoTrans 专用）：
/// C(:, jseg) = alpha·op(A)·op(B)(:, jseg) + beta·C(:,jseg)。
/// op(B) 的任意列段都是连续内存，各线程写不相交列段，无竞争、无锁、
/// 不依赖 BLAS 内部线程（规避 libgomp 嵌套并行区风险）。
inline void dgemmColsRaw(CBLAS_TRANSPOSE ta, int m, int n, int k,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb, double beta, double* C,
                         int ldc, int nthreads) {
    if (n <= 0 || m <= 0) return;
    static const bool forceSerial = std::getenv("AMLS_DGEMM_SERIAL") != nullptr;
    if (forceSerial) nthreads = 1;
    nthreads = std::min<int>(nthreads, (n + 31) / 32);
    if (nthreads <= 1) {
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, n, k, alpha, A, lda,
                    B, ldb, beta, C, ldc);
        return;
    }
    auto runSeg = [&](int j0, int j1) {
        if (j1 <= j0) return;
        const int jn = j1 - j0;
        blas::pinThisThread();   // 临时线程：OpenBLAS(OMP 版) ICV 钉 1，
                                 // 否则每线程内 dgemm 再开满核 → 过度订阅
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, jn, k, alpha, A, lda,
                    B + static_cast<int64_t>(j0) * ldb, ldb, beta,
                    C + static_cast<int64_t>(j0) * ldc, ldc);
    };
    std::vector<std::thread> ths;
    ths.reserve(static_cast<size_t>(nthreads - 1));
    const int64_t base = n / nthreads, extra = n % nthreads;
    int64_t j = 0;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t jn = base + (t < extra ? 1 : 0);
        const int64_t j0 = j, j1 = j + jn;
        j = j1;
        if (t == nthreads - 1)
            runSeg(j0, j1);
        else
            ths.emplace_back(runSeg, static_cast<int>(j0),
                             static_cast<int>(j1));
    }
    for (auto& th : ths) th.join();
}

/// 列分组并行 dgemm：C(:, jseg) = alpha·op(A)·op(B)(:, jseg) + beta·C(:,jseg)。
/// tb=Trans 时 op(B) 的列段 = B 的行段（列主序下行不连续）→ 线程内先
/// 物化成连续段再算；对调用方完全等价于单次 dgemm。
inline void dgemmCols(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const DefaultTraits::DenseMatrix& A,
                      const DefaultTraits::DenseMatrix& B, double beta,
                      DefaultTraits::DenseMatrix& C, int nthreads) {
    using Dense = DefaultTraits::DenseMatrix;
    const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
    const int k = static_cast<int>((ta == CblasNoTrans) ? A.cols() : A.rows());
    const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
    if (n <= 0 || m <= 0) return;
    if (tb == CblasNoTrans) {
        dgemmColsRaw(ta, m, n, k, alpha, A.data(),
                     static_cast<int>(A.rows()), B.data(),
                     static_cast<int>(B.rows()), beta, C.data(),
                     static_cast<int>(C.rows()), nthreads);
        return;
    }
    static const bool forceSerial = std::getenv("AMLS_DGEMM_SERIAL") != nullptr;
    if (forceSerial) nthreads = 1;
    nthreads = std::min<int>(nthreads, static_cast<int>((n + 31) / 32));
    if (nthreads <= 1) {
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A.data(),
                    static_cast<int>(A.rows()), B.data(),
                    static_cast<int>(B.rows()), beta, C.data(),
                    static_cast<int>(C.rows()));
        return;
    }
    const int ldb = static_cast<int>(B.rows());
    const int ldc = static_cast<int>(C.rows());
    auto runSeg = [&](int j0, int j1) {
        if (j1 <= j0) return;
        const int jn = j1 - j0;
        blas::pinThisThread();
        // op(B)=Bᵀ，op(B)(r, j0+c) = B(j0+c, r)：物化 bSeg(r,c) = B(j0+c, r)。
        Dense bSeg(k, jn);
        for (int c = 0; c < jn; ++c)
            for (int r = 0; r < k; ++r)
                bSeg(r, c) = B(static_cast<int64_t>(j0) + c,
                               static_cast<int64_t>(r));
        double* cPtr = C.data() + static_cast<int64_t>(j0) * ldc;
        cblas_dgemm(CblasColMajor, ta, CblasNoTrans, m, jn, k, alpha, A.data(),
                    static_cast<int>(A.rows()), bSeg.data(), k, beta, cPtr,
                    ldc);
    };
    std::vector<std::thread> ths;
    ths.reserve(static_cast<size_t>(nthreads - 1));
    const int64_t base = n / nthreads, extra = n % nthreads;
    int64_t j = 0;
    for (int t = 0; t < nthreads; ++t) {
        const int64_t jn = base + (t < extra ? 1 : 0);
        const int64_t j0 = j, j1 = j + jn;
        j = j1;
        if (t == nthreads - 1)
            runSeg(j0, j1);
        else
            ths.emplace_back(runSeg, j0, j1);
    }
    for (auto& th : ths) th.join();
    (void)sizeof(Dense);
}

/// 列压缩稠密块。
struct ColComp {
    using DenseMatrix = DefaultTraits::DenseMatrix;
    std::vector<int32_t> cols;  // 升序；data 第 j 列 = 逻辑第 cols[j] 列
    DenseMatrix data;           // ident: n×full；否则 n×cols.size()
    int64_t full = 0;           // 逻辑列数
    bool ident = true;          // 恒等映射（逻辑列 j = 物理列 j）

    int64_t rows() const { return data.rows(); }
    int64_t storedCols() const { return ident ? full : static_cast<int64_t>(cols.size()); }
    bool empty() const { return storedCols() == 0; }

    /// 逻辑列 logicalCol 的物理列号（必须在表中；调用方先经掩码/查找保证）。
    int64_t phys(int64_t logicalCol) const {
        if (ident) return logicalCol;
        return lowerBound(static_cast<int32_t>(logicalCol));
    }
    bool has(int64_t logicalCol) const {
        if (ident) return logicalCol < full;
        return std::binary_search(cols.begin(), cols.end(),
                                  static_cast<int32_t>(logicalCol));
    }

    /// 重建为给定升序、无重复的并集列集（保留重叠列数据，新列置零）。
    /// 满表自动退化为 ident。
    void rebuild(std::vector<int32_t> merged) {
        const int64_t r = data.rows();
        DenseMatrix nd =
            DenseMatrix::Zero(r, static_cast<int64_t>(merged.size()));
        int64_t p = 0;
        for (int64_t q = 0; q < static_cast<int64_t>(merged.size()); ++q) {
            while (p < static_cast<int64_t>(cols.size()) &&
                   cols[static_cast<size_t>(p)] < merged[static_cast<size_t>(q)])
                ++p;
            if (p < static_cast<int64_t>(cols.size()) &&
                cols[static_cast<size_t>(p)] == merged[static_cast<size_t>(q)]) {
                nd.col(q) = data.col(p);
                ++p;
            }
        }
        ident = static_cast<int64_t>(merged.size()) == full;
        cols = ident ? std::vector<int32_t>() : std::move(merged);
        data = std::move(nd);
    }

    /// 缩容释放：数据与列表清空（结构表项不动，供运行期按依赖序释放）。
    void shrink() {
        data.resize(0, 0);
        cols.clear();
        ident = false;
    }

private:
    int64_t lowerBound(int32_t logicalCol) const {
        auto it = std::lower_bound(cols.begin(), cols.end(), logicalCol);
        return static_cast<int64_t>(it - cols.begin());
    }
};

/// 列集合位图（填入检查：src ⊆ table 的 O(full/64) 子集测试）。
struct ColMask {
    std::vector<uint64_t> words;
    void init(int64_t n) { words.assign(static_cast<size_t>(n + 63) / 64, 0); }
    void set(int64_t j) { words[static_cast<size_t>(j >> 6)] |= uint64_t(1) << (j & 63); }
    bool get(int64_t j) const {
        return (words[static_cast<size_t>(j >> 6)] >> (j & 63)) & 1;
    }
    /// src ⊆ *this ?
    bool covers(const ColMask& src) const {
        for (size_t i = 0; i < src.words.size(); ++i)
            if (src.words[i] & ~words[i]) return false;
        return true;
    }
    /// 并入 src，返回是否有新增（调用方据此刻触发列插入）。
    bool merge(const ColMask& src) {
        bool grew = false;
        if (words.size() < src.words.size()) words.resize(src.words.size(), 0);
        for (size_t i = 0; i < src.words.size(); ++i) {
            if (src.words[i] & ~words[i]) grew = true;
            words[i] |= src.words[i];
        }
        return grew;
    }
    /// 枚举置位列（升序）。
    std::vector<int32_t> indices() const {
        std::vector<int32_t> idx;
        for (size_t i = 0; i < words.size(); ++i) {
            uint64_t x = words[i];
            while (x) {
                const int b = __builtin_ctzll(x);
                idx.push_back(static_cast<int32_t>(i * 64 + b));
                x &= x - 1;
            }
        }
        return idx;
    }
    int64_t count() const {
        int64_t c = 0;
        for (uint64_t x : words) c += __builtin_popcountll(x);
        return c;
    }
};

namespace blockops {

using Dense  = DefaultTraits::DenseMatrix;
using Sparse = DefaultTraits::SparseMatrix;

/// 值得开任务内并行的规模判据：负载升档后（opsTlsThreads>1）且积足够大。
inline bool parWorthy(int64_t m, int64_t n, int64_t k) {
    const int nt = opsTlsThreads();
    return nt > 1 && n >= 2 * nt && 2.0 * m * n * k >= 2.0e8;
}

// ---- gather / scatter 助手 ----
/// 行 gather：B(r,:) = A(idx[r],:)。
inline Dense gatherRows(const Dense& A, const std::vector<int32_t>& idx) {
    Dense B(static_cast<int64_t>(idx.size()), A.cols());
    for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
        B.row(r) = A.row(idx[static_cast<size_t>(r)]);
    return B;
}
/// 列 gather：B(:,c) = A(:,idx[c])。
inline Dense gatherCols(const Dense& A, const std::vector<int32_t>& idx) {
    Dense B(A.rows(), static_cast<int64_t>(idx.size()));
    for (int64_t c = 0; c < static_cast<int64_t>(idx.size()); ++c)
        B.col(c) = A.col(idx[static_cast<size_t>(c)]);
    return B;
}
/// 稀疏行 gather（dense 结果）。列主序稀疏按列扫描 O(nnz)，
/// 行号经 pos 映射——避免对 ColMajor 稀疏矩阵取行（O(nnz)/行）。
inline Dense gatherRowsSparse(const Sparse& A, const std::vector<int32_t>& idx) {
    Dense B = Dense::Zero(static_cast<int64_t>(idx.size()), A.cols());
    std::vector<int64_t> pos(static_cast<size_t>(A.rows()), -1);
    for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
        pos[static_cast<size_t>(idx[static_cast<size_t>(r)])] = r;
    for (int64_t c = 0; c < A.cols(); ++c)
        for (Sparse::InnerIterator it(A, static_cast<int>(c)); it; ++it) {
            const int64_t p = pos[static_cast<size_t>(it.row())];
            if (p >= 0) B(p, c) = it.value();
        }
    return B;
}
/// keep = {A 的物理列下标 j : A.cols[j] ∈ C.cols}（两升序归并）。
/// 右端零列 ⇒ 积列零增量 ⇒ 目标物理列整列跳过（dgemm n 维缩小）。
inline std::vector<int64_t> intersectColumns(const ColComp& A,
                                             const ColComp& C) {
    std::vector<int64_t> keep;
    const auto& ac = A.cols;
    const auto& cc = C.cols;
    size_t p = 0;
    for (size_t j = 0; j < ac.size(); ++j) {
        while (p < cc.size() && cc[p] < ac[j])
            ++p;
        if (p < cc.size() && cc[p] == ac[j])
            keep.push_back(static_cast<int64_t>(j));
    }
    return keep;
}
/// C 的逻辑列号 logicalCol → C.data 物理列（必须在 C.cols 中）。
inline int64_t physicalColumn(const ColComp& C, int32_t logicalCol) {
    const auto it = std::lower_bound(C.cols.begin(), C.cols.end(), logicalCol);
    return static_cast<int64_t>(it - C.cols.begin());
}
/// Adata 的 phys（升序物理列，A.ident 时即逻辑列）按连续段做
/// Adata(:,seg) += M·R(:,对应段)。R 的列与 phys 一一对应。
/// 大段在负载升档后按列分组并行（等价单次 dgemm）。
inline void segmentScaledAdd(Dense& Adata, const std::vector<int64_t>& phys,
                             const Dense& M, const Dense& R) {
    size_t s = 0;
    const int m = static_cast<int>(Adata.rows());
    const int k = static_cast<int>(M.cols());
    while (s < phys.size()) {
        size_t e = s + 1;
        while (e < phys.size() && phys[e] == phys[e - 1] + 1)
            ++e;
        const int64_t j0 = phys[s];
        const int jn = static_cast<int>(e - s);
        double* cPtr =
            Adata.data() + j0 * static_cast<int64_t>(Adata.rows());
        const double* rPtr = R.data() + s * static_cast<int64_t>(R.rows());
        if (parWorthy(m, jn, k))
            dgemmColsRaw(CblasNoTrans, m, jn, k, 1.0, M.data(),
                         static_cast<int>(M.rows()), rPtr,
                         static_cast<int>(R.rows()), 1.0, cPtr, m,
                         opsTlsThreads());
        else
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, jn, k,
                        1.0, M.data(), static_cast<int>(M.rows()), rPtr,
                        static_cast<int>(R.rows()), 1.0, cPtr, m);
        s = e;
    }
}
/// Adata 的 phys（升序物理列）按连续段 += T 的对应列（T 已完整算出）。
inline void segmentAdd(Dense& Adata, const std::vector<int64_t>& phys,
                       const Dense& T) {
    size_t s = 0;
    while (s < phys.size()) {
        size_t e = s + 1;
        while (e < phys.size() && phys[e] == phys[e - 1] + 1)
            ++e;
        const int64_t j0 = phys[s];
        const int64_t jn = static_cast<int64_t>(e - s);
        Adata.block(0, j0, Adata.rows(), jn) +=
            T.block(0, static_cast<int64_t>(s), Adata.rows(), jn);
        s = e;
    }
}

// ---------------------------------------------------------------------------
// 规范形态一：A += B·C
// ---------------------------------------------------------------------------

/// A += B·C（B/C 列压缩）：N4/N5 第一项（B=耦合块, C=u）与 N10 传播
///（B=μ̃(c,i), C=u(i,a)）共用。积列集 = C.cols，只落在 A.cols ∩ C.cols。
inline void accumulateProduct(ColComp& A, const ColComp& B, const ColComp& C) {
    if (B.empty() || A.empty() || C.empty() || A.data.rows() == 0) return;
    if (A.ident) {
        // A 全宽：更新列 = C.cols（逻辑列号即物理列）；右端行按
        // B.cols gather（B.ident 免 gather，右端即 C.data 全行）
        const Dense Cg = B.ident ? Dense(C.data) : gatherRows(C.data, B.cols);
        std::vector<int64_t> phys;
        phys.reserve(C.cols.size());
        for (int32_t l : C.cols) phys.push_back(static_cast<int64_t>(l));
        segmentScaledAdd(A.data, phys, B.data, Cg);
        return;
    }
    const auto keep = intersectColumns(A, C);
    if (keep.empty()) return;   // 右端零列 ⇒ 无增量
    // Cg(r,t) = C.data(B.cols[r], physC(A.cols[keep[t]]))
    const int64_t cb = B.ident ? B.data.cols()
                               : static_cast<int64_t>(B.cols.size());
    Dense Cg(cb, static_cast<int64_t>(keep.size()));
    for (size_t t = 0; t < keep.size(); ++t) {
        const int64_t pc = physicalColumn(C, A.cols[keep[t]]);
        if (B.ident)
            Cg.col(static_cast<int64_t>(t)) = C.data.col(pc);
        else
            for (int64_t r = 0; r < cb; ++r)
                Cg(r, static_cast<int64_t>(t)) =
                    C.data(B.cols[static_cast<size_t>(r)], pc);
    }
    segmentScaledAdd(A.data, keep, B.data, Cg);
}

/// A += B·C（B 稀疏乘子 = 叶节点原始耦合块；B 的列 = i 全行，右端行免
/// gather）。
inline void accumulateProduct(ColComp& A, const Sparse& B, const ColComp& C) {
    if (A.empty() || C.empty() || A.data.rows() == 0) return;
    if (A.ident) {
        const Dense t = B * C.data;
        for (size_t j = 0; j < C.cols.size(); ++j)
            A.data.col(C.cols[j]) += t.col(static_cast<int64_t>(j));
        return;
    }
    const auto keep = intersectColumns(A, C);
    if (keep.empty()) return;
    Dense Cg(C.data.rows(), static_cast<int64_t>(keep.size()));
    for (size_t t = 0; t < keep.size(); ++t)
        Cg.col(static_cast<int64_t>(t)) =
            C.data.col(physicalColumn(C, A.cols[keep[t]]));
    const Dense t = B * Cg;
    // A.data 的物理列 = keep 本身（j 即物理下标）
    segmentAdd(A.data, keep, t);
}

/// A += B·C（B 列压缩、C 稠密）：求和维 = B 的存储列，驱动 C 的行
/// gather。用于回代恢复 z_i += u(i,a)·z_a 与 N9 折叠 m = μ̃·gathered(Z)。
inline void accumulateProduct(Dense& A, const ColComp& B, const Dense& C) {
    if (B.empty() || A.rows() == 0 || C.cols() == 0) return;
    if (B.ident) {
        if (parWorthy(B.data.rows(), C.cols(), B.data.cols()))
            dgemmColsRaw(CblasNoTrans, static_cast<int>(B.data.rows()),
                         static_cast<int>(C.cols()),
                         static_cast<int>(B.data.cols()), 1.0, B.data.data(),
                         static_cast<int>(B.data.rows()), C.data(),
                         static_cast<int>(C.rows()), 1.0, A.data(),
                         static_cast<int>(A.rows()), opsTlsThreads());
        else
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(B.data.rows()),
                        static_cast<int>(C.cols()),
                        static_cast<int>(B.data.cols()), 1.0, B.data.data(),
                        static_cast<int>(B.data.rows()), C.data(),
                        static_cast<int>(C.rows()), 1.0, A.data(),
                        static_cast<int>(A.rows()));
        return;
    }
    const Dense Cg = gatherRows(C, B.cols);
    if (parWorthy(B.data.rows(), C.cols(), Cg.rows()))
        dgemmColsRaw(CblasNoTrans, static_cast<int>(B.data.rows()),
                     static_cast<int>(C.cols()), static_cast<int>(Cg.rows()),
                     1.0, B.data.data(), static_cast<int>(B.data.rows()),
                     Cg.data(), static_cast<int>(Cg.rows()), 1.0, A.data(),
                     static_cast<int>(A.rows()), opsTlsThreads());
    else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(B.data.rows()), static_cast<int>(C.cols()),
                    static_cast<int>(Cg.rows()), 1.0, B.data.data(),
                    static_cast<int>(B.data.rows()), Cg.data(),
                    static_cast<int>(Cg.rows()), 1.0, A.data(),
                    static_cast<int>(A.rows()));
}

// ---------------------------------------------------------------------------
// 规范形态二：A += Bᵀ·C
// ---------------------------------------------------------------------------

/// A += Bᵀ·C（B/C 列压缩）：N5 第二项 M̄ += uᵀ·Mtemp（唯一的新核）。
/// B/C 均节点行全（行 = 求和维，不压缩）；积的行 = B.cols（A 行不压缩，
/// 直接散布），积的列 = C.cols ∩ A.cols。
inline void accumulateTransposedProduct(ColComp& A, const ColComp& B,
                                        const ColComp& C) {
    if (B.empty() || A.empty() || C.empty() || A.data.rows() == 0) return;
    // 目标列 = A 的存储列中出现在 C.cols 里的部分。
    std::vector<int64_t> aPhys;   // A 物理列
    std::vector<int64_t> cPhys;   // 对应 C 物理列
    if (A.ident) {
        aPhys.reserve(C.cols.size());
        cPhys.reserve(C.cols.size());
        for (size_t j = 0; j < C.cols.size(); ++j) {
            aPhys.push_back(static_cast<int64_t>(C.cols[j]));
            cPhys.push_back(static_cast<int64_t>(j));
        }
    } else {
        const auto& ac = A.cols;
        const auto& cc = C.cols;
        size_t p = 0;
        for (size_t j = 0; j < ac.size(); ++j) {
            while (p < cc.size() && cc[p] < ac[j])
                ++p;
            if (p < cc.size() && cc[p] == ac[j]) {
                aPhys.push_back(static_cast<int64_t>(j));
                cPhys.push_back(static_cast<int64_t>(p));
            }
        }
    }
    if (aPhys.empty()) return;
    // Qg = C.data 的 cPhys 列 gather（n_i × kept）
    Dense Qg(C.data.rows(), static_cast<int64_t>(cPhys.size()));
    for (size_t t = 0; t < cPhys.size(); ++t)
        Qg.col(static_cast<int64_t>(t)) = C.data.col(cPhys[t]);
    // raw = B.dataᵀ·Qg（cb × kept，收缩维 = 节点全行）
    Dense raw;
    const int64_t cb = B.storedCols();
    raw.resize(cb, static_cast<int64_t>(cPhys.size()));
    if (parWorthy(cb, raw.cols(), B.data.rows()))
        dgemmColsRaw(CblasTrans, static_cast<int>(cb),
                     static_cast<int>(raw.cols()),
                     static_cast<int>(B.data.rows()), 1.0, B.data.data(),
                     static_cast<int>(B.data.rows()), Qg.data(),
                     static_cast<int>(Qg.rows()), 0.0, raw.data(),
                     static_cast<int>(raw.rows()), opsTlsThreads());
    else
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    static_cast<int>(cb), static_cast<int>(raw.cols()),
                    static_cast<int>(B.data.rows()), 1.0, B.data.data(),
                    static_cast<int>(B.data.rows()), Qg.data(),
                    static_cast<int>(Qg.rows()), 0.0, raw.data(),
                    static_cast<int>(raw.rows()));
    // 散布：raw 行 r → A.data 行 B.cols[r]（B.ident 时 r 即逻辑行）；
    //       raw 列 t → A.data 物理列 aPhys[t]。
    for (int64_t r = 0; r < cb; ++r) {
        const int64_t rowFull = B.ident ? r : static_cast<int64_t>(
                                        B.cols[static_cast<size_t>(r)]);
        for (size_t t = 0; t < aPhys.size(); ++t)
            A.data(rowFull, aPhys[t]) += raw(r, static_cast<int64_t>(t));
    }
}

/// A += Bᵀ·C（B 稠密、C 列压缩）：N8 种子 μ̃ += Zᵀ·Mtemp。
/// 积的行 = B 的列（A 行不压缩），积的列 = C.cols ∩ A.cols。
inline void accumulateTransposedProduct(ColComp& A, const Dense& B,
                                        const ColComp& C) {
    if (A.empty() || C.empty() || A.data.rows() == 0 || B.cols() == 0) return;
    std::vector<int64_t> aPhys;
    std::vector<int64_t> cPhys;
    if (A.ident) {
        aPhys.reserve(C.cols.size());
        cPhys.reserve(C.cols.size());
        for (size_t j = 0; j < C.cols.size(); ++j) {
            aPhys.push_back(static_cast<int64_t>(C.cols[j]));
            cPhys.push_back(static_cast<int64_t>(j));
        }
    } else {
        const auto& ac = A.cols;
        const auto& cc = C.cols;
        size_t p = 0;
        for (size_t j = 0; j < ac.size(); ++j) {
            while (p < cc.size() && cc[p] < ac[j])
                ++p;
            if (p < cc.size() && cc[p] == ac[j]) {
                aPhys.push_back(static_cast<int64_t>(j));
                cPhys.push_back(static_cast<int64_t>(p));
            }
        }
    }
    if (aPhys.empty()) return;
    Dense Qg(C.data.rows(), static_cast<int64_t>(cPhys.size()));
    for (size_t t = 0; t < cPhys.size(); ++t)
        Qg.col(static_cast<int64_t>(t)) = C.data.col(cPhys[t]);
    Dense raw(B.cols(), static_cast<int64_t>(cPhys.size()));
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(B.cols()), static_cast<int>(raw.cols()),
                static_cast<int>(B.rows()), 1.0, B.data(),
                static_cast<int>(B.rows()), Qg.data(),
                static_cast<int>(Qg.rows()), 0.0, raw.data(),
                static_cast<int>(raw.rows()));
    for (size_t t = 0; t < aPhys.size(); ++t)
        A.data.col(aPhys[t]) += raw.col(static_cast<int64_t>(t));
}

// ---------------------------------------------------------------------------
// 右端物化：Bᵀ 的稠密展开（N1/N3 右端构造）
// ---------------------------------------------------------------------------

/// Bᵀ 的稠密物化（full × B.rows()；压缩列按逻辑列号散布行）。
inline Dense expandTranspose(const ColComp& B) {
    Dense out = Dense::Zero(B.full, B.rows());
    if (B.empty()) return out;
    if (B.ident) return Dense(B.data.transpose());
    for (int64_t j = 0; j < static_cast<int64_t>(B.cols.size()); ++j)
        out.row(B.cols[static_cast<size_t>(j)]) = B.data.col(j);
    return out;
}
inline Dense expandTranspose(const Sparse& B) {
    return Dense(B.transpose());
}

} // namespace blockops
} // namespace amls
