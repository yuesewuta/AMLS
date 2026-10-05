#pragma once
// 列压缩块（ColComp）与可插拔算子层。
//
// 设计要点（见讨论记录）：
//   * ColComp = { cols(升序局部列号), data, full, ident }。ident 为真时
//     cols 为空、data 即全列稠密（diag 块、近根全耦合块的退化情形），
//     算子对 ident 走原始 BLAS 零间接开销路径。
//   * 列结构静态：表在块物化时建立，之后只读。零→非零的首次填入
//     （保守档）由 GE 侧在 structMutex_ 下做列合并插入，算子层不感知。
//   * 算子集 = gemm / gemv / subMul(A -= B·C) / addTransMul(A += (B·C)ᵀ)
//     / expandT(Bᵀ 物化) —— 插件点：ColCompOps 虚接口，默认 OpenBLAS
//     实现； GE 持有引用，替换后端只需注入另一个实现。
//   * 乘法方向约定：B 为乘子（其 cols 限制求和维并同时驱动右端行 gather），
//     A 为接收块（压缩接收时积只在其存储列上计算，右端列按 A.cols 再收）。
#include <algorithm>
#include <cstdint>
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
    static thread_local int v = 1;
    return v;
}

/// 列分组并行 dgemm：C(:, jseg) = alpha·op(A)·op(B)(:, jseg) + beta·C(:,jseg)。
/// 列主序下 op(B) 的任意列段都是连续内存（tb=NoTrans 取 B 的列段，
/// tb=Trans 取 B 的行段 = 其列主序列），各线程写不相交列段，无竞争、
/// 无锁、不依赖 BLAS 内部线程（规避 libgomp 嵌套并行区风险）。
/// 纯内联 dgemm 调用级拆分，对调用方完全等价于单次 dgemm。
inline void dgemmCols(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const DefaultTraits::DenseMatrix& A,
                      const DefaultTraits::DenseMatrix& B, double beta,
                      DefaultTraits::DenseMatrix& C, int nthreads) {
    using Dense = DefaultTraits::DenseMatrix;
    const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
    const int k = static_cast<int>((ta == CblasNoTrans) ? A.cols() : A.rows());
    const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
    if (n <= 0 || m <= 0) return;
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
    // 列段均分（余数摊给前几个线程）。
    // tb=NoTrans: op(B) 列段 = B 的列 j0..（列主序连续，指针偏移即可）
    // tb=Trans:   op(B) 列段 = B 的行 j0..（列主序下每列是 B 的一行、
    //             行内步长 ldb 不连续）→ 线程内先物化成连续 k×jn 段
    //             C 段 = Aᵀ·Bseg，即 dgemm(ta, NoTrans)。
    auto runSeg = [&](int j0, int j1) {
        if (j1 <= j0) return;
        const int jn = j1 - j0;
        blas::pinThisThread();   // 临时线程：OpenBLAS(OMP 版) ICV 钉 1，
                                 // 否则每线程内 dgemm 再开满核 → 过度订阅
        if (tb == CblasNoTrans) {
            const double* bPtr = B.data() + static_cast<int64_t>(j0) * ldb;
            double* cPtr = C.data() + static_cast<int64_t>(j0) * ldc;
            cblas_dgemm(CblasColMajor, ta, tb, m, jn, k, alpha, A.data(),
                        static_cast<int>(A.rows()), bPtr, ldb, beta, cPtr,
                        ldc);
            return;
        }
        // tb=Trans: op(B)=Bᵀ，op(B)(r, j0+c) = B(j0+c, r)（B 的第 j0+c 行），
        // 列主序下行不连续 → 物化时行列对调：bSeg(r,c) = B(j0+c, r)。
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

    /// 逻辑列 l 的物理列号（l 必须在表中；调用方先经掩码/查找保证）。
    int64_t phys(int64_t l) const {
        if (ident) return l;
        return lowerBound(l);
    }
    bool has(int64_t l) const {
        if (ident) return l < full;
        return std::binary_search(cols.begin(), cols.end(),
                                  static_cast<int32_t>(l));
    }

    /// 重建为给定升序、无重复的并集列集（保留重叠列数据，新列置零）。
    /// GE 保守档插入用。满表自动退化为 ident。
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

private:
    int64_t lowerBound(int32_t l) const {
        auto it = std::lower_bound(cols.begin(), cols.end(), l);
        return static_cast<int64_t>(it - cols.begin());
    }
};

/// 列集合位图（填入检查：src ⊆ table 的 O(full/64) 子集测试）。
struct ColMask {
    std::vector<uint64_t> w;
    void init(int64_t n) { w.assign(static_cast<size_t>(n + 63) / 64, 0); }
    void set(int64_t j) { w[static_cast<size_t>(j >> 6)] |= uint64_t(1) << (j & 63); }
    bool get(int64_t j) const {
        return (w[static_cast<size_t>(j >> 6)] >> (j & 63)) & 1;
    }
    /// src ⊆ *this ?
    bool covers(const ColMask& src) const {
        for (size_t i = 0; i < src.w.size(); ++i)
            if (src.w[i] & ~w[i]) return false;
        return true;
    }
    /// 并入 src，返回是否有新增（调用方据此刻触发列插入）。
    bool merge(const ColMask& src) {
        bool grew = false;
        if (w.size() < src.w.size()) w.resize(src.w.size(), 0);
        for (size_t i = 0; i < src.w.size(); ++i) {
            if (src.w[i] & ~w[i]) grew = true;
            w[i] |= src.w[i];
        }
        return grew;
    }
    /// 枚举置位列（升序）。
    std::vector<int32_t> indices() const {
        std::vector<int32_t> idx;
        for (size_t i = 0; i < w.size(); ++i) {
            uint64_t x = w[i];
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
        for (uint64_t x : w) c += __builtin_popcountll(x);
        return c;
    }
};

// ---------------------------------------------------------------------------
// 算子插件接口（默认实现见下 OpenBlasColCompOps）
// ---------------------------------------------------------------------------
class ColCompOps {
public:
    using Dense  = DefaultTraits::DenseMatrix;
    using Sparse = DefaultTraits::SparseMatrix;
    using Vector = DefaultTraits::Vector;

    virtual ~ColCompOps() = default;

    /// C = alpha·op(A)·op(B) + beta·C（A 可为列压缩；B/C 稠密）。
    /// ta=CblasTrans 时 C 按 A 的存储行空间解释（C.rows()=A.storedCols()，
    /// C(j,:)=Σ_r data(r,j)·B(r,:)）——回代恢复使用此约定。
    virtual void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const ColComp& A, const Dense& B, double beta,
                      Dense& C) const = 0;
    virtual void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const Dense& A, const Dense& B, double beta,
                      Dense& C) const = 0;

    /// y = alpha·op(A)·x + beta·y。
    virtual void gemv(CBLAS_TRANSPOSE ta, double alpha, const ColComp& A,
                      const Vector& x, double beta, Vector& y) const = 0;
    virtual void gemv(CBLAS_TRANSPOSE ta, double alpha, const Dense& A,
                      const Vector& x, double beta, Vector& y) const = 0;

    /// A -= B·C。乘子 B 的列限制求和维并驱动 C 的行 gather；
    /// 压缩接收时积只在 A 的存储列上计算（C 按 A.cols 二次收列）。
    virtual void subMul(Dense& A, const ColComp& B, const Dense& C) const = 0;
    virtual void subMul(ColComp& A, const ColComp& B, const Dense& C) const = 0;
    virtual void subMul(Dense& A, const Sparse& B, const Dense& C) const = 0;
    virtual void subMul(ColComp& A, const Sparse& B, const Dense& C) const = 0;
    /// A += B·C（同余三重积 −K·W·X 项的相反号路径）。
    virtual void addMul(ColComp& A, const ColComp& B, const Dense& C) const = 0;
    virtual void addMul(ColComp& A, const Sparse& B, const Dense& C) const = 0;

    /// A ±= (B·C)ᵀ（同余转置项；B 为 lo 侧乘子，结果限制在 A 的存储列）。
    virtual void subTransMul(ColComp& A, const ColComp& B, const Dense& C) const = 0;
    virtual void subTransMul(ColComp& A, const Sparse& B, const Dense& C) const = 0;
    virtual void addTransMul(ColComp& A, const ColComp& B, const Dense& C) const = 0;
    virtual void addTransMul(ColComp& A, const Sparse& B, const Dense& C) const = 0;

    /// B·C 的稠密物化（同余对角项中间量 t / t2）。
    virtual Dense mul(const ColComp& B, const Dense& C) const = 0;
    virtual Dense mul(const Sparse& B, const Dense& C) const = 0;

    /// Bᵀ 的稠密物化（回代右端；K_{i,a} = (a,i) 块的转置散布）。
    virtual Dense expandT(const ColComp& B) const = 0;
    virtual Dense expandT(const Sparse& B) const = 0;
};

/// 默认后端：OpenBLAS dgemm/dgemv + Eigen 稀疏乘。
class OpenBlasColCompOps final : public ColCompOps {
public:
    void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
              const ColComp& A, const Dense& B, double beta,
              Dense& C) const override {
        const int m = static_cast<int>((ta == CblasNoTrans) ? A.data.rows()
                                                            : (A.ident ? A.data.cols() : A.storedCols()));
        const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
        if (C.rows() != m || C.cols() != n) C.resize(m, n);
        if (!A.ident && A.cols.empty()) {   // 空表乘子：C = beta·C
            if (beta == 0.0) C.setZero();
            else C *= beta;
            return;
        }
        if (A.ident) {
            gemm(ta, tb, alpha, static_cast<const Dense&>(A.data), B, beta, C);
            return;
        }
        // op(A) 的列 = B 的行须经 A.cols 映射
        if (ta == CblasNoTrans) {
            const Dense Bg = rows(B, A.cols);          // (c×n)
            dgemm(CblasNoTrans, tb, alpha, A.data, Bg, beta, C);
        } else {
            // 存储行空间契约：C 已按 c×n 尺寸传入（见接口注释）
            dgemm(CblasTrans, tb, alpha, A.data, B, beta, C);
        }
    }
    void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
              const Dense& A, const Dense& B, double beta,
              Dense& C) const override {
        const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
        const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
        if (C.rows() != m || C.cols() != n) C.resize(m, n);
        dgemm(ta, tb, alpha, A, B, beta, C);
    }

    void gemv(CBLAS_TRANSPOSE ta, double alpha, const ColComp& A,
              const Vector& x, double beta, Vector& y) const override {
        if (!A.ident && A.cols.empty()) {   // 空表：y = beta·y
            if (beta == 0.0) y.setZero();
            else y *= beta;
            return;
        }
        if (A.ident) {
            gemv(ta, alpha, static_cast<const Dense&>(A.data), x, beta, y);
            return;
        }
        if (ta == CblasNoTrans) {
            const Vector xg = gather(x, A.cols);
            dgemv(CblasNoTrans, alpha, A.data, xg, beta, y);
        } else {
            Vector yt = Vector::Zero(A.storedCols());
            dgemv(CblasTrans, alpha, A.data, x, 0.0, yt);
            scatterAdd(y, A.cols, yt, beta);
        }
    }
    void gemv(CBLAS_TRANSPOSE ta, double alpha, const Dense& A,
              const Vector& x, double beta, Vector& y) const override {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        cblas_dgemv(CblasColMajor, ta, m, n, alpha, A.data(), m, x.data(), 1,
                    beta, y.data(), 1);
    }

    void subMul(Dense& A, const ColComp& B, const Dense& C) const override {
        if (B.empty()) return;
        if (B.ident) {
            dgemm(CblasNoTrans, CblasNoTrans, -1.0, B.data, C, 1.0, A);
            return;
        }
        const Dense Cg = rows(C, B.cols);   // (c_b×n)
        dgemm(CblasNoTrans, CblasNoTrans, -1.0, B.data, Cg, 1.0, A);
    }
    void subMul(ColComp& A, const ColComp& B, const Dense& C) const override {
        if (B.empty() || A.empty()) return;
        if (A.ident) { subMul(static_cast<Dense&>(A.data), B, C); return; }
        // B.ident 时存储即全列：右端取 C 的 A.cols 列（m×c_a），
        // 不得按空表 gather（k=0 会静默丢项）
        const Dense Cg = B.ident ? colsOf(C, A.cols) : sub(C, B.cols, A.cols);
        dgemm(CblasNoTrans, CblasNoTrans, -1.0, B.data, Cg, 1.0, A.data);
    }
    void subMul(Dense& A, const Sparse& B, const Dense& C) const override {
        A.noalias() -= B * C;
    }
    void subMul(ColComp& A, const Sparse& B, const Dense& C) const override {
        if (A.empty()) return;
        if (A.ident) { A.data.noalias() -= B * C; return; }
        const Dense Cg = colsOf(C, A.cols);   // (n_i×c_a)
        A.data.noalias() -= B * Cg;
    }

    void addMul(ColComp& A, const ColComp& B, const Dense& C) const override {
        accMul(A, B, C, +1.0, false);
    }
    void addMul(ColComp& A, const Sparse& B, const Dense& C) const override {
        if (A.empty()) return;
        if (A.ident) { A.data.noalias() += B * C; return; }
        const Dense Cg = colsOf(C, A.cols);
        A.data.noalias() += B * Cg;
    }

    void subTransMul(ColComp& A, const ColComp& B, const Dense& C) const override {
        accMul(A, B, C, -1.0, true);
    }
    void subTransMul(ColComp& A, const Sparse& B, const Dense& C) const override {
        accTransMulSparse(A, B, C, -1.0);
    }
    void addTransMul(ColComp& A, const ColComp& B, const Dense& C) const override {
        accMul(A, B, C, +1.0, true);
    }
    void addTransMul(ColComp& A, const Sparse& B, const Dense& C) const override {
        accTransMulSparse(A, B, C, +1.0);
    }

    Dense mul(const ColComp& B, const Dense& C) const override {
        Dense out;
        gemm(CblasNoTrans, CblasNoTrans, 1.0, B, C, 0.0, out);
        return out;
    }
    Dense mul(const Sparse& B, const Dense& C) const override {
        return Dense(B * C);
    }

    Dense expandT(const ColComp& B) const override {
        Dense out = Dense::Zero(B.full, B.rows());
        if (B.empty()) return out;
        if (B.ident) return Dense(B.data.transpose());
        for (int64_t j = 0; j < static_cast<int64_t>(B.cols.size()); ++j)
            out.row(B.cols[static_cast<size_t>(j)]) = B.data.col(j);
        return out;
    }
    Dense expandT(const Sparse& B) const override {
        return Dense(B.transpose());
    }

    // ---- 内部工具 ----
    /// subMul/addMul/subTransMul/addTransMul 的公共实现（trans 选择转置项）。
    /// 转置项统一用单次 dgemm(ta=Trans,tb=Trans) 直写目标：
    ///   A ±= (B·C)ᵀ = op(Cg)ᵀ·op(S)ᵀ，无中间乘积、无显式转置拷贝。
    /// 负载自适应：DAG 饥饿升档后 opsTlsThreads()>1，主 dgemm 按目标列
    /// 分组用临时线程并行（各线程写不相交列段，等价于单次 dgemm）。
    static bool parWorthy(int64_t m, int64_t n, int64_t k) {
        const int nt = opsTlsThreads();
        return nt > 1 && n >= 2 * nt && 2.0 * m * n * k >= 2.0e8;
    }
    static void accMul(ColComp& A, const ColComp& B, const Dense& C,
                       double sign, bool trans) {
        if (B.empty() || A.empty()) return;
        if (!trans) {
            if (A.ident) {
                const Dense Cg = B.ident ? Dense(C) : rows(C, B.cols);
                if (parWorthy(B.data.rows(), A.data.cols(), Cg.rows()))
                    dgemmCols(CblasNoTrans, CblasNoTrans, sign, B.data, Cg,
                              1.0, A.data, opsTlsThreads());
                else
                    dgemm(CblasNoTrans, CblasNoTrans, sign, B.data, Cg, 1.0,
                          A.data);
                return;
            }
            const Dense Cg =
                B.ident ? colsOf(C, A.cols) : sub(C, B.cols, A.cols);  // (c_b×c_a)
            if (parWorthy(B.data.rows(), A.data.cols(), Cg.rows()))
                dgemmCols(CblasNoTrans, CblasNoTrans, sign, B.data, Cg, 1.0,
                          A.data, opsTlsThreads());
            else
                dgemm(CblasNoTrans, CblasNoTrans, sign, B.data, Cg, 1.0,
                      A.data);
            return;
        }
        // 转置项：A ±= (B·C)ᵀ。C 经 B.cols 行 gather（c_b×n_hi）；
        // B.ident 时存储即全列，不得按空表 gather。
        const Dense Cg = B.ident ? Dense(C) : rows(C, B.cols);
        if (A.ident) {
            // 全尺寸：(B.data·Cg)ᵀ = Cgᵀ·B.dataᵀ，单次 dgemm 直写 A.data
            if (parWorthy(Cg.cols(), A.data.cols(), Cg.rows()))
                dgemmCols(CblasTrans, CblasTrans, sign, Cg, B.data, 1.0,
                          A.data, opsTlsThreads());
            else
                dgemm(CblasTrans, CblasTrans, sign, Cg, B.data, 1.0, A.data);
            return;
        }
        // 压缩接收：列限制在 A.cols。S = B.data(A.cols, B.cols)（c_a×c_b），
        // (B·C)ᵀ(:,A.cols) = Cgᵀ·Sᵀ；B.ident 时 S 为 B.data 的 A.cols 行。
        // 并行路径：S 按 A.cols 段分组，每线程独立 sub-gather + dgemm。
        if (parWorthy(Cg.cols(), A.data.cols(), Cg.rows())) {
            static const bool forceSerial =
                std::getenv("AMLS_DGEMM_SERIAL") != nullptr;
            const int nt = forceSerial ? 1 : std::min<int64_t>(
                             opsTlsThreads(),
                             (A.data.cols() + 31) / 32);
            Dense& Ad = A.data;
            const auto& aCols = A.cols;
            const auto& bCols = B.cols;
            const bool bIdent = B.ident;
            const Dense& Bd = B.data;
            auto runSeg = [&](int64_t j0, int64_t j1) {
                if (j1 <= j0) return;
                blas::pinThisThread();   // 同上：临时线程钉 BLAS 单线程
                std::vector<int32_t> seg(aCols.begin() + j0,
                                         aCols.begin() + j1);
                const Dense S = bIdent ? rows(Bd, seg)
                                       : sub(Bd, seg, bCols);
                double* cPtr = Ad.data() + j0 * static_cast<int64_t>(Ad.rows());
                cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                            static_cast<int>(Cg.cols()),
                            static_cast<int>(j1 - j0),
                            static_cast<int>(Cg.rows()), sign, Cg.data(),
                            static_cast<int>(Cg.rows()), S.data(),
                            static_cast<int>(S.rows()), 1.0, cPtr,
                            static_cast<int>(Ad.rows()));
            };
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(nt > 0 ? nt - 1 : 0));
            const int64_t n = static_cast<int64_t>(aCols.size());
            const int64_t base = n / nt, extra = n % nt;
            int64_t j = 0;
            for (int t = 0; t < nt; ++t) {
                const int64_t jn = base + (t < extra ? 1 : 0);
                const int64_t j0 = j, j1 = j + jn;
                j = j1;
                if (t == nt - 1)
                    runSeg(j0, j1);
                else
                    ths.emplace_back(runSeg, j0, j1);
            }
            for (auto& th : ths) th.join();
            return;
        }
        const Dense S =
            B.ident ? rows(B.data, A.cols) : sub(B.data, A.cols, B.cols);
        dgemm(CblasTrans, CblasTrans, sign, Cg, S, 1.0, A.data);
    }
    static void accTransMulSparse(ColComp& A, const Sparse& B, const Dense& C,
                                  double sign) {
        if (A.empty()) return;
        if (A.ident) {
            const Dense s = B * C;
            if (sign > 0) A.data.noalias() += s.transpose();
            else A.data.noalias() -= s.transpose();
            return;
        }
        const Dense Br = rowsSparse(B, A.cols);  // (c_a×n_i)
        // A ±= (B·C)ᵀ(:,A.cols) = ± Cᵀ·Brᵀ：单次 dgemm 直写，无转置拷贝
        dgemm(CblasTrans, CblasTrans, sign, C, Br, 1.0, A.data);
    }
    static void dgemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, double alpha,
                      const Dense& A, const Dense& B, double beta, Dense& C) {
        const int m = static_cast<int>((ta == CblasNoTrans) ? A.rows() : A.cols());
        const int k = static_cast<int>((ta == CblasNoTrans) ? A.cols() : A.rows());
        const int n = static_cast<int>((tb == CblasNoTrans) ? B.cols() : B.rows());
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A.data(),
                    static_cast<int>(A.rows()), B.data(),
                    static_cast<int>(B.rows()), beta, C.data(),
                    static_cast<int>(C.rows()));
    }
    static void dgemv(CBLAS_TRANSPOSE ta, double alpha, const Dense& A,
                      const Vector& x, double beta, Vector& y) {
        cblas_dgemv(CblasColMajor, ta, static_cast<int>(A.rows()),
                    static_cast<int>(A.cols()), alpha, A.data(),
                    static_cast<int>(A.rows()), x.data(), 1, beta, y.data(), 1);
    }
    /// 行 gather：B(r,:) = A(idx[r],:)。
    static Dense rows(const Dense& A, const std::vector<int32_t>& idx) {
        Dense B(static_cast<int64_t>(idx.size()), A.cols());
        for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
            B.row(r) = A.row(idx[static_cast<size_t>(r)]);
        return B;
    }
    /// 列 gather：B(:,c) = A(:,idx[c])。
    static Dense colsOf(const Dense& A, const std::vector<int32_t>& idx) {
        Dense B(A.rows(), static_cast<int64_t>(idx.size()));
        for (int64_t c = 0; c < static_cast<int64_t>(idx.size()); ++c)
            B.col(c) = A.col(idx[static_cast<size_t>(c)]);
        return B;
    }
    /// 行列双 gather：B(r,c) = A(idxR[r], idxC[c])。
    static Dense sub(const Dense& A, const std::vector<int32_t>& idxR,
                     const std::vector<int32_t>& idxC) {
        Dense B(static_cast<int64_t>(idxR.size()), static_cast<int64_t>(idxC.size()));
        for (int64_t c = 0; c < static_cast<int64_t>(idxC.size()); ++c) {
            const auto col = A.col(idxC[static_cast<size_t>(c)]);
            for (int64_t r = 0; r < static_cast<int64_t>(idxR.size()); ++r)
                B(r, c) = col(idxR[static_cast<size_t>(r)]);
        }
        return B;
    }
    /// 稀疏行 gather（dense 结果）。列主序稀疏按列扫描 O(nnz)，
    /// 行号经 pos 映射——避免对 ColMajor 稀疏矩阵取行（O(nnz)/行）。
    static Dense rowsSparse(const Sparse& A, const std::vector<int32_t>& idx) {
        Dense B = Dense::Zero(static_cast<int64_t>(idx.size()), A.cols());
        std::vector<int64_t> pos(static_cast<size_t>(A.rows()), -1);
        for (int64_t r = 0; r < static_cast<int64_t>(idx.size()); ++r)
            pos[static_cast<size_t>(idx[static_cast<size_t>(r)])] = r;
        for (int64_t c = 0; c < A.cols(); ++c)
            for (typename Sparse::InnerIterator it(A, static_cast<int>(c)); it;
                 ++it) {
                const int64_t p = pos[static_cast<size_t>(it.row())];
                if (p >= 0) B(p, c) = it.value();
            }
        return B;
    }
    static Vector gather(const Vector& x, const std::vector<int32_t>& idx) {
        Vector g(static_cast<int64_t>(idx.size()));
        for (int64_t j = 0; j < static_cast<int64_t>(idx.size()); ++j)
            g[j] = x[idx[static_cast<size_t>(j)]];
        return g;
    }
    static void scatterAdd(Vector& y, const std::vector<int32_t>& idx,
                           const Vector& g, double beta) {
        for (int64_t j = 0; j < y.size(); ++j) y[j] *= beta;
        for (int64_t j = 0; j < static_cast<int64_t>(idx.size()); ++j)
            y[idx[static_cast<size_t>(j)]] += g[j];
    }
};

/// 全局默认算子实例（OpenBLAS 后端）。
inline const ColCompOps& defaultColCompOps() {
    static const OpenBlasColCompOps inst;
    return inst;
}

} // namespace amls
