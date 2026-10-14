#include "linalg/col_ops.hpp"

#include <cstring>
#include <utility>

#include <openblas/cblas.h>

#include "core/logging.hpp"
#include "linalg/gemm_cols.hpp"
#include "linalg/lapack.h"

namespace amls {

bool& useColAccGemv() {
    static bool value = true;
    return value;
}

int32_t sytrfFactor(DefaultTraits::DenseMatrix& a,
                    std::vector<int32_t>& ipiv) {
    const int32_t n = static_cast<int32_t>(a.rows());
    ipiv.assign(static_cast<size_t>(std::max(n, 0)), 0);
    if (n == 0) return 0;
    // 先查最优工作区，再实算（netlib dsytrf）
    double sizeQuery = 0;
    int32_t info = 0;
    int32_t lwork = -1;
    const char uplo = 'L';
    dsytrf_(&uplo, &n, a.data(), &n, ipiv.data(), &sizeQuery, &lwork, &info);
    if (info != 0) return info;
    const int32_t workSize = static_cast<int32_t>(sizeQuery);
    std::vector<double> work(static_cast<size_t>(std::max(workSize, n)));
    lwork = static_cast<int32_t>(work.size());
    dsytrf_(&uplo, &n, a.data(), &n, ipiv.data(), work.data(), &lwork, &info);
    return info;
}

int32_t sytrsSolve(int32_t n, const DefaultTraits::DenseMatrix& lu,
                   const std::vector<int32_t>& ipiv, double* b,
                   int32_t nrhs) {
    if (n == 0 || nrhs == 0) return 0;
    const char uplo = 'L';
    int32_t info = 0;
    dsytrs_(&uplo, &n, &nrhs, lu.data(), &n, ipiv.data(), b, &n, &info);
    return info;
}

namespace {

/// 列压缩结构一致性校验（行列数 / 状态数组 / 数据缓冲互相吻合）。
bool colCompShapeOk(const char* caller, const char* name,
                    const ColCompMat& m) {
    const int64_t vc = m.validCols();
    if (!m.numericReady() || m.rows() < 0 || m.totalCols() < 0 ||
        vc > m.totalCols() || m.activeCols() < 0 || m.activeCols() > vc ||
        m.dense().rows() != m.rows() ||
        m.dense().cols() != vc ||
        m.dense().size() !=
            static_cast<int64_t>(m.rows()) * static_cast<int64_t>(vc)) {
        AMLS_LOG_ERROR(
            "[colops] {} 失败：{} 列压缩结构不一致 (rows={}, totalCols={}, "
            "validCols={}, activeCols={}, dense={}x{})。",
            caller, name, m.rows(), m.totalCols(), vc, m.activeCols(),
            m.dense().rows(), m.dense().cols());
        return false;
    }
    // 解码后原始列号严格递增且落在 [0, totalCols)，激活计数与实际一致
    int64_t act = 0;
    for (int64_t s = 0; s < vc; ++s) {
        const int64_t c = m.origColAt(s);
        if (c < 0 || c >= m.totalCols() || (s > 0 && c <= m.origColAt(s - 1))) {
            AMLS_LOG_ERROR(
                "[colops] {} 失败：{} 列状态数组非法（存储列 {} → 原始列 {}）。",
                caller, name, s, c);
            return false;
        }
        if (m.colActive(s)) ++act;
    }
    if (act != m.activeCols()) {
        AMLS_LOG_ERROR(
            "[colops] {} 失败：{} 有效列计数不符（记录 {}，实际 {}）。",
            caller, name, m.activeCols(), act);
        return false;
    }
    return true;
}

/// A 存储列必须覆盖 C 的实际有效列（更新落点可写；占位零列贡献恒 0，
/// 不要求覆盖）——预留列定理的运行时断言。
bool coverageOk(const char* caller, const ColCompMat& A, const ColCompMat& C) {
    for (int64_t sc = 0; sc < C.validCols(); ++sc) {
        if (C.colActive(sc) && A.storageCol(C.origColAt(sc)) < 0) {
            AMLS_LOG_ERROR(
                "[colops] {} 失败：A 存储列未覆盖 C 有效列（原始列 {}）。",
                caller, C.origColAt(sc));
            return false;
        }
    }
    return true;
}

/// 收集 C 实际有效列为连续右端 Cg（rows × nnzC），返回各列的存储列号。
std::vector<int64_t> gatherActiveCols(const ColCompMat& C,
                                      std::vector<double>& cg) {
    const int64_t nnzC = C.activeCols();
    std::vector<int64_t> cStore(static_cast<size_t>(nnzC));
    cg.resize(static_cast<size_t>(C.rows() * nnzC));
    for (int64_t sc = 0, t = 0; sc < C.validCols(); ++sc) {
        if (!C.colActive(sc)) continue;
        cStore[static_cast<size_t>(t)] = sc;
        std::memcpy(cg.data() + static_cast<size_t>(t) * C.rows(),
                    C.dense().col(sc).data(),
                    static_cast<size_t>(C.rows()) * sizeof(double));
        ++t;
    }
    return cStore;
}

/// 乘积列 t 散加到 A 的存储列 od（整列）并激活。
void scatterColumn(ColCompMat& A, int64_t od, const double* prod, int64_t t,
                   int64_t prodRows) {
    double* dst = A.dense().col(A.storageCol(od)).data();
    const double* pc = prod + static_cast<size_t>(t) * prodRows;
    for (int64_t r = 0; r < prodRows; ++r) dst[r] += pc[r];
    A.markColNonZero(od);
}

} // namespace

bool opsColSolve(int64_t n, const DefaultTraits::DenseMatrix& lu,
                 const std::vector<int32_t>& ipiv, const ColCompMat& D,
                 ColCompMat& X) {
    if (n <= 0 || lu.rows() != n || lu.cols() != n ||
        ipiv.size() != static_cast<size_t>(n)) {
        AMLS_LOG_ERROR(
            "[colops] opsColSolve 失败：分解输入非法 (n={}, lu={}x{}, "
            "ipiv={})。",
            n, lu.rows(), lu.cols(), ipiv.size());
        return false;
    }
    if (!colCompShapeOk("opsColSolve", "D", D)) return false;
    if (D.rows() != n) {
        AMLS_LOG_ERROR(
            "[colops] opsColSolve 失败：D 行数 {} 与分解阶数 {} 不符。",
            D.rows(), n);
        return false;
    }
    ColCompMat result;
    if (!result.initializeConstraintFromRhs(D)) return false;
    if (result.validCols() > 0) {
        // 存储列连续，单次多右端回代
        const int32_t info = sytrsSolve(
            static_cast<int32_t>(n), lu, ipiv, result.dense().data(),
            static_cast<int32_t>(result.validCols()));
        if (info != 0) {
            AMLS_LOG_ERROR("[colops] opsColSolve 失败：sytrs info = {}。",
                           static_cast<int>(info));
            return false;
        }
    }
    X = std::move(result);
    return true;
}

bool opsColAccGemv(const ColCompMat& B, bool transB, const ColCompMat& C,
                   ColCompMat& A) {
    if (!colCompShapeOk("opsColAccGemv", "B", B) ||
        !colCompShapeOk("opsColAccGemv", "C", C) ||
        !colCompShapeOk("opsColAccGemv", "A", A))
        return false;
    if (transB) {
        if (B.rows() != C.rows()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemv 失败：transB 要求 B 有 {} 行，实际 "
                "{}。",
                C.rows(), B.rows());
            return false;
        }
        if (A.rows() != B.totalCols()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemv 失败：transB 要求 A 行数 {} 等于 B "
                "总列数 {}。",
                A.rows(), B.totalCols());
            return false;
        }
    } else {
        if (B.totalCols() != C.rows()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemv 失败：非转置要求 B 总列数 {} 等于 C "
                "行数 {}。",
                B.totalCols(), C.rows());
            return false;
        }
        if (A.rows() != B.rows()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemv 失败：非转置要求 A 行数 {} 等于 B "
                "行数 {}。",
                A.rows(), B.rows());
            return false;
        }
    }
    if (A.totalCols() != C.totalCols()) {
        AMLS_LOG_ERROR(
            "[colops] opsColAccGemv 失败：A 总列数 {} 必须等于 C 总列数 {}。",
            A.totalCols(), C.totalCols());
        return false;
    }
    if (!coverageOk("opsColAccGemv", A, C)) return false;
    if (B.activeCols() == 0 || C.activeCols() == 0) return true;

    if (transB) {
        std::vector<double> t(static_cast<size_t>(B.validCols()));
        for (int64_t sc = 0; sc < C.validCols(); ++sc) {
            if (!C.colActive(sc)) continue;
            const int64_t od = C.origColAt(sc);
            // t = Bᵀ·C(:, od)（占位列分量恒 0，散加时只取 B 有效列）
            cblas_dgemv(CblasColMajor, CblasTrans,
                        static_cast<int>(C.rows()),
                        static_cast<int>(B.validCols()), 1.0,
                        B.dense().data(), static_cast<int>(C.rows()),
                        C.dense().col(sc).data(), 1, 0.0, t.data(), 1);
            double* dst = A.dense().col(A.storageCol(od)).data();
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    dst[B.origColAt(sb)] += t[static_cast<size_t>(sb)];
            A.markColNonZero(od);
        }
    } else {
        std::vector<double> xs(static_cast<size_t>(B.validCols()), 0.0);
        std::vector<double> y(static_cast<size_t>(B.rows()));
        for (int64_t sc = 0; sc < C.validCols(); ++sc) {
            if (!C.colActive(sc)) continue;
            const int64_t od = C.origColAt(sc);
            const double* cCol = C.dense().col(sc).data();
            // xs 仅 B 有效列对应位置有值（占位列对应位置恒 0），
            // y = B_compressed · xs
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    xs[static_cast<size_t>(sb)] =
                        cCol[B.origColAt(sb)];
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        static_cast<int>(B.rows()),
                        static_cast<int>(B.validCols()), 1.0,
                        B.dense().data(), static_cast<int>(B.rows()),
                        xs.data(), 1, 0.0, y.data(), 1);
            std::fill(xs.begin(), xs.end(), 0.0);  // 复位供下一列复用
            double* dst = A.dense().col(A.storageCol(od)).data();
            for (int64_t r = 0; r < B.rows(); ++r) dst[r] += y[static_cast<size_t>(r)];
            A.markColNonZero(od);
        }
    }
    return true;
}

bool opsColAccGemm(const ColCompMat& B, bool transB, const ColCompMat& C,
                   ColCompMat& A) {
    if (!colCompShapeOk("opsColAccGemm", "B", B) ||
        !colCompShapeOk("opsColAccGemm", "C", C) ||
        !colCompShapeOk("opsColAccGemm", "A", A))
        return false;
    if (transB) {
        if (B.rows() != C.rows() || A.rows() != B.totalCols()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemm 失败：transB 维度不符 (B.rows={}, "
                "C.rows={}, A.rows={}, B.totalCols={})。",
                B.rows(), C.rows(), A.rows(), B.totalCols());
            return false;
        }
    } else {
        if (B.totalCols() != C.rows() || A.rows() != B.rows()) {
            AMLS_LOG_ERROR(
                "[colops] opsColAccGemm 失败：非转置维度不符 (B.totalCols={}, "
                "C.rows={}, A.rows={}, B.rows={})。",
                B.totalCols(), C.rows(), A.rows(), B.rows());
            return false;
        }
    }
    if (A.totalCols() != C.totalCols()) {
        AMLS_LOG_ERROR(
            "[colops] opsColAccGemm 失败：A 总列数 {} 必须等于 C 总列数 {}。",
            A.totalCols(), C.totalCols());
        return false;
    }
    if (!coverageOk("opsColAccGemm", A, C)) return false;
    if (B.activeCols() == 0 || C.activeCols() == 0) return true;

    const int64_t nnzC = C.activeCols();
    std::vector<double> cg;
    const std::vector<int64_t> cStore = gatherActiveCols(C, cg);
    if (transB) {
        // prod = B_compressedᵀ · Cg：B.validCols × nnzC
        std::vector<double> prod(static_cast<size_t>(B.validCols() * nnzC));
        dgemmColsRaw(CblasTrans, static_cast<int>(B.validCols()),
                     static_cast<int>(nnzC), static_cast<int>(C.rows()), 1.0,
                     B.dense().data(), static_cast<int>(C.rows()), cg.data(),
                     static_cast<int>(C.rows()), 0.0, prod.data(),
                     static_cast<int>(B.validCols()), opsTlsThreads());
        for (int64_t t = 0; t < nnzC; ++t) {
            const int64_t od = C.origColAt(cStore[static_cast<size_t>(t)]);
            const double* pc =
                prod.data() + static_cast<size_t>(t) * B.validCols();
            double* dst = A.dense().col(A.storageCol(od)).data();
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    dst[B.origColAt(sb)] += pc[static_cast<size_t>(sb)];
            A.markColNonZero(od);
        }
    } else {
        // bg 列 = Cg 列按 B 有效列行 gather（B 总列宽 × nnzC，占位列恒 0）
        std::vector<double> bg(static_cast<size_t>(B.validCols() * nnzC), 0.0);
        for (int64_t t = 0; t < nnzC; ++t) {
            const double* cc = cg.data() + static_cast<size_t>(t) * C.rows();
            double* bc = bg.data() + static_cast<size_t>(t) * B.validCols();
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    bc[sb] = cc[B.origColAt(sb)];
        }
        std::vector<double> prod(static_cast<size_t>(B.rows() * nnzC));
        dgemmColsRaw(CblasNoTrans, static_cast<int>(B.rows()),
                     static_cast<int>(nnzC),
                     static_cast<int>(B.validCols()), 1.0, B.dense().data(),
                     static_cast<int>(B.rows()), bg.data(),
                     static_cast<int>(B.validCols()), 0.0, prod.data(),
                     static_cast<int>(B.rows()), opsTlsThreads());
        for (int64_t t = 0; t < nnzC; ++t)
            scatterColumn(A, C.origColAt(cStore[static_cast<size_t>(t)]),
                          prod.data(), t, B.rows());
    }
    return true;
}

namespace {

/// 稠密 B 版公共校验：结构/维度/覆盖性；通过时输出 k=C.rows、m=乘积行数。
bool denseAccShapeOk(const char* op, const DefaultTraits::DenseMatrix& B,
                     bool transB, const ColCompMat& C, const ColCompMat& A,
                     int64_t& k, int64_t& m) {
    if (!colCompShapeOk(op, "C", C) || !colCompShapeOk(op, "A", A))
        return false;
    k = C.rows();
    m = transB ? B.cols() : B.rows();
    if ((transB ? B.rows() : B.cols()) != k) {
        AMLS_LOG_ERROR("[colops] {} 失败：B 内维与 C 行数 {} 不符。", op, k);
        return false;
    }
    if (A.rows() != m || A.totalCols() != C.totalCols()) {
        AMLS_LOG_ERROR(
            "[colops] {} 失败：A 形状 {}x{} 与乘积 {}x{} 不符。",
            op, A.rows(), A.totalCols(), m, C.totalCols());
        return false;
    }
    return coverageOk(op, A, C);
}

} // namespace

bool opsColAccGemv(const DefaultTraits::DenseMatrix& B, bool transB,
                   const ColCompMat& C, ColCompMat& A) {
    int64_t k = 0, m = 0;
    if (!denseAccShapeOk("opsColAccGemv(dense)", B, transB, C, A, k, m))
        return false;
    if (C.activeCols() == 0 || m == 0 || k == 0) return true;
    // 流式：逐 C 有效列 gemv（B 稠密零拷贝），乘积列整列散加到 A
    std::vector<double> t(static_cast<size_t>(m));
    const CBLAS_TRANSPOSE trans = transB ? CblasTrans : CblasNoTrans;
    for (int64_t sc = 0; sc < C.validCols(); ++sc) {
        if (!C.colActive(sc)) continue;
        const int64_t od = C.origColAt(sc);
        if (transB)
            cblas_dgemv(CblasColMajor, trans, static_cast<int>(k),
                        static_cast<int>(m), 1.0, B.data(), static_cast<int>(k),
                        C.dense().col(sc).data(), 1, 0.0, t.data(), 1);
        else
            cblas_dgemv(CblasColMajor, trans, static_cast<int>(m),
                        static_cast<int>(k), 1.0, B.data(), static_cast<int>(m),
                        C.dense().col(sc).data(), 1, 0.0, t.data(), 1);
        scatterColumn(A, od, t.data(), 0, m);
    }
    return true;
}

bool opsColAccGemm(const DefaultTraits::DenseMatrix& B, bool transB,
                   const ColCompMat& C, ColCompMat& A) {
    int64_t k = 0, m = 0;
    if (!denseAccShapeOk("opsColAccGemm(dense)", B, transB, C, A, k, m))
        return false;
    const int64_t nnzC = C.activeCols();
    if (nnzC == 0 || m == 0 || k == 0) return true;
    std::vector<double> cg;
    const std::vector<int64_t> cStore = gatherActiveCols(C, cg);
    std::vector<double> prod(static_cast<size_t>(m * nnzC));
    if (transB)
        dgemmColsRaw(CblasTrans, static_cast<int>(m),
                     static_cast<int>(nnzC), static_cast<int>(k), 1.0,
                     B.data(), static_cast<int>(k), cg.data(),
                     static_cast<int>(k), 0.0, prod.data(),
                     static_cast<int>(m), opsTlsThreads());
    else
        dgemmColsRaw(CblasNoTrans, static_cast<int>(m),
                     static_cast<int>(nnzC), static_cast<int>(k), 1.0,
                     B.data(), static_cast<int>(m), cg.data(),
                     static_cast<int>(k), 0.0, prod.data(),
                     static_cast<int>(m), opsTlsThreads());
    for (int64_t t = 0; t < nnzC; ++t)
        scatterColumn(A, C.origColAt(cStore[static_cast<size_t>(t)]),
                      prod.data(), t, m);
    return true;
}

bool opsColDenseProd(const ColCompMat& A, const DefaultTraits::DenseMatrix& B,
                     DefaultTraits::DenseMatrix& out) {
    if (!colCompShapeOk("opsColDenseProd", "A", A)) return false;
    if (B.rows() != A.totalCols()) {
        AMLS_LOG_ERROR(
            "[colops] opsColDenseProd 失败：B 行数 {} 必须等于 A 总列数 {}。",
            B.rows(), A.totalCols());
        return false;
    }
    const int64_t m = A.rows(), p = B.cols();
    const int64_t nnzA = A.activeCols();
    out = DefaultTraits::DenseMatrix::Zero(m, p);
    if (nnzA == 0 || m == 0 || p == 0) return true;
    // 全列激活：缓冲已是稠密全宽，直接单次 gemm（零拷贝）
    if (nnzA == A.totalCols() && A.validCols() == A.totalCols()) {
        dgemmColsRaw(CblasNoTrans, static_cast<int>(m), static_cast<int>(p),
                     static_cast<int>(nnzA), 1.0, A.dense().data(),
                     static_cast<int>(m), B.data(), static_cast<int>(B.rows()),
                     0.0, out.data(), static_cast<int>(m), opsTlsThreads());
        return true;
    }
    // gather A 有效列 Ag (m × nnzA) 与 B 对应行 Bg (nnzA × p)，单次 gemm
    std::vector<double> ag(static_cast<size_t>(m * nnzA));
    std::vector<double> bg(static_cast<size_t>(nnzA * p), 0.0);
    for (int64_t s = 0, t = 0; s < A.validCols(); ++s) {
        if (!A.colActive(s)) continue;
        std::memcpy(ag.data() + static_cast<size_t>(t) * m,
                    A.dense().col(s).data(),
                    static_cast<size_t>(m) * sizeof(double));
        for (int64_t c = 0; c < p; ++c)
            bg[static_cast<size_t>(c) * nnzA + t] = B(A.origColAt(s), c);
        ++t;
    }
    dgemmColsRaw(CblasNoTrans, static_cast<int>(m), static_cast<int>(p),
                 static_cast<int>(nnzA), 1.0, ag.data(), static_cast<int>(m),
                 bg.data(), static_cast<int>(nnzA), 0.0, out.data(),
                 static_cast<int>(m), opsTlsThreads());
    return true;
}

} // namespace amls
