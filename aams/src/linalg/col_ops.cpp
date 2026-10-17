#include "linalg/col_ops.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <openblas/cblas.h>

#include "core/logging.hpp"
#include "linalg/gemm_cols.hpp"
#include "linalg/lapack.h"

namespace aams {

bool& useColAccGemv() {
    static bool value = std::getenv("AAMS_COLACC_GEMV") == nullptr ||
                        std::getenv("AAMS_COLACC_GEMV")[0] != '0';
    return value;
}

// ---- 性能剖析计数器（AAMS_PROFILE=1 时经 colOpsDumpStats 输出）----
// gemv/gemm 两条 opsColAcc 路径的调用数、累计耗时、累计 flops、累计
// 临时分配字节数。原子量，多线程安全；剖析关闭时开销为几次 relaxed 加法。
namespace {
struct OpAccStats {
    std::atomic<uint64_t> gemvCalls{0}, gemvNs{0}, gemvFlops{0};
    std::atomic<uint64_t> gemmCalls{0}, gemmNs{0}, gemmFlops{0};
    std::atomic<uint64_t> tmpAllocBytes{0};
    std::atomic<uint64_t> hist[5] = {{0},{0},{0},{0},{0}};
    std::atomic<uint64_t> histNs[5] = {{0},{0},{0},{0},{0}};
    std::atomic<uint64_t> bElems{0};
};
OpAccStats& opAccStats() {
    static OpAccStats s;
    return s;
}
// 剖析门控：默认关（热路径零原子开销），AAMS_PROFILE=1 才记录。
inline bool opAccEnabled() {
    static const bool on = std::getenv("AAMS_PROFILE") != nullptr;
    return on;
}
}  // namespace

void colOpsDumpStats() {
    if (std::getenv("AAMS_PROFILE") == nullptr) return;
    auto& st = opAccStats();
    std::fprintf(stderr,
                 "[prof] colAcc gemv: calls=%llu flops=%.3e t=%.3fs\n"
                 "[prof] colAcc gemm: calls=%llu flops=%.3e t=%.3fs\n"
                 "[prof] colAcc tmp-alloc=%.2f GB\n"
                 "[prof] colAcc flops/call hist: <100:%llu <1k:%llu <10k:%llu <100k:%llu >=100k:%llu | Belems=%.3e\n",
                 (unsigned long long)st.gemvCalls.load(std::memory_order_relaxed),
                 static_cast<double>(st.gemvFlops.load(std::memory_order_relaxed)),
                 static_cast<double>(st.gemvNs.load(std::memory_order_relaxed)) * 1e-9,
                 (unsigned long long)st.gemmCalls.load(std::memory_order_relaxed),
                 static_cast<double>(st.gemmFlops.load(std::memory_order_relaxed)),
                 static_cast<double>(st.gemmNs.load(std::memory_order_relaxed)) * 1e-9,
                 static_cast<double>(st.tmpAllocBytes.load(std::memory_order_relaxed)) / 1073741824.0,
                 (unsigned long long)st.hist[0].load(std::memory_order_relaxed),
                 (unsigned long long)st.hist[1].load(std::memory_order_relaxed),
                 (unsigned long long)st.hist[2].load(std::memory_order_relaxed),
                 (unsigned long long)st.hist[3].load(std::memory_order_relaxed),
                 (unsigned long long)st.hist[4].load(std::memory_order_relaxed),
                 static_cast<double>(st.bElems.load(std::memory_order_relaxed)));
    std::fprintf(stderr,
                 "[prof] colAcc gemv t/hist (s): <100:%.3f <1k:%.3f <10k:%.3f <100k:%.3f >=100k:%.3f\n",
                 static_cast<double>(st.histNs[0].load(std::memory_order_relaxed)) * 1e-9,
                 static_cast<double>(st.histNs[1].load(std::memory_order_relaxed)) * 1e-9,
                 static_cast<double>(st.histNs[2].load(std::memory_order_relaxed)) * 1e-9,
                 static_cast<double>(st.histNs[3].load(std::memory_order_relaxed)) * 1e-9,
                 static_cast<double>(st.histNs[4].load(std::memory_order_relaxed)) * 1e-9);
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

/// 列压缩结构一致性校验。O(1) 快检每次调用都走；O(validCols) 深检
/// （原列号升序、激活计数一致）只在 AAMS_COLCOMP_VALIDATE=1 时做——
/// 4.58M 次/轮的热调用里全列扫描曾占每调用 ~30µs，是 REDUCTION 最大
/// 单项开销，属调试级检查不该上热路径。
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
    static const bool deep = std::getenv("AAMS_COLCOMP_VALIDATE") != nullptr;
    if (!deep) return true;
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
/// 令 A(:,od) 原地膨胀）。同样按 env 门控深扫；散加路径自身对
/// storageCol<0 仍有兜底检查。
bool coverageOk(const char* caller, const ColCompMat& A, const ColCompMat& C) {
    static const bool deep = std::getenv("AAMS_COLCOMP_VALIDATE") != nullptr;
    if (!deep) return true;
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

/// 累加落点列指针（storageCol<0 时报错返回空——覆盖性检查的散加端兜底）。
double* accDstCol(const char* caller, ColCompMat& A, int64_t od) {
    const int64_t sc = A.storageCol(od);
    if (sc < 0) {
        AMLS_LOG_ERROR(
            "[colops] {} 失败：A 存储列未覆盖原始列 {}。", caller, od);
        return nullptr;
    }
    return A.dense().col(sc).data();
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
    double* dst = accDstCol("scatterColumn", A, od);
    if (!dst) return;
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
    auto& stGV = opAccStats();
    const auto tGV = std::chrono::steady_clock::now();
    struct GemvRec {
        decltype(tGV) t0;
        int b = -1;  // flops 桶号（实算调用才 >=0）
        ~GemvRec() {
            if (!opAccEnabled()) return;
            const auto ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());
            auto& st = opAccStats();
            st.gemvCalls.fetch_add(1, std::memory_order_relaxed);
            st.gemvNs.fetch_add(ns, std::memory_order_relaxed);
            if (b >= 0)
                st.histNs[b].fetch_add(ns, std::memory_order_relaxed);
        }
    } gemvRec{tGV};
    // 空块短路放最前：shapeOk/coverageOk 是全列扫描，空调用占比 ~69%，
    // 先付校验再判空等于让多数调用纯交学费。
    if (B.activeCols() == 0 || C.activeCols() == 0) return true;
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
    if (opAccEnabled()) {
        stGV.gemvFlops.fetch_add(
            static_cast<uint64_t>(2.0 * static_cast<double>(B.rows()) *
                                  static_cast<double>(B.validCols()) *
                                  static_cast<double>(C.activeCols())),
            std::memory_order_relaxed);
        const double fl = 2.0 * static_cast<double>(B.rows()) *
                          static_cast<double>(B.validCols()) *
                          static_cast<double>(C.activeCols());
        const int b = fl < 100.0 ? 0 : fl < 1e3 ? 1 : fl < 1e4 ? 2
                    : fl < 1e5 ? 3 : 4;
        gemvRec.b = b;
        stGV.hist[b].fetch_add(1, std::memory_order_relaxed);
        stGV.bElems.fetch_add(
            static_cast<uint64_t>(static_cast<double>(B.rows()) *
                                  static_cast<double>(B.validCols())),
            std::memory_order_relaxed);
    }

    // 路径选择：
    //  - acC 小：逐列 gemv（每列 ~600ns 调用开销，列少时最便宜）
    //  - acC 大：分块 gemm（w=32 列一块）——每列一次 dgemv 的调用开销
    //    ×50 列 ≈ 30µs/调用；但 OpenBLAS 小 dgemm 驱动开销 ~5–10µs，
    //    B 太小时反亏，故要求 B 有一定规模（实测 4096 最优）。
    //    AAMS_NO_FUSED=1 强制逐列（A/B 计时用）。
    const int64_t acC = C.activeCols();
    const int64_t rowsAcc = transB ? C.rows() : B.rows();
    const int64_t vbAcc = B.validCols();
    const bool useChunk = std::getenv("AAMS_NO_FUSED") == nullptr &&
                          acC >= 8 && rowsAcc * vbAcc >= 4096;
    if (useChunk) {
        static thread_local std::vector<double> tCg, tProd, tCc;
        constexpr int64_t w = 32;
        const std::vector<int64_t> cStore = gatherActiveCols(C, tCg);
        const int64_t crows = C.rows();
        tProd.resize(static_cast<size_t>(
            (transB ? vbAcc : rowsAcc) * w));
        if (!transB) tCc.resize(static_cast<size_t>(vbAcc * w));
        for (int64_t t0 = 0; t0 < acC; t0 += w) {
            const int64_t nw = std::min<int64_t>(w, acC - t0);
            if (transB) {
                // prod(vb×nw) = B_denseᵀ · Cg 块
                dgemmColsRaw(CblasTrans, static_cast<int>(vbAcc),
                             static_cast<int>(nw), static_cast<int>(crows),
                             1.0, B.dense().data(), static_cast<int>(crows),
                             tCg.data() + static_cast<size_t>(t0) * crows,
                             static_cast<int>(crows), 0.0, tProd.data(),
                             static_cast<int>(vbAcc), opsTlsThreads());
            } else {
                // Cc(sb,t) = Cg(origColAt(sb), t0+t)：占位列置 0，
                // cg 行序 = B 原始列序 → 顺序读、散射写
                std::fill_n(tCc.data(), static_cast<size_t>(vbAcc * nw), 0.0);
                for (int64_t t = 0; t < nw; ++t) {
                    const double* cc = tCg.data() +
                        static_cast<size_t>(t0 + t) * crows;
                    double* bcol = tCc.data() + static_cast<size_t>(t) * vbAcc;
                    for (int64_t sb = 0; sb < vbAcc; ++sb)
                        if (B.colActive(sb))
                            bcol[sb] = cc[B.origColAt(sb)];
                }
                // prod(rows×nw) = B_dense · Cc
                dgemmColsRaw(CblasNoTrans, static_cast<int>(rowsAcc),
                             static_cast<int>(nw), static_cast<int>(vbAcc),
                             1.0, B.dense().data(), static_cast<int>(rowsAcc),
                             tCc.data(), static_cast<int>(vbAcc), 0.0,
                             tProd.data(), static_cast<int>(rowsAcc),
                             opsTlsThreads());
            }
            for (int64_t t = 0; t < nw; ++t) {
                const int64_t od = C.origColAt(cStore[static_cast<size_t>(t0 + t)]);
                if (transB) {
                    // 转置散加：prod 行 = B 存储列 → 原始列散射
                    double* dst = accDstCol("opsColAccGemv", A, od);
                    if (!dst) return false;
                    const double* pc = tProd.data() +
                        static_cast<size_t>(t) * vbAcc;
                    for (int64_t sb = 0; sb < vbAcc; ++sb)
                        if (B.colActive(sb))
                            dst[B.origColAt(sb)] += pc[sb];
                    A.markColNonZero(od);
                } else {
                    scatterColumn(A, od, tProd.data(), t, rowsAcc);
                }
            }
        }
        return true;
    }

    // 逐列路径：缓冲 thread_local 复用（1.37M 次实调用 × 2–3 次 malloc
    // 的分配争用不可忽视）；xs 每列用后清零供下一列复用。
    static thread_local std::vector<double> tlT, tlXs, tlY;
    if (transB) {
        tlT.resize(static_cast<size_t>(B.validCols()));
        double* t = tlT.data();
        for (int64_t sc = 0; sc < C.validCols(); ++sc) {
            if (!C.colActive(sc)) continue;
            const int64_t od = C.origColAt(sc);
            // t = Bᵀ·C(:, od)（占位列分量恒 0，散加时只取 B 有效列）
            cblas_dgemv(CblasColMajor, CblasTrans,
                        static_cast<int>(C.rows()),
                        static_cast<int>(B.validCols()), 1.0,
                        B.dense().data(), static_cast<int>(C.rows()),
                        C.dense().col(sc).data(), 1, 0.0, t, 1);
            double* dst = accDstCol("opsColAccGemv", A, od);
            if (!dst) return false;
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    dst[B.origColAt(sb)] += t[sb];
            A.markColNonZero(od);
        }
    } else {
        tlXs.assign(static_cast<size_t>(B.validCols()), 0.0);
        tlY.resize(static_cast<size_t>(B.rows()));
        double* xs = tlXs.data();
        double* y = tlY.data();
        for (int64_t sc = 0; sc < C.validCols(); ++sc) {
            if (!C.colActive(sc)) continue;
            const int64_t od = C.origColAt(sc);
            const double* cCol = C.dense().col(sc).data();
            // xs 仅 B 有效列对应位置有值（占位列对应位置恒 0），
            // y = B_compressed · xs
            for (int64_t sb = 0; sb < B.validCols(); ++sb)
                if (B.colActive(sb))
                    xs[sb] = cCol[B.origColAt(sb)];
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        static_cast<int>(B.rows()),
                        static_cast<int>(B.validCols()), 1.0,
                        B.dense().data(), static_cast<int>(B.rows()),
                        xs, 1, 0.0, y, 1);
            std::fill_n(xs, B.validCols(), 0.0);  // 复位供下一列复用
            double* dst = accDstCol("opsColAccGemv", A, od);
            if (!dst) return false;
            for (int64_t r = 0; r < B.rows(); ++r) dst[r] += y[r];
            A.markColNonZero(od);
        }
    }
    return true;
}

bool opsColAccBatch(const ColCompMat& L,
                    const std::vector<const ColCompMat*>& psis,
                    const std::vector<ColCompMat*>& targets,
                    const std::vector<std::mutex*>& mtxs) {
    constexpr const char* kWho = "opsColAccBatch";
    if (psis.size() != targets.size() || psis.size() != mtxs.size()) {
        AMLS_LOG_ERROR("[colops] {} 失败：psis/targets/mtxs 数量不一致。", kWho);
        return false;
    }
    if (!colCompShapeOk(kWho, "L", L)) return false;
    for (size_t i = 0; i < psis.size(); ++i)
        if (!colCompShapeOk(kWho, "psi", *psis[i])) return false;

    // 收集 L 的有效列成连续左端（行 = L.rows，列 = 有效存储列），
    // 同时记录原始列号供右端行 gather。
    const int64_t actB = L.activeCols();
    if (actB == 0) return true;  // 恒零贡献：所有目标由调用方直接倒计数
    static thread_local std::vector<double> tlL, tlC, tlY;
    static thread_local std::vector<int64_t> tlOrigB;
    tlL.resize(static_cast<size_t>(L.rows() * actB));
    tlOrigB.resize(static_cast<size_t>(actB));
    {
        double* ld = tlL.data();
        int64_t g = 0;
        for (int64_t sb = 0; sb < L.validCols(); ++sb) {
            if (!L.colActive(sb)) continue;
            tlOrigB[static_cast<size_t>(g)] = L.origColAt(sb);
            std::memcpy(ld + static_cast<size_t>(g) * L.rows(),
                        L.dense().col(sb).data(),
                        static_cast<size_t>(L.rows()) * sizeof(double));
            ++g;
        }
    }

    // 右端拼接：C_all(actB × nAll)，行按 L 有效存储列序（与 tlL 列对齐）。
    // 各 Ψ 列宽可不同，逐列搬运并跳过占位列。
    int64_t nAll = 0;
    for (const ColCompMat* p : psis) {
        if (p->rows() != L.totalCols()) {
            AMLS_LOG_ERROR(
                "[colops] {} 失败：Ψ 行数 {} ≠ L 总列数 {}。", kWho, p->rows(),
                L.totalCols());
            return false;
        }
        nAll += p->activeCols();
    }
    if (nAll == 0) return true;
    tlC.resize(static_cast<size_t>(actB * nAll));
    {
        double* cd = tlC.data();
        int64_t off = 0;
        for (const ColCompMat* p : psis) {
            for (int64_t sc = 0; sc < p->validCols(); ++sc) {
                if (!p->colActive(sc)) continue;
                double* col = cd + static_cast<size_t>(off) * actB;
                for (int64_t g = 0; g < actB; ++g)
                    col[g] = p->dense().col(sc)[tlOrigB[static_cast<size_t>(g)]];
                ++off;
            }
        }
    }

    tlY.resize(static_cast<size_t>(L.rows() * nAll));
    dgemmColsRaw(CblasNoTrans, static_cast<int>(L.rows()),
                 static_cast<int>(nAll), static_cast<int>(actB), 1.0,
                 tlL.data(), static_cast<int>(L.rows()), tlC.data(),
                 static_cast<int>(actB), 0.0, tlY.data(),
                 static_cast<int>(L.rows()), opsTlsThreads());

    // 逐目标持锁散加各自列段（Ψ_a 的有效列 ↔ C_all 全局列段）。
    int64_t off = 0;
    for (size_t i = 0; i < psis.size(); ++i) {
        const ColCompMat& p = *psis[i];
        ColCompMat& A = *targets[i];
        if (p.activeCols() == 0) continue;
        // 覆盖性预检放锁外（只读）
        for (int64_t sc = 0; sc < p.validCols(); ++sc) {
            if (p.colActive(sc) && A.storageCol(p.origColAt(sc)) < 0) {
                AMLS_LOG_ERROR(
                    "[colops] {} 失败：目标 {} 存储列未覆盖 Ψ 有效列（原始列"
                    " {}）。",
                    kWho, i, p.origColAt(sc));
                return false;
            }
        }
        std::unique_lock<std::mutex> lock;
        if (mtxs[i]) lock = std::unique_lock<std::mutex>(*mtxs[i]);
        for (int64_t sc = 0; sc < p.validCols(); ++sc) {
            if (!p.colActive(sc)) continue;
            double* dst = accDstCol(kWho, A, p.origColAt(sc));
            if (!dst) return false;
            const double* ycol =
                tlY.data() + static_cast<size_t>(off) * L.rows();
            for (int64_t r = 0; r < L.rows(); ++r) dst[r] += ycol[r];
            A.markColNonZero(p.origColAt(sc));
            ++off;
        }
    }
    return true;
}

bool opsColAccBatchTrans(const ColCompMat& B,
                         const std::vector<const ColCompMat*>& Cs,
                         const std::vector<ColCompMat*>& targets,
                         const std::vector<std::mutex*>& mtxs) {
    constexpr const char* kWho = "opsColAccBatchTrans";
    if (Cs.size() != targets.size() || Cs.size() != mtxs.size()) {
        AMLS_LOG_ERROR("[colops] {} 失败：Cs/targets/mtxs 数量不一致。", kWho);
        return false;
    }
    if (!colCompShapeOk(kWho, "B", B)) return false;
    for (size_t i = 0; i < Cs.size(); ++i)
        if (!colCompShapeOk(kWho, "C", *Cs[i])) return false;
    if (B.activeCols() == 0) return true;

    // B 的有效行（存储列号 + 原始列号）预提取，散射循环零判断
    static thread_local std::vector<int64_t> tlActSb, tlActOrig;
    tlActSb.clear();
    tlActOrig.clear();
    for (int64_t sb = 0; sb < B.validCols(); ++sb) {
        if (!B.colActive(sb)) continue;
        tlActSb.push_back(sb);
        tlActOrig.push_back(B.origColAt(sb));
    }

    // 右端拼接 C_all(B.rows × nAll)
    int64_t nAll = 0;
    for (const ColCompMat* c : Cs) {
        if (c->rows() != B.rows()) {
            AMLS_LOG_ERROR("[colops] {} 失败：C 行数 {} ≠ B 行数 {}。", kWho,
                           c->rows(), B.rows());
            return false;
        }
        nAll += c->activeCols();
    }
    if (nAll == 0) return true;
    static thread_local std::vector<double> tlC, tlProd;
    tlC.resize(static_cast<size_t>(B.rows() * nAll));
    {
        double* cd = tlC.data();
        for (const ColCompMat* c : Cs) {
            for (int64_t sc = 0; sc < c->validCols(); ++sc) {
                if (!c->colActive(sc)) continue;
                std::memcpy(cd, c->dense().col(sc).data(),
                            static_cast<size_t>(B.rows()) * sizeof(double));
                cd += B.rows();
            }
        }
    }

    // prod(B.validCols × nAll) = Bᵀ · C_all（占位行的乘积被散射丢弃）
    tlProd.resize(static_cast<size_t>(B.validCols() * nAll));
    dgemmColsRaw(CblasTrans, static_cast<int>(B.validCols()),
                 static_cast<int>(nAll), static_cast<int>(B.rows()), 1.0,
                 B.dense().data(), static_cast<int>(B.rows()), tlC.data(),
                 static_cast<int>(B.rows()), 0.0, tlProd.data(),
                 static_cast<int>(B.validCols()), opsTlsThreads());

    // 逐目标持锁散射各自列段：dst[B.origColAt(sb)] += prod(sb, g)
    const double* pc = tlProd.data();
    for (size_t i = 0; i < Cs.size(); ++i) {
        const ColCompMat& c = *Cs[i];
        ColCompMat& A = *targets[i];
        if (c.activeCols() == 0) continue;
        std::unique_lock<std::mutex> lock;
        if (mtxs[i]) lock = std::unique_lock<std::mutex>(*mtxs[i]);
        for (int64_t sc = 0; sc < c.validCols(); ++sc) {
            if (!c.colActive(sc)) continue;
            double* dst = accDstCol(kWho, A, c.origColAt(sc));
            if (!dst) return false;
            for (size_t t = 0; t < tlActSb.size(); ++t)
                dst[tlActOrig[t]] += pc[tlActSb[t]];
            A.markColNonZero(c.origColAt(sc));
            pc += B.validCols();
        }
    }
    return true;
}

bool opsColAccGemm(const ColCompMat& B, bool transB, const ColCompMat& C,
                   ColCompMat& A) {
    auto& stGM = opAccStats();
    const auto tGM = std::chrono::steady_clock::now();
    struct GemmRec {
        decltype(tGM) t0;
        ~GemmRec() {
            if (!opAccEnabled()) return;
            opAccStats().gemmCalls.fetch_add(1, std::memory_order_relaxed);
            opAccStats().gemmNs.fetch_add(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0).count()),
                std::memory_order_relaxed);
        }
    } gemmRec{tGM};
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
    if (opAccEnabled())
        stGM.gemmFlops.fetch_add(
        static_cast<uint64_t>(2.0 * static_cast<double>(B.rows()) *
                              static_cast<double>(B.validCols()) *
                              static_cast<double>(C.activeCols())),
        std::memory_order_relaxed);

    const int64_t nnzC = C.activeCols();
    std::vector<double> cg;
    const std::vector<int64_t> cStore = gatherActiveCols(C, cg);
    if (opAccEnabled())
        stGM.tmpAllocBytes.fetch_add(
            static_cast<uint64_t>(cg.size() * sizeof(double)),
            std::memory_order_relaxed);
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
            double* dst = accDstCol("opsColAccGemv", A, od);
            if (!dst) return false;
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

} // namespace aams
