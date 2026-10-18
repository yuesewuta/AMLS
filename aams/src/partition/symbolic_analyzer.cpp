#include "partition/symbolic_analyzer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "core/types.hpp"

namespace aams {
namespace sym {

namespace {

using SparseMatrix = DefaultTraits::SparseMatrix;
using Index = DefaultTraits::Index;

// 用 Eigen SimplicialLDLT（NaturalOrdering, 已是预排列）做符号分析。
// 输入 A 是原始矩阵，permND 为 METIS 排列（perm[new]=old）。
// 构造 A_permuted = P A Pᵀ，analyzePattern 后提取 L。
SparseMatrix buildPermutedLower(const SparseMatrix& A,
                                const std::vector<int64_t>& permND);
FactorMeta analyzeWithEigen(const SparseMatrix& A,
                            const std::vector<int64_t>& permND) {
    const int64_t n = A.rows();
    const SparseMatrix Bl = buildPermutedLower(A, permND);
    // 符号分解（不数值分解）。必须显式 NaturalOrdering：SimplicialLDLT 默认
    // 是 AMDOrdering，会在内部再做一次 AMD 重排，使 matrixL 落在 AMD 坐标系，
    // 提取出的 parentColumn/columnCount 与 ND 坐标完全错位（超节点树全毁）。
    // 输入已按 permND 预排列，这里要求保持自然序。
    using LdlT = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                                       Eigen::NaturalOrdering<Index>>;
    LdlT ldl;
    ldl.analyzePattern(Bl);
    if (ldl.info() != Eigen::Success) {
        std::fprintf(stderr,
                     "[sym] analyzePattern 失败 info=%d n=%lld nnz=%lld\n",
                     (int)ldl.info(), (long long)n, (long long)Bl.nonZeros());
        throw std::runtime_error("symbolic_analyzer: analyzePattern 失败");
    }
    // 必须数值分解：analyzePattern 只填 L 的列计数（outerIndexPtr），行索引
    // innerIndexPtr 由 factorize 才写入（Eigen init_matrix 只 partial_sum +
    // resizeNonZeros）。只 analyze 就读 matrixL 的 inner 是未初始化内存，
    // 表现为 etree 几乎全根、全 1-DOF 块。K+M（lumped M>0）SPD 保证成功。
    ldl.factorize(Bl);
    if (ldl.info() != Eigen::Success) {
        std::fprintf(stderr,
                     "[sym] factorize 失败 info=%d n=%lld（要求 SPD 的 K+M）\n",
                     (int)ldl.info(), (long long)n);
        throw std::runtime_error("symbolic_analyzer: factorize 失败");
    }
    const auto& Lv = ldl.matrixL();
    Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t> L = Lv;
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[sym] L nnz=%lld (from numeric factor)\n",
                     (long long)L.nonZeros());
    return extractFactorMeta(L, n);
}

// 排列 + 取下三角（analyzeWithEigen 前半段抽出共用）。
SparseMatrix buildPermutedLower(const SparseMatrix& A,
                                const std::vector<int64_t>& permND) {
    const int64_t n = A.rows();
    // 重排: B = P A Pᵀ (P: permND[new]=old)
    std::vector<Index> inv(static_cast<size_t>(n));
    for (Index j = 0; j < static_cast<Index>(permND.size()); ++j)
        inv[static_cast<size_t>(permND[static_cast<size_t>(j)])] = j;
    std::vector<Eigen::Triplet<double, Index>> trips;
    trips.reserve(static_cast<size_t>(A.nonZeros()));
    for (Index c = 0; c < A.outerSize(); ++c)
        for (SparseMatrix::InnerIterator it(A, c); it; ++it)
            trips.emplace_back(inv[static_cast<size_t>(it.row())],
                               inv[static_cast<size_t>(it.col())], it.value());
    SparseMatrix B(n, n);
    B.setFromTriplets(trips.begin(), trips.end());
    B.makeCompressed();
    // 取对称下三角（结构语义: 无向图）
    SparseMatrix Bl(n, n);
    {
        std::vector<Eigen::Triplet<double, Index>> lt;
        for (Index c = 0; c < B.outerSize(); ++c)
            for (SparseMatrix::InnerIterator it(B, c); it; ++it)
                if (it.row() >= it.col())
                    lt.emplace_back(it.row(), it.col(), it.value());
        Bl.setFromTriplets(lt.begin(), lt.end());
        Bl.makeCompressed();
    }
    return Bl;
}

// 纯符号分析（无数值分解）。拆成两半：
//   parentColumn —— etree 直接从输入图算（CSparse cs_etree；已对拍
//     SimplicialLDLT 因子的 min-row 父列，逐列一致）；
//   columnCount —— Eigen SimplicialLDLT::analyzePattern（NaturalOrdering）
//     只填 L 的列计数 outerIndexPtr（真实符号填充计数，见
//     analyzeWithEigen 注释），行索引未初始化但我们不读；免去
//     factorize 的 O(Σ列²) 算术（cant_31 节点级 3.4s → ~0.1s）。
// AAMS_SYM_EIGEN=1 可回退 Eigen 数值路径 A/B。
FactorMeta analyzeStructureOnly(const SparseMatrix& BlLower, int64_t n) {
    using LMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>;
    FactorMeta meta;
    meta.parentColumn.assign(static_cast<size_t>(n), -1);
    meta.columnCount.assign(static_cast<size_t>(n), 0);
    if (n == 0) return meta;
    const bool prof = std::getenv("AAMS_PROFILE") != nullptr;
    auto clockNow = [] { return std::chrono::steady_clock::now(); };
    auto profMark = [&](const char* what,
                        std::chrono::steady_clock::time_point t0) {
        if (prof)
            std::fprintf(stderr, "[prof] sym %s=%.2fs\n", what,
                         std::chrono::duration<double>(clockNow() - t0).count());
    };
    auto ts = clockNow();
    // etree（CSparse cs_etree 算法；上三角列邻接 = 下三角转置）。
    // 父列恒 > 子列，故 seed(>k) 的祖先链全部落在 (k, n) 内。
    LMat Et = BlLower.transpose();
    Et.makeCompressed();
    profMark("transpose", ts);
    auto te = clockNow();
    std::vector<int64_t> parent(static_cast<size_t>(n), -1);
    std::vector<int64_t> ancestor(static_cast<size_t>(n), -1);
    for (int64_t k = 0; k < n; ++k) {
        for (LMat::InnerIterator it(Et, k); it; ++it) {
            int64_t i = it.row();  // i < k
            while (i != -1 && i < k) {
                const int64_t inext = ancestor[static_cast<size_t>(i)];
                ancestor[static_cast<size_t>(i)] = k;
                if (inext == -1) parent[static_cast<size_t>(i)] = k;
                i = inext;
            }
        }
    }
    profMark("etree", te);
    auto tc = clockNow();
    // 列计数：analyzePattern 的符号填充计数（不得 factorize——那正是
    // 要省掉的数值分解）。
    using LdlT = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                                       Eigen::NaturalOrdering<Index>>;
    LdlT ldl;
    ldl.analyzePattern(BlLower);
    if (ldl.info() != Eigen::Success)
        throw std::runtime_error("symbolic_analyzer: analyzePattern 失败");
    const auto& Lm = ldl.matrixL().nestedExpression();
    const int64_t nnzL = Lm.outerIndexPtr()[n];
    // LDLT 的 L 单位对角隐式不存，计数 +1 对齐 factorize 后含对角的语义
    // （extractFactorMeta hi-lo 含对角；cant_31 对拍 27 = 26+1）。
    for (int64_t j = 0; j < n; ++j)
        meta.columnCount[static_cast<size_t>(j)] =
            Lm.outerIndexPtr()[j + 1] - Lm.outerIndexPtr()[j] + 1;
    profMark("counts", tc);
    if (prof)
        std::fprintf(stderr, "[prof] sym n=%lld nnzL=%lld\n", (long long)n,
                     (long long)nnzL);
    // etree(A) = etree(L)（Liu 基本定理），父列直接用 A 上算出的 etree。
    meta.parentColumn = std::move(parent);
    // 连通分量根：沿 parent 链并查集收缩（与 extractFactorMeta 相同）。
    std::vector<int64_t> root(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) root[static_cast<size_t>(j)] = j;
    std::function<int64_t(int64_t)> find = [&](int64_t x) -> int64_t {
        while (root[static_cast<size_t>(x)] != x) {
            root[static_cast<size_t>(x)] =
                root[static_cast<size_t>(root[static_cast<size_t>(x)])];
            x = root[static_cast<size_t>(x)];
        }
        return x;
    };
    for (int64_t j = 0; j < n; ++j) {
        const int64_t p = meta.parentColumn[static_cast<size_t>(j)];
        if (p >= 0) {
            const int64_t rj = find(j);
            const int64_t rp = find(p);
            if (rj != rp) root[static_cast<size_t>(rj)] = rp;
        }
    }
    std::vector<char> seen(static_cast<size_t>(n), 0);
    for (int64_t j = 0; j < n; ++j) {
        const int64_t r = find(j);
        if (!seen[static_cast<size_t>(r)]) {
            seen[static_cast<size_t>(r)] = 1;
            meta.componentRoots.push_back(r);
        }
    }
    return meta;
}

}  // namespace

FactorMeta extractFactorMeta(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>& L, int64_t n) {
    FactorMeta meta;
    meta.parentColumn.assign(static_cast<size_t>(n), -1);
    meta.columnCount.assign(static_cast<size_t>(n), 0);
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    for (int64_t j = 0; j < n; ++j) {
        const int64_t lo = outer[j], hi = outer[j + 1];
        meta.columnCount[static_cast<size_t>(j)] = hi - lo;
        // etree 父 = 列 j 中 > j 的最小行号（取 min 而非"第一个存储的"，
        // 不依赖 Eigen 内层索引有序性）。
        int64_t par = -1;
        for (int64_t p = lo; p < hi; ++p) {
            const int64_t r = inner[p];
            if (r > j && (par < 0 || r < par)) par = r;
        }
        meta.parentColumn[static_cast<size_t>(j)] = par;
    }
    // 连通分量根: 沿 parent 链向上找到顶层列（parent 链末端）。
    // 注意: L 的叶列 parent=-1 是因为无非对角元，但消去树里真正的根是
    // 那些"无父列"且沿链到顶的列；叶列 parent=-1 时它本身就是根。
    // 用并查集按 parent 链收缩: parent[j] 若 >=0 则 j 的根 = parent[j] 的根。
    std::vector<int64_t> root(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) root[static_cast<size_t>(j)] = j;
    std::function<int64_t(int64_t)> find = [&](int64_t x) -> int64_t {
        while (root[static_cast<size_t>(x)] != x) {
            root[static_cast<size_t>(x)] =
                root[static_cast<size_t>(root[static_cast<size_t>(x)])];
            x = root[static_cast<size_t>(x)];
        }
        return x;
    };
    for (int64_t j = 0; j < n; ++j) {
        const int64_t p = meta.parentColumn[static_cast<size_t>(j)];
        if (p >= 0) {
            const int64_t rj = find(j);
            const int64_t rp = find(p);
            if (rj != rp) root[static_cast<size_t>(rj)] = rp;
        }
    }
    std::vector<char> seen(static_cast<size_t>(n), 0);
    for (int64_t j = 0; j < n; ++j) {
        const int64_t r = find(j);
        if (!seen[static_cast<size_t>(r)]) {
            seen[static_cast<size_t>(r)] = 1;
            meta.componentRoots.push_back(r);
        }
    }
    return meta;
}

SymbolicAnalysis analyzeSymmetrized(const SparseMatrix& A,
                                    const std::vector<int64_t>& permND) {
    const int64_t n = A.rows();
    if (static_cast<int64_t>(permND.size()) != n)
        throw std::runtime_error("symbolic_analyzer: 排列长度 != n");
    SymbolicAnalysis sa;
    sa.perm = permND;
    sa.inversePerm.assign(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i)
        sa.inversePerm[static_cast<size_t>(permND[static_cast<size_t>(i)])] = i;

    // 默认走纯符号路径（无算术，结构结果与数值分解一致）；A/B 用
    // AAMS_SYM_EIGEN=1 回退 SimplicialLDLT 数值路径。
    static const bool symEigen = std::getenv("AAMS_SYM_EIGEN") != nullptr;
    FactorMeta meta;
    if (std::getenv("AAMS_SYM_DIFF")) {
        const SparseMatrix Bl = buildPermutedLower(A, permND);
        FactorMeta a = analyzeStructureOnly(Bl, n);
        FactorMeta b = analyzeWithEigen(A, permND);
        int64_t nCnt = 0, nPar = 0, firstBad = -1;
        for (int64_t j = 0; j < n; ++j) {
            if (a.columnCount[static_cast<size_t>(j)] !=
                b.columnCount[static_cast<size_t>(j)]) {
                ++nCnt;
                if (firstBad < 0) firstBad = j;
            }
            if (a.parentColumn[static_cast<size_t>(j)] !=
                b.parentColumn[static_cast<size_t>(j)])
                ++nPar;
        }
        std::fprintf(stderr,
                     "[symdiff] n=%lld countMismatch=%lld parentMismatch=%lld "
                     "firstBadCol=%lld\n",
                     (long long)n, (long long)nCnt, (long long)nPar,
                     (long long)firstBad);
        if (firstBad >= 0)
            std::fprintf(stderr,
                         "[symdiff] col %lld: symCount=%lld eigCount=%lld "
                         "symParent=%lld eigParent=%lld\n",
                         (long long)firstBad,
                         (long long)a.columnCount[static_cast<size_t>(firstBad)],
                         (long long)b.columnCount[static_cast<size_t>(firstBad)],
                         (long long)a.parentColumn[static_cast<size_t>(firstBad)],
                         (long long)b.parentColumn[static_cast<size_t>(firstBad)]);
        meta = std::move(b);
    } else {
        meta = symEigen ? analyzeWithEigen(A, permND)
                        : analyzeStructureOnly(buildPermutedLower(A, permND),
                                               n);
    }
    sa.parentColumn = std::move(meta.parentColumn);
    sa.columnCount = std::move(meta.columnCount);
    sa.componentRoots = std::move(meta.componentRoots);
    return sa;
}

}  // namespace sym
}  // namespace aams
