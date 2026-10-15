#include "partition/symbolic_analyzer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "core/types.hpp"

namespace amls {
namespace sym {

namespace {

using SparseMatrix = DefaultTraits::SparseMatrix;
using Index = DefaultTraits::Index;

// 用 Eigen SimplicialLDLT（NaturalOrdering, 已是预排列）做符号分析。
// 输入 A 是原始矩阵，permND 为 METIS 排列（perm[new]=old）。
// 构造 A_permuted = P A Pᵀ，analyzePattern 后提取 L。
FactorMeta analyzeWithEigen(const SparseMatrix& A,
                            const std::vector<int64_t>& permND,
                            bool wantPatterns) {
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
    // 模式源选择（AMLS_SUPERNODE_PATTERN）:
    //   k（默认）— 排列后输入的下三角（K 图，无填充）。Abaqus 的真实判据
    //     （fca 精确嵌套标定: cant_10 {3:1293,6:11,12:4} / cant_20
    //     {3:9143,6:48,9:2,12:4}，合并率 ~1%；cant_20_K 实测 {3:8614,6:97,9:4}）。
    //     K 嵌套对仍是 etree 链段，块特征值/缩聚与 factor 模式逐位一致
    //     （cant_20_K 12 模态实测相同）。
    //   factor — L 的填充列结构。合并块 = 因子嵌套链前缀；规则网格区的
    //     填充等结构链更长（合并率 ~44% vs K 模式 ~1%），尺寸仍均匀
    //     （≤链上限）但与 Abaqus 合并模式差异大。仅作 A/B 备选。
    // 注意: 曾疑似 K 模式导致负特征值——后查明是输入矩阵 STIF2_full 的
    // 1e36 罚对角在缩聚中的数值灾难（HEAD/factor 模式同样复现），与判据无关。
    const SparseMatrix* patternSource = &Bl;
    if (wantPatterns) {
        const char* pat = std::getenv("AMLS_SUPERNODE_PATTERN");
        if (pat && std::string_view(pat) == "factor") patternSource = nullptr;
    }
    return extractFactorMeta(L, n, wantPatterns, patternSource);
}

}  // namespace

FactorMeta extractFactorMeta(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>& L, int64_t n,
    bool wantPatterns, const SparseMatrix* patternSource) {
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
    if (wantPatterns) {
        // 双模式（见 analyzeWithEigen 注释）: patternSource 非空（默认 k）
        // 复制其下三角（K 图）；为空（AMLS_SUPERNODE_PATTERN=factor）复制
        // L 的填充列结构。两种模式都是列内升序、含自身对角，供
        // supernode_builder 的 sameSupernodePattern 做嵌套比较。
        if (patternSource == nullptr) {
            meta.colPtr.assign(outer, outer + n + 1);
            meta.colIdx.assign(inner, inner + L.nonZeros());
        } else {
            meta.colPtr.assign(static_cast<size_t>(n) + 1, 0);
            for (int64_t j = 0; j < n; ++j) {
                const int64_t lo = patternSource->outerIndexPtr()[j],
                              hi = patternSource->outerIndexPtr()[j + 1];
                for (int64_t p = lo; p < hi; ++p) {
                    const int64_t r = patternSource->innerIndexPtr()[p];
                    if (r >= j) meta.colIdx.push_back(r);
                }
                meta.colPtr[static_cast<size_t>(j) + 1] =
                    static_cast<int64_t>(meta.colIdx.size());
            }
        }
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
                                    const std::vector<int64_t>& permND,
                                    bool wantPatterns) {
    const int64_t n = A.rows();
    if (static_cast<int64_t>(permND.size()) != n)
        throw std::runtime_error("symbolic_analyzer: 排列长度 != n");
    SymbolicAnalysis sa;
    sa.perm = permND;
    sa.inversePerm.assign(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i)
        sa.inversePerm[static_cast<size_t>(permND[static_cast<size_t>(i)])] = i;

    FactorMeta meta = analyzeWithEigen(A, permND, wantPatterns);
    sa.parentColumn = std::move(meta.parentColumn);
    sa.columnCount = std::move(meta.columnCount);
    sa.componentRoots = std::move(meta.componentRoots);
    sa.colPtr = std::move(meta.colPtr);
    sa.colIdx = std::move(meta.colIdx);
    return sa;
}

}  // namespace sym
}  // namespace amls
