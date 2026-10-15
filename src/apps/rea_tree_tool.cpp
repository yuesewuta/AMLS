// rea_tree_tool: 独立验证 REA-AMLS 树构造（不跑求解，只测树形态）。
// 用法: rea_tree_tool --K file [--max-block-dofs N] [--collapse-dofs N]
//                     [--leaf-aggr-dofs N]
// 输出: 树统计 + 块大小分布 + K 边祖先可比计数（对齐 Abaqus stf 树形诊断）。
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "core/logging.hpp"
#include "core/types.hpp"
#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "rea/aggregation.hpp"
#include "rea/tree_builder.hpp"

using namespace amls;

int main(int argc, char** argv) {
    std::string kFile, mFile, caseSpec;
    amls::rea::AbaqusTreeOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { return argv[++i]; };
        if (s == "--K") kFile = next();
        else if (s == "--M") mFile = next();
        else if (s == "--case") caseSpec = next();
        else if (s == "--max-block-dofs") opt.maxBlockDofs = std::stoll(next());
        else if (s == "--collapse-dofs") opt.collapseMaxDofs = std::stoll(next());
        else if (s == "--leaf-aggr-dofs") opt.leafAggrMaxDofs = std::stoll(next());
        else if (s == "--min-block-dofs") opt.minBlockDofs = std::stoll(next());
        else if (s == "--no-aggr-leaves") opt.aggregateLeaves = false;
        else if (s == "--no-collapse") opt.collapseMode = 0;
        else if (s == "--super-root-dofs") opt.superRootDofs = std::stoll(next());
        else if (s == "--dofs-per-node") opt.dofsPerNode = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    DefaultTraits::SparseMatrix K, M;
    if (!caseSpec.empty()) {
        // grid3d:N
        const auto colon = caseSpec.find(':');
        const std::string type = caseSpec.substr(0, colon);
        const int64_t n = std::stoll(caseSpec.substr(colon + 1));
        if (type == "grid3d") {
            auto p = gridGraph3d(n, n, n);
            K = p.K; M = p.M;
        } else if (type == "grid") {
            auto p = gridGraph2d(n, n);
            K = p.K; M = p.M;
        } else if (type == "path") {
            auto p = pathGraph(n);
            K = p.K; M = p.M;
        } else {
            std::fprintf(stderr, "unknown case %s\n", type.c_str());
            return 2;
        }
    } else {
        K = readMatrixMarket<DefaultTraits>(kFile);
        if (!mFile.empty())
            M = readMatrixMarket<DefaultTraits>(mFile);
    }
    // 与 PKU 求解路径一致（solver.cpp:183-185）：数值和 K+M。
    // 结构上与 |K|+|M| 并集等价，但 SPD 数值使 AMLS_DUMP_TREE 诊断路径的
    // 数值 factorize 能成功（全 1 值矩阵非正定，会导致回退读取到脏的 L）。
    DefaultTraits::SparseMatrix A = K;
    if (M.rows() > 0)
        A = K + M;

    // 工具默认允许合成超根（空 DOF 结构合并点），避免多根森林直接报错；
    // 单根时该参数不生效。
    if (opt.superRootDofs <= 0)
        opt.superRootDofs = A.rows();

    std::fprintf(stderr,
                 "[rea-tree] n=%lld nnz=%lld maxBlockDofs=%lld "
                 "collapse=%lld leafAggr=%lld mode=%d aggrLeaves=%d\n",
                 (long long)A.rows(), (long long)A.nonZeros(),
                 (long long)opt.maxBlockDofs,
                 (long long)opt.collapseMaxDofs,
                 (long long)opt.leafAggrMaxDofs, opt.collapseMode,
                 opt.aggregateLeaves ? 1 : 0);

    auto res = amls::rea::buildAbaqusTree(A, opt);
    const auto& tree = res.tree;
    std::vector<int64_t> sizes;
    {
        const auto order = tree.postOrder();
        for (const int64_t id : order)
            sizes.push_back(static_cast<int64_t>(
                tree.node(static_cast<size_t>(id)).indices.size()));
    }
    std::sort(sizes.rbegin(), sizes.rend());
    std::fprintf(stderr, "[rea-tree] nodes=%lld permSize=%lld\n",
                 (long long)tree.nNodes(), (long long)res.perm.size());
    std::fprintf(stderr, "[rea-tree] top-sizes:");
    for (int64_t i = 0;
         i < static_cast<int64_t>(std::min<int64_t>(16, sizes.size())); ++i)
        std::fprintf(stderr, " %lld", (long long)sizes[static_cast<size_t>(i)]);
    std::fprintf(stderr, "\n");
    // 大小直方图
    std::map<int64_t, int64_t> hist;
    for (const auto& s : sizes) hist[s]++;
    std::fprintf(stderr, "[rea-tree] size-dist:");
    for (const auto& [sz, cnt] : hist)
        std::fprintf(stderr, " %lld:%lld", (long long)sz, (long long)cnt);
    std::fprintf(stderr, "\n");
    return 0;
}