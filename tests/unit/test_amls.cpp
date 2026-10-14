// AMLS 求解器端到端单元测试（双池节点序 DAG + 频率制截断）。
// 缩聚结构性质与暴力对拍已随 PKU 列压缩移植迁至
// test_pipeline_integration / test_gauss_module / test_colcomp。
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Dense>

#include "amls/solver.hpp"
#include "io/case_generator.hpp"

using namespace amls;
using DM = DefaultTraits::DenseMatrix;

TEST_CASE("求解器:DAG 全保留 ⇒ 精确谱与回代残差", "[solver]") {
    // numThreads=4 走双池节点序 DAG；局部截断 (2π·γ·maxFreq)² 覆盖全谱
    //（grid 10×10 λmax≈7.9 < (2π·0.6)²）⇒ 谱精确、回代残差机器精度。
    auto pair = gridGraph2d(10, 10);
    AmlsOptions opt;
    opt.nLevels = 3;
    opt.partitionMethod = GENERATE(std::string("bisect"), std::string("nd"));
    opt.leafSize = 20;
    opt.minFreq = 0.0;
    opt.maxFreq = 0.3;
    opt.gainCoefficient = 2.0;
    opt.numThreads = 4;
    opt.verbosity = 0;
    opt.recoverVectors = true;
    AmlsSolver solver;
    auto sol = solver.solve({pair.K, pair.M}, opt);
    auto ref = gridGraph2dSpectrum(10, 10);
    REQUIRE(sol.eigenvalues.size() >= 10);
    for (int64_t j = 0; j < 10; ++j)
        REQUIRE(std::abs(sol.eigenvalues[j] - ref[j]) <=
                1e-8 * std::max(1.0, std::abs(ref[j])));
    for (int64_t j = 0; j < 3; ++j) {
        DM rz = DM(pair.K) * sol.eigenvectors.col(j) -
                sol.eigenvalues[j] * (DM(pair.M) * sol.eigenvectors.col(j));
        REQUIRE(rz.norm() <
                1e-8 * (DM(pair.K) * sol.eigenvectors.col(j)).norm());
    }
}
