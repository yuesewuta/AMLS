#include <catch2/catch_test_macros.hpp>
#include "amls/solver.hpp"
#include "io/case_generator.hpp"
#include "reference/error_metrics.hpp"

TEST_CASE("Recovered vectors are mass normalized for diagonal and general mass", "[solver][normalize]") {
    for (bool consistent : {false, true}) {
        auto p = amls::pathGraph(40, consistent);
        p.M *= 3.5;
        amls::AmlsOptions options;
        // path 40 + M×3.5：λmax = 4/3.5 ≈ 1.14；maxFreq=0.4Hz（λ_hi≈6.3）
        // 覆盖全谱 ⇒ 全保留 ⇒ 输出前 5 阶即真谱前 5 阶。
        options.minFreq = 0.0;
        options.maxFreq = 0.4;
        options.gainCoefficient = 2.0;
        options.extractOrder = 5;
        options.leafSize = 8;
        options.nLevels = 3;
        options.numThreads = 4;
        options.verbosity = 0;
        amls::AmlsSolver solver;
        auto solution = solver.solve({p.K, p.M}, options);
        for (int col = 0; col < 5; ++col) {
            auto z = solution.eigenvectors.col(col);
            REQUIRE(std::abs(z.dot(p.M * z) - 1.0) < 1e-12);
            REQUIRE(amls::ErrorMetrics::residual(p.K, p.M,
                                                solution.eigenvalues[col], z) < 1e-10);
        }
    }
}
