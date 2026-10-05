// amls_verify:误差评估与参数扫描（验证方案 §3.3）,写验证报告。
// 用法:
//   amls_verify [--suite all|closed|hb|nd] [--nev K] [--outdir DIR] [--quick]
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/SparseCholesky>

#include "amls/solver.hpp"
#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "reference/error_metrics.hpp"
#include "reference/reference_solver.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

struct Row {
    std::string caseName;
    double eta = 0;
    int nLevels = 0;
    int64_t p = 0;
    double maxRelErr = 0;
    double maxFreqErr = 0;
    double phase1 = 0, phase2 = 0, back = 0, partition = 0;
    bool pass = false;
};

std::vector<Row> rows;

void runCase(const std::string& name, amls::PencilPair pencil,
             const amls::DefaultTraits::Vector& lambdaRef, int64_t nev,
             double eta, int nLevels, bool useShift, bool quiet = true,
             int64_t leafSize = 500) {
    amls::AmlsOptions opt;
    opt.nev = nev;
    opt.thresholdFactor = eta;
    opt.nLevels = nLevels;
    opt.leafSize = leafSize;
    opt.shift = useShift ? 1e-8 : 0.0;
    opt.verbosity = quiet ? 0 : 1;

    amls::Pencil<amls::DefaultTraits> km{pencil.K, pencil.M};
    amls::AmlsSolver solver;
    Row row;
    row.caseName = name;
    row.eta = eta;
    row.nLevels = nLevels;
    try {
        auto sol = solver.solve(km, opt);
        row.p = sol.p;
        // 比较时把移位减掉
        amls::DefaultTraits::Vector theta = sol.theta;
        if (useShift)
            theta.array() -= opt.shift;
        const auto errs = amls::ErrorMetrics::eigenvalueRelErr(theta, lambdaRef);
        row.maxRelErr = 0;
        for (double e : errs)
            row.maxRelErr = std::max(row.maxRelErr, e);
        const auto ferrs = amls::ErrorMetrics::frequencyRelErr(theta, lambdaRef);
        for (double e : ferrs)
            row.maxFreqErr = std::max(row.maxFreqErr, e);
        // 5 类架构计时映射:phase1 = 高斯消去 + 缩聚,phase2 = 降阶特征解
        row.phase1 = (sol.timing.count("gauss") ? sol.timing.at("gauss") : 0) +
                     (sol.timing.count("condense") ? sol.timing.at("condense")
                                                   : 0);
        row.phase2 = sol.timing.count("reduced") ? sol.timing.at("reduced") : 0;
        row.back = sol.timing.count("backtransform")
                       ? sol.timing.at("backtransform") : 0;
        row.partition = sol.timing.count("partition")
                            ? sol.timing.at("partition") : 0;
        row.pass = row.maxRelErr <= 0.04;
    } catch (const std::exception& e) {
        row.maxRelErr = -1;
        std::cerr << "[verify] " << name << " η=" << eta << " L=" << nLevels
                  << " 失败: " << e.what() << "\n";
    }
    rows.push_back(row);
    std::cout << "[verify] " << name << " η=" << eta << " L=" << nLevels
              << " p=" << row.p << " maxRelErr=" << row.maxRelErr
              << (row.pass ? " PASS" : " FAIL") << "\n";
}

/// 数值 SPD 快检:全部对角元 > 0 且对称 Cholesky 可分解。
static bool numericallySpd(const amls::DefaultTraits::SparseMatrix& A) {
    for (int c = 0; c < A.cols(); ++c) {
        bool hasDiag = false;
        for (amls::DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
            if (it.row() == c) { hasDiag = it.value() > 0; break; }
        if (!hasDiag) return false;
    }
    Eigen::SimplicialLLT<amls::DefaultTraits::SparseMatrix> llt(A);
    return llt.info() == Eigen::Success;
}

amls::PencilPair loadPair(const std::string& kPath, const std::string& mPath,
                          bool useIdentityM) {
    amls::PencilPair p;
    p.K = amls::readMatrixMarket<amls::DefaultTraits>(kPath);
    if (!useIdentityM && !mPath.empty())
        p.M = amls::readMatrixMarket<amls::DefaultTraits>(mPath);
    else {
        p.M.resize(p.K.rows(), p.K.cols());
        p.M.setIdentity();
    }
    // M 数值非正定（如 bcsstm01 零对角 / bcsstm13 不定）违反方法前提
    // （K 对称、M 对称正定）,回退单位阵并在输出中注明。
    if (!numericallySpd(p.M)) {
        std::cout << "[verify] 注意: " << kPath << " 的 M 非正定,改用单位阵\n";
        p.M.resize(p.K.rows(), p.K.cols());
        p.M.setIdentity();
    }
    return p;
}

void writeReport(const std::string& dir) {
    {
        std::ofstream csv(dir + "/verify_results.csv");
        csv << "case,eta,nLevels,p,maxRelErr,maxFreqErr,phase1_s,phase2_s,"
               "back_s,partition_s,pass\n";
        for (const auto& r : rows) {
            csv << r.caseName << "," << r.eta << "," << r.nLevels << "," << r.p
                << "," << r.maxRelErr << "," << r.maxFreqErr << "," << r.phase1
                << "," << r.phase2 << "," << r.back << "," << r.partition
                << "," << (r.pass ? 1 : 0) << "\n";
        }
    }
    {
        std::ofstream md(dir + "/verify_report.md");
        md << "# AMLS 验证报告\n\n";
        md << "主验收指标:特征值相对误差 ≤ 4%（等价于频率误差 ≤ 2%）。\n\n";
        md << "| 算例 | η | 层数 | p | max ε_λ | max ε_f | Phase1(s) | 通过 |\n";
        md << "|---|---|---|---|---|---|---|---|\n";
        int pass = 0, total = 0;
        for (const auto& r : rows) {
            ++total;
            if (r.pass)
                ++pass;
            md << "| " << r.caseName << " | " << r.eta << " | " << r.nLevels
               << " | " << r.p << " | " << r.maxRelErr << " | " << r.maxFreqErr
               << " | " << r.phase1 << " | " << (r.pass ? "✅" : "❌")
               << " |\n";
        }
        md << "\n通过 " << pass << "/" << total << "。\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string suite = "all", outdir = "build/verify";
    int64_t nev = 50;
    bool quick = false;
    std::string matrixDir = "tests/matrices";

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() {
            if (++i >= argc) {
                std::cerr << "参数缺失\n";
                std::exit(2);
            }
            return std::string(argv[i]);
        };
        if (s == "--suite") suite = next();
        else if (s == "--outdir") outdir = next();
        else if (s == "--nev") nev = std::stoll(next());
        else if (s == "--matrices") matrixDir = next();
        else if (s == "--quick") quick = true;
        else {
            std::cerr << "未知参数 " << s << "\n";
            std::exit(2);
        }
    }

    const std::vector<double> etas = quick ? std::vector<double>{2.0}
                                           : std::vector<double>{1.2, 1.5, 2.0, 3.0, 4.0};
    const std::vector<int> levels = quick ? std::vector<int>{4}
                                          : std::vector<int>{2, 3, 4, 5, 6};

    // ---- 闭式谱算例（正确性回归）----
    if (suite == "all" || suite == "closed") {
        {
            auto p = amls::pathGraph(2000);
            auto ref = amls::pathGraphSpectrum(2000);
            for (double eta : etas)
                runCase("path2000", p, ref, nev, eta, 4, false);
        }
        {
            auto p = amls::pathGraph(2000, true);
            auto ref = amls::pathGraphSpectrum(2000, true);
            runCase("path2000-consM", p, ref, nev, 2.0, 4, false);
        }
        {
            auto p = amls::gridGraph2d(64, 64);
            auto ref = amls::gridGraph2dSpectrum(64, 64);
            for (double eta : etas)
                // 叶块 1000（≈32×32 子网格）可解析目标模态波长;默认 500 时 η=2 误差 ~8%
                runCase("grid64x64", p, ref, nev, eta, 4, false, true, 1000);
        }
        {
            auto p = amls::pathGraph(500);
            auto ref = amls::pathGraphSpectrum(500);
            for (int L : levels)
                runCase("path500", p, ref, nev, 2.0, L, false);
        }
    }

    // ---- HB 矩阵（LAPACK 金标准参考）----
    if (suite == "all" || suite == "hb") {
        // 每矩阵:(K, M, 该矩阵的 η 扫描列表;空 = 用全局 η)
        // bcsstk09 按验证方案 §4 预案提高 η(η≤4 时 ε_λ ~ 5–9% > 4%,η=8 达 0.7%)。
        const std::vector<
            std::tuple<std::string, std::string, std::vector<double>>>
            pairs = {
                {"bcsstk01/bcsstk01.mtx", "bcsstm01/bcsstm01.mtx", {}},
                {"bcsstk08/bcsstk08.mtx", "bcsstm08/bcsstm08.mtx", {}},
                {"bcsstk09/bcsstk09.mtx", "bcsstm09/bcsstm09.mtx", {2.0, 3.0, 4.0, 8.0}},
                {"bcsstk13/bcsstk13.mtx", "bcsstm13/bcsstm13.mtx", {}},
            };
        for (const auto& [k, m, etasM] : pairs) {
            try {
                auto p = loadPair(matrixDir + "/" + k, matrixDir + "/" + m, false);
                amls::ReferenceSolver refSolver;
                auto ref = refSolver.smallestEigenvalues(p.K, p.M, nev);
                const int64_t n = p.K.rows();
                const std::vector<double> defEtas =
                    quick ? std::vector<double>{2.0} : etas;
                const auto& useEtas = !etasM.empty() ? etasM : defEtas;
                for (double eta : useEtas)
                    runCase(k.substr(0, k.find('/')), p, ref, nev, eta,
                            n > 2000 ? 4 : 3, false);
            } catch (const std::exception& e) {
                std::cerr << "[verify] 跳过 " << k << ": " << e.what() << std::endl;
            }
        }
    }

    // ---- ND 系列（Spectra 移位反演参考;无 M,用单位阵并注明）----
    if (suite == "all" || suite == "nd") {
        const std::vector<std::string> names = quick
            ? std::vector<std::string>{"nd3k"}
            : std::vector<std::string>{"nd3k", "nd6k"};
        for (const auto& nm : names) {
            try {
                auto p = loadPair(matrixDir + "/" + nm + "/" + nm + ".mtx", "",
                                  true);
                std::cout << "[verify] " << nm << ": 无 M,使用单位阵（验证方案 §2.2 注明）\n";
                amls::ReferenceSolver refSolver;
                auto ref = refSolver.smallestEigenvalues(p.K, p.M, nev);
                for (double eta : (quick ? std::vector<double>{2.0} : etas))
                    runCase(nm, p, ref, nev, eta, 5, false);
            } catch (const std::exception& e) {
                std::cerr << "[verify] 跳过 " << nm << ": " << e.what() << "\n";
            }
        }
    }

    writeReport(outdir);
    int pass = 0;
    for (const auto& r : rows)
        if (r.pass)
            ++pass;
    std::cout << "[verify] 完成: " << pass << "/" << rows.size()
              << " 通过;报告在 " << outdir << "/\n";
    return pass == static_cast<int>(rows.size()) ? 0 : 1;
}
