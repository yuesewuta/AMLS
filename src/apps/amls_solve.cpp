// amls_solve:求解驱动。
// 用法:
//   amls_solve --case path:N [--consistent-mass] [--nev K] [--nlevels L]
//              [--eta H] [--shift S] [--fixed-keep K0] [--quiet]
//   amls_solve --K file.mtx --M file.mtx [同上]
//   amls_solve --K file.mtx                 (M = I,并注明)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "amls/solver.hpp"
#include "core/logging.hpp"
#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "reference/error_metrics.hpp"

namespace {

struct Args {
    std::string kFile, mFile;
    std::string caseType;      // "path" / "grid"
    int64_t caseN = 0, caseNy = 0;
    bool consistentMass = false;
    amls::AmlsOptions opt;
    std::string outFile;
};

void usage() {
    std::cerr <<
        "用法:\n"
        "  amls_solve --case path:N [--consistent-mass] [选项]\n"
        "  amls_solve --case grid:NX,NY [选项]\n"
        "  amls_solve --K K.mtx [--M M.mtx] [选项]\n"
        "选项: --nev K --nlevels L --eta H --shift S --fixed-keep K0\n"
        "      --leaf-size S --threads T --no-recover --quiet --out FILE\n";
    std::exit(2);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() {
            if (++i >= argc)
                usage();
            return std::string(argv[i]);
        };
        if (s == "--K") a.kFile = next();
        else if (s == "--M") a.mFile = next();
        else if (s == "--case") {
            const std::string c = next();
            const auto colon = c.find(':');
            if (colon == std::string::npos)
                usage();
            a.caseType = c.substr(0, colon);
            const std::string rest = c.substr(colon + 1);
            const auto comma = rest.find(',');
            if (comma == std::string::npos) {
                a.caseN = std::stoll(rest);
            } else {
                a.caseN  = std::stoll(rest.substr(0, comma));
                a.caseNy = std::stoll(rest.substr(comma + 1));
            }
        }
        else if (s == "--consistent-mass") a.consistentMass = true;
        else if (s == "--nev") a.opt.nev = std::stoll(next());
        else if (s == "--nlevels") a.opt.nLevels = std::stoi(next());
        else if (s == "--eta") a.opt.thresholdFactor = std::stod(next());
        else if (s == "--shift") a.opt.shift = std::stod(next());
        else if (s == "--fixed-keep") a.opt.fixedKeep = std::stoll(next());
        else if (s == "--leaf-size") a.opt.leafSize = std::stoll(next());
        else if (s == "--threads") a.opt.numThreads = std::stoi(next());
        else if (s == "--no-recover") a.opt.recoverVectors = false;
        else if (s == "--quiet") a.opt.verbosity = 0;
        else if (s == "--log-level") amls::log::setLevel(next());
        else if (s == "--out") a.outFile = next();
        else usage();
    }
    if (a.kFile.empty() && a.caseType.empty())
        usage();
    return a;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);

    amls::PencilPair pencil;
    amls::DefaultTraits::Vector lambdaRef;  // 闭式谱（若可用）

    if (!a.caseType.empty()) {
        if (a.caseType == "path") {
            pencil = amls::pathGraph(a.caseN, a.consistentMass);
            lambdaRef = amls::pathGraphSpectrum(a.caseN, a.consistentMass);
        } else if (a.caseType == "grid") {
            pencil = amls::gridGraph2d(a.caseN, a.caseNy);
            lambdaRef = amls::gridGraph2dSpectrum(a.caseN, a.caseNy);
        } else if (a.caseType == "grid3d") {
            pencil = amls::gridGraph3d(a.caseN, a.caseN, a.caseN);
            lambdaRef = amls::gridGraph3dSpectrum(a.caseN, a.caseN, a.caseN);
        } else {
            std::cerr << "未知算例类型: " << a.caseType << "\n";
            usage();
        }
    } else {
        pencil.K = amls::readMatrixMarket<amls::DefaultTraits>(a.kFile);
        if (!a.mFile.empty())
            pencil.M = amls::readMatrixMarket<amls::DefaultTraits>(a.mFile);
        else {
            pencil.M.resize(pencil.K.rows(), pencil.K.cols());
            pencil.M.setIdentity();
            std::cerr << "[amls_solve] 注意: 未提供 M,使用单位阵\n";
        }
    }

    amls::Pencil<amls::DefaultTraits> km{pencil.K, pencil.M};
    amls::AmlsSolver solver;
    amls::AmlsSolver::Solution sol;
    try {
        sol = solver.solve(km, a.opt);
    } catch (const std::exception& e) {
        std::cerr << "[amls_solve] 求解失败: " << e.what() << "\n";
        return 1;
    }

    // 输出
    if (!a.outFile.empty()) {
        std::ofstream out(a.outFile);
        out.precision(17);
        for (int64_t j = 0; j < sol.theta.size(); ++j)
            out << j + 1 << " " << sol.theta[j] << "\n";
    }

    std::cout.precision(10);
    std::cout << "# AMLS 求解完成 n=" << pencil.K.rows()
              << " nev=" << a.opt.nev << " p=" << sol.p
              << " 树节点=" << solver.lastTreeNodes() << "\n";
    std::cout << "# 计时(s): ";
    for (const auto& [k, v] : sol.timing)
        std::cout << k << "=" << v << " ";
    std::cout << "\n";
    for (int64_t j = 0; j < sol.theta.size() && j < 10; ++j) {
        std::cout << j + 1 << "\t" << sol.theta[j];
        if (lambdaRef.size() > j) {
            const double err =
                std::abs(sol.theta[j] - lambdaRef[j]) /
                (std::abs(lambdaRef[j]) > 0 ? std::abs(lambdaRef[j]) : 1.0);
            std::cout << "\tref=" << lambdaRef[j] << "\trelerr=" << err;
        }
        std::cout << "\n";
    }
    if (sol.theta.size() > 10)
        std::cout << "...（共 " << sol.theta.size() << " 阶）\n";

    // 残差自洽性（第 1 与最后 1 阶）
    if (sol.Z.cols() > 0) {
        const double r1 = amls::ErrorMetrics::residual(
            pencil.K, pencil.M, sol.theta[0], sol.Z.col(0));
        const double rn = amls::ErrorMetrics::residual(
            pencil.K, pencil.M, sol.theta[sol.theta.size() - 1],
            sol.Z.col(sol.Z.cols() - 1));
        std::cout << "# 残差: 第1阶=" << r1 << " 末阶=" << rn << "\n";
    }
    return 0;
}
