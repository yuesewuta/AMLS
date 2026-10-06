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
#include <iomanip>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

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
    double massScale = 1.0;    // M ← α·M（f_user 协议的质量缩放系数）
    double fuserHz = 0.0;      // >0: 截止频率 f_user (Hz)，λ_cut = (2πf_user)²
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
        "      --fuser F(Hz) --mass-scale A --leaf-size S --threads T\n"
        "      --no-recover --quiet --out FILE\n";
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
        else if (s == "--fuser") a.fuserHz = std::stod(next());
        else if (s == "--mass-scale") a.massScale = std::stod(next());
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
    Args a = parse(argc, argv);

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

    if (a.massScale <= 0.0)
        usage();
    if (a.massScale != 1.0) {
        pencil.M *= a.massScale;
        std::cout << "[amls_solve] 质量缩放: M <- " << a.massScale
                  << " * M\n";
    }
    if (a.fuserHz > 0.0) {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        // f_user 是"关注频段"上限(其下模态 ≈ nev);分量模态截断取
        // η·f_user(η = thresholdFactor,AMLS 惯例:整体低模态需要高于它的
        // 分量模态张成,直接用 f_user 截断会欠采样 p < nev)。
        a.opt.lambdaCut = a.opt.thresholdFactor * kTwoPi * kTwoPi *
                          a.fuserHz * a.fuserHz;
        std::cout << "[amls_solve] f_user = " << a.fuserHz << " Hz (η="
                  << a.opt.thresholdFactor << ") -> lambda_cut = "
                  << a.opt.lambdaCut << "（分量截断；输出全部降阶模态）\n";
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
        for (int64_t j = 0; j < sol.eigenvalues.size(); ++j)
            out << j + 1 << " " << sol.eigenvalues[j] << "\n";
    }

    std::cout.precision(10);
    std::cout << "# AMLS 求解完成 n=" << pencil.K.rows()
              << " nev=" << a.opt.nev << " p=" << sol.reducedSize
              << " 树节点=" << solver.lastTreeNodes() << "\n";
    std::cout << "# 计时(s): ";
    for (const auto& [k, v] : sol.timing)
        std::cout << k << "=" << v << " ";
    std::cout << "\n";
    for (int64_t j = 0; j < sol.eigenvalues.size() && j < 10; ++j) {
        std::cout << j + 1 << "\t" << sol.eigenvalues[j];
        if (lambdaRef.size() > j) {
            const double err =
                std::abs(sol.eigenvalues[j] - lambdaRef[j]) /
                (std::abs(lambdaRef[j]) > 0 ? std::abs(lambdaRef[j]) : 1.0);
            std::cout << "\tref=" << lambdaRef[j] << "\trelerr=" << err;
        }
        std::cout << "\n";
    }
    if (sol.eigenvalues.size() > 10)
        std::cout << "...（共 " << sol.eigenvalues.size() << " 阶）\n";

    // 残差自洽性（第 1 与最后 1 阶）
    if (sol.eigenvectors.cols() > 0) {
        const double r1 = amls::ErrorMetrics::residual(
            pencil.K, pencil.M, sol.eigenvalues[0], sol.eigenvectors.col(0));
        const double rn = amls::ErrorMetrics::residual(
            pencil.K, pencil.M, sol.eigenvalues[sol.eigenvalues.size() - 1],
            sol.eigenvectors.col(sol.eigenvectors.cols() - 1));
        std::cout << "# 残差: 第1阶=" << r1 << " 末阶=" << rn << "\n";
    }
#ifdef _WIN32
    // 进程峰值工作集（统一 DAG 内存验证的直接读数）
    {
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            std::cout << "# 内存: 峰值工作集=" << std::fixed
                      << std::setprecision(2)
                      << pmc.PeakWorkingSetSize / 1073741824.0 << "GB\n";
    }
#endif
    return 0;
}
