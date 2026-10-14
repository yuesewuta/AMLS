// amls_solve:求解驱动。
// 用法见 --help。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include "amls/solver.hpp"
#include "core/logging.hpp"
#include "io/case_generator.hpp"
#include "io/matrix_market.hpp"
#include "reference/error_metrics.hpp"

namespace {

// GetProcessTimes reports kernel + user CPU time summed over all process threads.
// -1 means unavailable (including non-Windows builds); wall time remains available.
double processCpuSeconds() {
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER k{}, u{};
        k.LowPart = kernel.dwLowDateTime;
        k.HighPart = kernel.dwHighDateTime;
        u.LowPart = user.dwLowDateTime;
        u.HighPart = user.dwHighDateTime;
        return static_cast<double>(k.QuadPart) * 1e-7 +
               static_cast<double>(u.QuadPart) * 1e-7;
    }
#endif
    return -1.0;
}

void stageDetail(const std::string& detail) {
    std::cerr << "[amls_solve] " << detail << '\n' << std::flush;
}

class StageTimer {
public:
    explicit StageTimer(const char* stage)
        : stage_(stage), start_(std::chrono::steady_clock::now()),
          cpuStart_(processCpuSeconds()), exceptions_(std::uncaught_exceptions()) {
        stageDetail(std::string("stage=") + stage_ + " begin");
    }
    ~StageTimer() {
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
        const double cpuEnd = processCpuSeconds();
        const double cpu = cpuStart_ >= 0.0 && cpuEnd >= 0.0
                               ? cpuEnd - cpuStart_ : -1.0;
        const auto flags = std::cerr.flags();
        const auto precision = std::cerr.precision();
        std::cerr << "[amls_solve] stage=" << stage_
                  << (std::uncaught_exceptions() > exceptions_ ? " failed" : " done")
                  << std::fixed << std::setprecision(6)
                  << " wall_seconds=" << wall << " cpu_seconds=" << cpu
                  << '\n' << std::flush;
        std::cerr.flags(flags);
        std::cerr.precision(precision);
    }
private:
    const char* stage_;
    std::chrono::steady_clock::time_point start_;
    double cpuStart_;
    int exceptions_;
};

struct Args {
    std::string kFile, mFile;
    std::string caseType;      // "path" / "grid" / "grid3d"
    int64_t caseN = 0, caseNy = 0;
    bool consistentMass = false;
    double massScale = 1.0;    // M ← α·M（f_user 协议的质量缩放系数）
    double fuserHz = 0.0;      // >0: 截止频率 f_user (Hz)，λ_cut = (2πf_user)²
    amls::AmlsOptions opt;
    std::string outFile;
};

void usage(int exitCode = 2) {
    std::cerr <<
        "用法:\n"
        "  amls_solve --case path:N [--consistent-mass] [选项]\n"
        "  amls_solve --case grid:NX,NY [选项]\n"
        "  amls_solve --case grid3d:N [选项] (单个 N，三维均为 N)\n"
        "  amls_solve --K K.mtx [--M M.mtx] [选项]\n"
        "选项: --nev K --nlevels L --eta H\n"
        "      --min-freq F0 --max-freq F1 --fuser F(Hz)\n"
        "      --mass-scale A --leaf-size S --threads T\n"
        "      --partition bisect|nd|external|supernode|rea --tree-file FILE\n"
        "      --reduced pku|rea  (REA 缩聚特征阶段: 蒸馏+子空间迭代+Ritz)\n"
        "      --cutoff2 C --cutoff3 C --shift S (REA 蒸馏参数, <=0 跟随 --eta)\n"
        "      --rea-block-dofs B --rea-collapse-dofs C --rea-leaf-aggr-dofs L\n"
        "      --no-subspace-iter  (REA: 关闭子空间迭代, 直接广义 Ritz)\n"
        "      --no-recover --quiet --out FILE\n"
        "      --help\n" << std::flush;
    std::exit(exitCode);
}

int64_t caseDimension(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("算例尺寸必须为正整数: " + value);
    const int64_t n = std::stoll(value);
    if (n <= 0)
        throw std::invalid_argument("算例尺寸必须为正整数: " + value);
    return n;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() {
            if (++i >= argc)
                throw std::invalid_argument("缺少参数值: " + s);
            return std::string(argv[i]);
        };
        if (s == "--help" || s == "-h") usage(0);
        else if (s == "--K") a.kFile = next();
        else if (s == "--M") a.mFile = next();
        else if (s == "--case") {
            const std::string c = next();
            const auto colon = c.find(':');
            if (colon == std::string::npos)
                throw std::invalid_argument("算例格式应为 类型:尺寸");
            a.caseType = c.substr(0, colon);
            const std::string rest = c.substr(colon + 1);
            const auto comma = rest.find(',');
            if (a.caseType == "grid") {
                if (comma == std::string::npos || rest.find(',', comma + 1) != std::string::npos)
                    throw std::invalid_argument("grid 仅接受 NX,NY 两个尺寸");
                a.caseN = caseDimension(rest.substr(0, comma));
                a.caseNy = caseDimension(rest.substr(comma + 1));
            } else if (a.caseType == "path" || a.caseType == "grid3d") {
                if (comma != std::string::npos)
                    throw std::invalid_argument(a.caseType + " 仅接受单个尺寸 N");
                a.caseN = caseDimension(rest);
                a.caseNy = 0;
            } else {
                throw std::invalid_argument("未知算例类型: " + a.caseType);
            }
        }
        else if (s == "--consistent-mass") a.consistentMass = true;
        else if (s == "--nev") a.opt.extractOrder = std::stoll(next());
        else if (s == "--nlevels") a.opt.nLevels = std::stoi(next());
        else if (s == "--eta") a.opt.gainCoefficient = std::stod(next());
        else if (s == "--min-freq") a.opt.minFreq = std::stod(next());
        else if (s == "--max-freq") a.opt.maxFreq = std::stod(next());
        else if (s == "--fuser") a.fuserHz = std::stod(next());
        else if (s == "--mass-scale") a.massScale = std::stod(next());
        else if (s == "--leaf-size") a.opt.leafSize = std::stoll(next());
        else if (s == "--partition") a.opt.partitionMethod = next();
        else if (s == "--tree-file") a.opt.externalTreeFile = next();
        else if (s == "--max-block-dofs") a.opt.maxBlockDofs = std::stoll(next());
        else if (s == "--reduced") a.opt.reducedSolver = next();
        else if (s == "--cutoff2") a.opt.cutoff2 = std::stod(next());
        else if (s == "--cutoff3") a.opt.cutoff3 = std::stod(next());
        else if (s == "--shift") a.opt.globalShift = std::stod(next());
        else if (s == "--rea-block-dofs") a.opt.reaMaxBlockDofs = std::stoll(next());
        else if (s == "--rea-collapse-dofs") a.opt.reaCollapseDofs = std::stoll(next());
        else if (s == "--rea-leaf-aggr-dofs") a.opt.reaLeafAggrDofs = std::stoll(next());
        else if (s == "--no-subspace-iter") a.opt.subspaceIteration = false;
        else if (s == "--threads") a.opt.numThreads = std::stoi(next());
        else if (s == "--no-recover") a.opt.recoverVectors = false;
        else if (s == "--quiet") a.opt.verbosity = 0;
        else if (s == "--log-level") amls::log::setLevel(next());
        else if (s == "--out") a.outFile = next();
        else throw std::invalid_argument("未知选项: " + s);
    }
    if (a.kFile.empty() && a.caseType.empty())
        throw std::invalid_argument("需要 --K 或 --case");
    if (a.opt.partitionMethod != "bisect" && a.opt.partitionMethod != "nd" &&
        a.opt.partitionMethod != "external" && a.opt.partitionMethod != "supernode" &&
        a.opt.partitionMethod != "rea")
        throw std::invalid_argument("--partition 仅接受 bisect、nd、external、supernode 或 rea");
    if (a.opt.reducedSolver != "pku" && a.opt.reducedSolver != "rea")
        throw std::invalid_argument("--reduced 仅接受 pku 或 rea");
    if (a.opt.maxBlockDofs < 1)
        throw std::invalid_argument("--max-block-dofs 必须为正");
    if (!std::isfinite(a.massScale) || a.massScale <= 0.0)
        throw std::invalid_argument("--mass-scale 必须为有限正数");
    if (a.fuserHz > 0.0) {
        a.opt.minFreq = 0.0;
        a.opt.maxFreq = a.fuserHz;
    }
    if (a.opt.maxFreq <= 0.0)
        throw std::invalid_argument("必须提供 --fuser F 或 --max-freq F1（关注频段上界，Hz）");
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        // Includes parsing, input, preparation (including pencil copies), solve,
        // result checks and output. It excludes process startup and shutdown.
        StageTimer endToEnd("end_to_end");
        Args a = parse(argc, argv);
        const int requestedThreads = a.opt.numThreads;
        if (a.opt.numThreads <= 0)
            a.opt.numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        std::cerr << "[amls_solve] config partition=" << a.opt.partitionMethod
                  << " leaf=" << a.opt.leafSize << " levels=" << a.opt.nLevels
                  << " threads=" << a.opt.numThreads
                  << " requested_threads=" << requestedThreads
                  << " levels_policy=" << (a.opt.partitionMethod == "nd" ? "adaptive" : "bounded")
                  << '\n' << std::flush;

        amls::PencilPair pencil;
        amls::DefaultTraits::Vector lambdaRef;
        {
            StageTimer read("read");
            if (!a.caseType.empty()) {
                stageDetail("generate case=" + a.caseType);
                if (a.caseType == "path")
                    pencil = amls::pathGraph(a.caseN, a.consistentMass);
                else if (a.caseType == "grid")
                    pencil = amls::gridGraph2d(a.caseN, a.caseNy);
                else
                    pencil = amls::gridGraph3d(a.caseN, a.caseN, a.caseN);
            } else {
                stageDetail("read K=" + a.kFile);
                pencil.K = amls::readMatrixMarket<amls::DefaultTraits>(a.kFile);
                if (!a.mFile.empty()) {
                    stageDetail("read M=" + a.mFile);
                    pencil.M = amls::readMatrixMarket<amls::DefaultTraits>(a.mFile);
                }
            }
        }

        amls::Pencil<amls::DefaultTraits> km;
        {
            StageTimer prepare("input_prepare");
            if (a.caseType == "path")
                lambdaRef = amls::pathGraphSpectrum(a.caseN, a.consistentMass);
            else if (a.caseType == "grid")
                lambdaRef = amls::gridGraph2dSpectrum(a.caseN, a.caseNy);
            else if (a.caseType == "grid3d")
                lambdaRef = amls::gridGraph3dSpectrum(a.caseN, a.caseN, a.caseN);
            else if (a.mFile.empty()) {
                pencil.M.resize(pencil.K.rows(), pencil.K.cols());
                pencil.M.setIdentity();
                stageDetail("注意: 未提供 M,使用单位阵");
            }
            if (a.massScale != 1.0) {
                pencil.M *= a.massScale;
                lambdaRef /= a.massScale;
                std::cout << "[amls_solve] 质量缩放: M <- " << a.massScale << " * M\n";
            }
            km = amls::Pencil<amls::DefaultTraits>{pencil.K, pencil.M};
        }

        amls::AmlsSolver solver;
        amls::AmlsSolver::Solution sol;
        {
            StageTimer solve("solve");
            sol = solver.solve(km, a.opt);
        }

        {
            StageTimer check("result_check");
            std::cout.precision(10);
            std::cout << "# AMLS 求解完成 n=" << pencil.K.rows()
                      << " maxFreq=" << a.opt.maxFreq << "Hz"
                      << " extractOrder=" << a.opt.extractOrder
                      << " p=" << sol.reducedSize
                      << " 输出=" << sol.eigenvalues.size()
                      << " 树节点=" << solver.lastTreeNodes() << "\n";
            std::cout << "# 计时(s): ";
            for (const auto& [k, v] : sol.timing)
                std::cout << k << "=" << v << " ";
            std::cout << "\n";
            for (int64_t j = 0; j < sol.eigenvalues.size() && j < 10; ++j) {
                std::cout << j + 1 << "\t" << sol.eigenvalues[j];
                if (lambdaRef.size() > j) {
                    const double err = std::abs(sol.eigenvalues[j] - lambdaRef[j]) /
                        (std::abs(lambdaRef[j]) > 0 ? std::abs(lambdaRef[j]) : 1.0);
                    std::cout << "\tref=" << lambdaRef[j] << "\trelerr=" << err;
                }
                std::cout << "\n";
            }
            if (sol.eigenvalues.size() > 10)
                std::cout << "...（共 " << sol.eigenvalues.size() << " 阶）\n";

            // 残差自洽性（第 1 与最后 1 阶）。
            if (sol.eigenvalues.size() > 0 && sol.eigenvectors.cols() > 0) {
                const double r1 = amls::ErrorMetrics::residual(
                    pencil.K, pencil.M, sol.eigenvalues[0], sol.eigenvectors.col(0));
                const double rn = amls::ErrorMetrics::residual(
                    pencil.K, pencil.M, sol.eigenvalues[sol.eigenvalues.size() - 1],
                    sol.eigenvectors.col(sol.eigenvectors.cols() - 1));
                const int64_t j200 = std::min<int64_t>(199, sol.eigenvalues.size() - 1);
                const double r200 = amls::ErrorMetrics::residual(
                    pencil.K, pencil.M, sol.eigenvalues[j200], sol.eigenvectors.col(j200));
                std::cout << "# 残差: 第1阶=" << r1 << " 第" << j200 + 1
                          << "阶=" << r200 << " 末阶=" << rn << "\n";
            }
        }

        if (!a.outFile.empty()) {
            stageDetail("write out=" + a.outFile);
            std::ofstream out(a.outFile);
            if (!out.is_open())
                throw std::runtime_error("无法打开输出文件: " + a.outFile);
            out.precision(17);
            for (int64_t j = 0; j < sol.eigenvalues.size(); ++j)
                out << j + 1 << " " << sol.eigenvalues[j] << "\n";
            out.close();
            if (!out)
                throw std::runtime_error("写入输出文件失败: " + a.outFile);
        }
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            std::cout << "# 内存: 峰值工作集=" << std::fixed << std::setprecision(2)
                      << pmc.PeakWorkingSetSize / 1073741824.0 << "GB\n";
#endif
        std::cout << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[amls_solve] 失败: " << e.what() << '\n' << std::flush;
        return 1;
    }
}
