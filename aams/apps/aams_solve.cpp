// aams_solve: AAMS（Abaqus AMS 逆向复原）求解驱动。
//
// 参数语义对齐 Abaqus/Standard *FREQUENCY, EIGENSOLVER=AMS：
//   nev, fmin, fmax [, cutoff1, cutoff2, cutoff3]（默认 5.0 / 1.7 / 1.1）
//   λ_cut = (2π·cutoff1·fmax)² + shift（std_pamsDriver 反编译公式 [R]）
// 隐藏参数经 ABA_PAMS_* 环境变量（与 Abaqus 同名同义）。
#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "ams/pams_driver.hpp"
#include "io/matrix_market.hpp"
#include "reference/error_metrics.hpp"

namespace {

void usage(int exitCode = 2) {
    std::cerr <<
        "用法:\n"
        "  aams_solve --K K.mtx --M M.mtx --fmax F1 [选项]\n"
        "选项:\n"
        "  --nev N            请求模态数（输出截断，不影响提取）\n"
        "  --fmin F0 --fmax F1  关注频段 (Hz)；fmax 必填\n"
        "  --cutoff1 C        子结构局部截断乘子（默认 5.0）\n"
        "  --cutoff2 C        蒸馏阈值乘子（默认 1.7）\n"
        "  --cutoff3 C        初始子空间乘子（默认 1.1）\n"
        "  --shift S          全局 shift（= ABA_PAMS_GLOBALSHIFT）\n"
        "  --enhanced         启用增强截断（叶起 ≤ enhanced-level 层）\n"
        "  --enhanced-cutoff C --enhanced-level L\n"
        "  --last-cutoff C    根层截断乘子（默认 cutoff1×10）\n"
        "  --dofs-per-node N  网格节点收缩宽度: 0=自动 3, 1=逐 DOF\n"
        "  --max-block-dofs B --collapse-dofs C --leaf-aggr-dofs L\n"
        "  --reduced direct|distill  缩聚特征路径（默认 direct=DSYEVR 直接；\n"
        "                            distill=蒸馏+一次子空间迭代+Ritz）\n"
        "  --no-subspace-iter --no-recover --quiet\n"
        "  --threads T --out FILE --help\n"
        "环境变量: ABA_PAMS_GLOBALSHIFT / ABA_PAMS_LAST_CUTOFF /\n"
        "  ABA_PAMS_ENHANCED_CUTOFF / ABA_PAMS_ENHANCED_LEVEL（未显式给参时生效）\n"
        << std::flush;
    std::exit(exitCode);
}

struct Args {
    std::string kFile, mFile, outFile;
    aams::AmsOptions opt;
    bool enhancedSet = false, shiftSet = false, lastSet = false;
};

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
        else if (s == "--nev") a.opt.nev = std::stoll(next());
        else if (s == "--subspace-max") a.opt.subspaceMax = std::stoll(next());
        else if (s == "--fmin") a.opt.fminHz = std::stod(next());
        else if (s == "--fmax") a.opt.fmaxHz = std::stod(next());
        else if (s == "--cutoff1") a.opt.cutoff1 = std::stod(next());
        else if (s == "--cutoff2") a.opt.cutoff2 = std::stod(next());
        else if (s == "--cutoff3") a.opt.cutoff3 = std::stod(next());
        else if (s == "--shift") { a.opt.globalShift = std::stod(next()); a.shiftSet = true; }
        else if (s == "--enhanced") { a.opt.enhancedEnabled = true; a.enhancedSet = true; }
        else if (s == "--enhanced-cutoff") { a.opt.enhancedCutoff = std::stod(next()); a.enhancedSet = true; }
        else if (s == "--enhanced-level") { a.opt.enhancedLevel = std::stoi(next()); a.enhancedSet = true; }
        else if (s == "--last-cutoff") { a.opt.lastCutoff = std::stod(next()); a.lastSet = true; }
        else if (s == "--dofs-per-node") a.opt.dofsPerNode = std::stoi(next());
        else if (s == "--max-block-dofs") a.opt.maxBlockDofs = std::stoll(next());
        else if (s == "--collapse-dofs") a.opt.collapseMaxDofs = std::stoll(next());
        else if (s == "--leaf-aggr-dofs") a.opt.leafAggrMaxDofs = std::stoll(next());
        else if (s == "--no-subspace-iter") a.opt.subspaceIteration = false;
        else if (s == "--reduced") a.opt.reducedMethod = next();
        else if (s == "--no-recover") a.opt.recoverVectors = false;
        else if (s == "--threads") a.opt.numThreads = std::stoi(next());
        else if (s == "--quiet") a.opt.verbosity = 0;
        else if (s == "--out") a.outFile = next();
        else if (s == "--no-env") a.opt.readEnv = false;
        else throw std::invalid_argument("未知选项: " + s);
    }
    if (a.kFile.empty())
        throw std::invalid_argument("需要 --K");
    if (a.opt.fmaxHz <= 0.0)
        throw std::invalid_argument("必须提供 --fmax F1（Hz）");
    // 显式指定的参数不再被环境变量覆盖。
    if (a.enhancedSet) a.opt.readEnv = false;   // 简化：显式增强设置时跳过 env
    if (a.shiftSet || a.lastSet) a.opt.readEnv = false;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        Args a = parse(argc, argv);
        if (a.opt.numThreads <= 0)
            a.opt.numThreads =
                std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

        aams::Pencil<aams::DefaultTraits> km;
        const bool prof = std::getenv("AAMS_PROFILE") != nullptr;
        auto clockNow = [] { return std::chrono::steady_clock::now(); };
        auto tp = clockNow();
        km.K = aams::readMatrixMarket<aams::DefaultTraits>(a.kFile);
        const double parseK =
            std::chrono::duration<double>(clockNow() - tp).count();
        double parseM = 0.0;
        if (!a.mFile.empty()) {
            tp = clockNow();
            km.M = aams::readMatrixMarket<aams::DefaultTraits>(a.mFile);
            parseM = std::chrono::duration<double>(clockNow() - tp).count();
        } else {
            km.M.resize(km.K.rows(), km.K.cols());
            km.M.setIdentity();
            std::cerr << "[aams_solve] 未提供 M，使用单位阵\n";
        }
        if (prof)
            std::cerr << "[prof] parse K=" << parseK << "s M=" << parseM
                      << "s\n"
                      << std::flush;

        aams::AmsSolver solver;
        aams::AmsSolver::Solution sol = solver.solve(km, a.opt);

        // ---- 特征值表（对齐 std_PrintEigenvalueTable 三列格式）----
        std::cout.precision(10);
        std::cout << "\n\n M O D E   E X T R A C T I O N\n\n"
                  << " MODE    EIGENVALUE        FREQUENCY\n"
                  << "        (RAD/TIME)      (CYCLES/TIME)\n\n";
        const int64_t m = sol.eigenvalues.size();
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        for (int64_t j = 0; j < m; ++j) {
            const double lam = sol.eigenvalues[j];
            const double freq = lam > 0.0 ? std::sqrt(lam) / kTwoPi : 0.0;
            std::cout << std::setw(6) << j + 1 << "  " << std::setw(13)
                      << lam << "  " << std::setw(12) << freq << "\n";
        }

        // 残差自检（首/末阶）。
        if (sol.eigenvectors.cols() > 0) {
            const double r1 = aams::ErrorMetrics::residual(
                km.K, km.M, sol.eigenvalues[0], sol.eigenvectors.col(0));
            const double rn = aams::ErrorMetrics::residual(
                km.K, km.M, sol.eigenvalues[m - 1],
                sol.eigenvectors.col(m - 1));
            std::cout << "# residual: first=" << r1 << " last=" << rn << "\n";
        }

        std::cout << "\n P A M S   T I M I N G   D A T A\n";
        for (const auto& [k, v] : sol.timing)
            std::cout << "  " << k << " = " << std::fixed
                      << std::setprecision(3) << v << " s\n";
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "  TOTAL = " << wall << " s\n";
        std::cout << std::flush;

        if (!a.outFile.empty()) {
            std::ofstream out(a.outFile);
            if (!out.is_open())
                throw std::runtime_error("无法打开输出文件: " + a.outFile);
            out.precision(17);
            for (int64_t j = 0; j < m; ++j)
                out << j + 1 << " " << sol.eigenvalues[j] << "\n";
        }
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            std::cout << "# peak working set = "
                      << std::fixed << std::setprecision(2)
                      << pmc.PeakWorkingSetSize / 1073741824.0 << " GB\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[aams_solve] 失败: " << e.what() << '\n' << std::flush;
        return 1;
    }
}
