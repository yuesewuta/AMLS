#pragma once
// pme_SolverParameters 的 C++ 复原（184 字节结构）。
//
// 逆向依据 [R]（pams_pseudocode.txt:45522-45586，std_pamsDriver 构造段）：
//   dVar9   = fmax·4π²·fmax + globalShift        = (2π·fmax)² + shift
//   [0x78]  = cutoff1² · dVar9                   —— 子结构主截断 ω1²
//   [0x80]  = lastCutoff² · dVar9                —— 末层/根层截断 ωlast²
//              （ABA_PAMS_LAST_CUTOFF，未设置默认 cutoff1 × 10.0）
//   [0x88]  = (0.6·cutoff1)² · dVar9             —— 第二截断 ωif²
//              （比主截断更紧；消费点 [INF]，推测用于界面/约束模态）
//   [0x90]  = dVar9                              —— ω0²
//   [0x98]  = enhancedCutoff² · dVar9            —— 增强截断 ωenh²
//   [0x48]  = nev；[0xa0] = 3000（内附子空间迭代上限）
//
// 隐藏参数按 Abaqus 语义从环境变量读取（同 pamsDriver/ApplyEnhancedCutoffs）：
//   ABA_PAMS_GLOBALSHIFT         shift 加入 (2π·fmax)²；保留 DOF/声学时强制 0
//   ABA_PAMS_ENHANCEDSHIFTVALUE  覆盖 shift（与 dVar9 取较小者）
//   ABA_PAMS_LAST_CUTOFF         末层截断乘子（默认 cutoff1×10）
//   ABA_PAMS_ENHANCED_CUTOFF     增强乘子（默认 50.0；设置该变量即启用增强）
//   ABA_PAMS_ENHANCED_LEVEL      增强生效的树 level 阈值（默认 3）
//
// level 语义 [R]（pme_core_pseudocode.txt:45683-45713）：子结构 level 从叶向
// 上计数——叶 level=0，内部节点 level = max(子 level) + 1。因此增强截断
// （level ≤ 3）作用于叶及其上方至多 3 层（小子结构保留几乎全部模态）。
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace aams {

constexpr double kFourPiSquared = 39.478417604357354;  // pamsDriver 常量 [R]

struct PamsParameters {
    // ---- std_pamsDriver param_5..param_10（诊断打印名即参数名 [R]）----
    double fmaxHz = 0.0;        // "Maximum frequency"
    double fminHz = 0.0;        // "Mimimum frequency"
    double cutoff1 = 5.0;       // "AMS cutoff1"（默认 5.0，官方文档 [D]）
    double cutoff2 = 1.7;       // "AMS cutoff2"（蒸馏阈值）
    double cutoff3 = 1.1;       // "AMS cutoff3"（初始子空间）
    int64_t nev = 0;            // "Number of requested eigenvalues"
    double globalShift = 0.0;   // ABA_PAMS_GLOBALSHIFT
    bool enhancedEnabled = false;   // ABA_PAMS_ENHANCED_CUTOFF 存在时启用
    double enhancedCutoff = 50.0;   // 默认 50.0（0x40490000 double 解读 [R]）
    int enhancedLevel = 3;          // 叶起计数，level ≤ enhancedLevel 生效
    double lastCutoff = 0.0;        // 0 = 自动 cutoff1 × 10.0
    int64_t subspaceMax = 3000;     // pme_SolverParameters[0xa0]
    int numThreads = 0;

    /// (2π·fmax)² + shift —— pamsDriver dVar9 [R]。
    double dVar9() const {
        return fmaxHz * kFourPiSquared * fmaxHz + globalShift;
    }
    /// 子结构主截断（记录偏移 0x18，ApplyClassicCutoffs 默认分支）。
    double omega1Sq() const { return cutoff1 * cutoff1 * dVar9(); }
    /// 末层/根层截断（param[0x80]；父链被标记时分发）。
    double omegaLastSq() const {
        const double c = lastCutoff > 0.0 ? lastCutoff : cutoff1 * 10.0;
        return c * c * dVar9();
    }
    /// 第二截断（param[0x88] = (0.6·cutoff1)²·dVar9）。
    double omegaInterfaceSq() const {
        constexpr double kIf = 0.6;
        const double c = cutoff1 * kIf;
        return c * c * dVar9();
    }
    /// 增强截断（param[0x98]；仅 enhancedEnabled）。
    double omegaEnhSq() const {
        return enhancedCutoff * enhancedCutoff * dVar9();
    }

    /// 按 Abaqus 语义应用环境变量覆盖（std_pamsDriver + ApplyEnhancedCutoffs）。
    void applyEnv() {
        if (const char* s = std::getenv("ABA_PAMS_GLOBALSHIFT"))
            globalShift = std::atof(s);
        if (const char* s = std::getenv("ABA_PAMS_ENHANCEDSHIFTVALUE")) {
            const double v = std::atof(s);
            // pamsDriver: 与 dVar9 取较小者（[R] 45574-45585）
            globalShift = std::min(v, dVar9());
        }
        if (const char* s = std::getenv("ABA_PAMS_LAST_CUTOFF"))
            lastCutoff = std::atof(s);
        if (std::getenv("ABA_PAMS_ENHANCED_CUTOFF")) {
            enhancedEnabled = true;
            enhancedCutoff =
                std::atof(std::getenv("ABA_PAMS_ENHANCED_CUTOFF"));
        }
        if (const char* s = std::getenv("ABA_PAMS_ENHANCED_LEVEL"))
            enhancedLevel = std::atoi(s);
    }
};

}  // namespace aams
