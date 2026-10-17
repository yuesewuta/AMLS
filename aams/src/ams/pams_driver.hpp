#pragma once
// std_pamsDriver 的 C++ 复原 —— AAMS 主驱动器。
//
// 逆向依据 [R]（docs/abaqus_ams/AMS求解流程与参数机制逆向分析.md §2）：
//   std_pamsDriver:
//     [1] METIS ORDERING          → aams::buildAbaqusTree（NodeND 在网格节点图
//                                   上排序 + BuildTreeAfterMetis etree +
//                                   CreateFundamentalSuperNodes + Collapse2/
//                                   AggregateLeaves 聚合）
//     [4] SYMBOLIC FACTOR         → 同上（符号分析内嵌）
//     [7] pme_SolverParameters    → aams::PamsParameters（精确 cutoff 公式）
//     [9] pme_SMPAmsEigenSolver::Solve 三大阶段：
//         vtable[0x10] Reduction Phase
//           —— 子结构生成 + 局部特征（cutoff1 截断）+ 约束模态 Ψ + 缩聚
//              （GaussEliminationModule/TempBlockManager 列压缩实现）
//         vtable[0x18] Reduced Eigensolution Phase
//           —— pme_SmpReducedEigenSolver：蒸馏 → 一次子空间迭代 → Ritz
//              （aams::distillRitz，cutoff2/cutoff3 定义起始子空间）
//         vtable[0x20] Recovery Phase
//           —— pme_SmpAmsRecovery：沿树 Ψ 传播恢复全模态 + M 归一
//   [10] 特征值表打印（MODE / EIGENVALUE / FREQUENCY，同 std_PrintEigenvalueTable）
//
// 计时阶段名对齐 Abaqus "PAMS Timing Data" 三行 + 细分阶段。
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ams/solver_parameters.hpp"
#include "core/types.hpp"

namespace aams {

struct AmsOptions {
    // ---- AMS 频率参数（std_pamsDriver param_5..param_10 语义）----
    double fmaxHz = 0.0;    // 必填 >0
    double fminHz = 0.0;    // 特征值提取下界（0 = 含刚体/负谱）
    double cutoff1 = 5.0;   // 子结构局部截断乘子
    double cutoff2 = 1.7;   // 蒸馏阈值乘子
    double cutoff3 = 1.1;   // 初始子空间乘子
    int64_t nev = 0;        // >0: 输出截到前 nev 阶（Abaqus 语义：不影响提取）
    int64_t subspaceMax = 0; // >0: 覆盖蒸馏子空间规模上限（0=par 默认 3000）
    int numThreads = 0;     // <=0 = 硬件并发数

    // ---- 隐藏参数（默认 false 时由 applyEnv 按 ABA_PAMS_* 读取）----
    bool readEnv = true;
    double globalShift = 0.0;
    bool enhancedEnabled = false;
    double enhancedCutoff = 50.0;
    int enhancedLevel = 3;
    double lastCutoff = 0.0;    // 0 = cutoff1×10

    // ---- 树构造（AbaqusTreeOptions 透传）----
    int     dofsPerNode = 0;          // 0=自动 3，1=逐 DOF
    int64_t maxBlockDofs = 4096;      // UPPER_CUTOFF
    int64_t collapseMaxDofs = 120;    // LEAF_SN_LIMIT
    int64_t leafAggrMaxDofs = 20;     // LEAF_AGGR_LIMIT
    int64_t minBlockDofs = 80;        // MIN_SN_SIZE
    int     collapseMode = 2;
    bool    aggregateLeaves = true;

    // ---- 缩聚特征阶段 ----
    // "direct"  —— pme 缩聚特征求解的 DSYEVR 直接路径 [R]（StaCore
    //              DSYEVRCoreEigenSolver）：对 (K̃,M̃) 频段区间稠密求解。
    //              与蒸馏路径同为 Abaqus 内部的两种选择；默认 direct。
    // "distill" —— pme_SmpReducedEigenSolver 蒸馏+一次子空间迭代+Ritz 的
    //              稠密适配（对整体缩减笔一次性蒸馏；Abaqus 实现为按子树
    //              分组蒸馏，适配版起始子空间较穷，高阶精度受限）。
    std::string reducedMethod = "direct";
    bool subspaceIteration = true;    // 一次子空间迭代（Abaqus 默认开）
    bool recoverVectors = true;       // Recovery Phase（false 只算特征值）
    int  verbosity = 1;
};

class AmsSolver {
public:
    using PencilT = Pencil<DefaultTraits>;
    using Vector  = DefaultTraits::Vector;
    using DenseMatrix = DefaultTraits::DenseMatrix;
    struct Solution {
        DefaultTraits::Vector eigenvalues;        // λ = ω² 升序
        DefaultTraits::DenseMatrix eigenvectors;  // n×m，M-归一（原始坐标）
        int64_t reducedSize = 0;                  // 缩聚子空间规模 p
        PamsParameters params;                    // 实际生效的求解参数
        std::map<std::string, double> timing;     // 秒（PAMS Timing Data）
    };

    /// KM.K/M 不被修改。
    Solution solve(const Pencil<DefaultTraits>& KM, const AmsOptions& opt);
};

}  // namespace aams
