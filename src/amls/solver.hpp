#pragma once
// AMLS 求解器协调器 —— PKU 列压缩架构编排层：
//   step1 MetisPartitioner      剖分 + 重排（树索引重映射到重排坐标）
//   step2 OriginalBlocks → GaussEliminationModule（消去）→ TempBlockManager
//         （缩聚），由 NodeTaskExecutor 双线程池 11 任务 DAG 驱动
//   step3 子空间特征问题：EigenPairSaver/SubspaceMassSaver 组装 + dsygvx
//         频率区间 [(2π·minFreq)², (2π·maxFreq)²]
//   step4 恢复（按深度分层拉取式 X_i = Φ_i·z_i + Σ_a Ψ_{i,a}·X_a）+ 反排列
//         （Φ M-正交 + 子空间解 B-正交 ⇒ 输出自动 M-归一，无事后归一化）
#include <cstdint>
#include <map>
#include <string>

#include "core/types.hpp"

namespace amls {

class SeparatorTree;

struct AmlsOptions {
    int     nLevels           = 6;
    int64_t leafSize          = 500;
    std::string partitionMethod = "nd";   // "nd" 单次 NodeND（默认）| "bisect" 递归二分 | "external" 外部树 | "supernode" 超节点树 | "rea" Abaqus AMS 风格树
    std::string externalTreeFile;         // partitionMethod=="external" 时: 树文件 / "perm:" 排列文件
    int64_t maxBlockDofs = 1000;          // partitionMethod=="supernode" 时每块 DOF 上限
    double  minFreq           = 0.0;    // 关注频段下界（Hz，含；0 = 含刚体/负谱）
    double  maxFreq           = 0.0;    // 关注频段上界（Hz，必填 >0）
    double  gainCoefficient   = 2.0;    // γ：分量模态局部截断 (2π·γ·maxFreq)²（Abaqus cutoff1）
    int64_t extractOrder      = 0;      // >0 最终输出截到前 extractOrder 阶；0 = 频段内全部
    int64_t denseProjectedMax = 5000;   // P12 切换点（仅告警）
    int     numThreads        = 0;      // <=0 = 硬件并发数
    bool    recoverVectors    = true;   // step4 恢复特征向量（false 时只算特征值）
    int     verbosity         = 1;
    // ---- REA（Abaqus AMS 语义）----
    std::string reducedSolver = "pku";  // "pku" 直接 dsygvx | "rea" 蒸馏+一次子空间迭代+Ritz
    int64_t reaMaxBlockDofs = 4096;   // 块 DOF 上限（ABA_PAMS_UPPER_CUTOFF=4096）
    int64_t reaCollapseDofs = 120;    // Collapse2 合并上限（LEAF_SN_LIMIT=120）
    int64_t reaLeafAggrDofs = 20;     // AggregateLeaves 上限（LEAF_AGGR_LIMIT=20）
    int     reaDofsPerNode = 0;   // 网格节点收缩宽度（0=自动 3，1=逐 DOF）
    double  cutoff2 = 0.0;    // 蒸馏阈值倍数（param_8；<=0 = 跟随 gainCoefficient）
    double  cutoff3 = 0.0;    // 初始子空间倍数（param_9；<=0 = 跟随 gainCoefficient）
    double  globalShift = 0.0;  // 全局 shift（ABA_PAMS_GLOBALSHIFT）
    bool    subspaceIteration = true;  // 一次子空间迭代（Abaqus 默认开）
};

class AmlsSolver {
public:
    using PencilT = Pencil<DefaultTraits>;
    using Vector  = DefaultTraits::Vector;
    using DenseMatrix = DefaultTraits::DenseMatrix;

    struct Solution {
        Vector eigenvalues;       // 近似特征值（升序，频段内模态）
        DenseMatrix eigenvectors; // M-归一近似特征向量（n × m；recoverVectors=false 时为空）
        int64_t reducedSize = 0;  // 降阶问题规模 p
        std::map<std::string, double> timing;  // 秒
    };

    Solution solve(const PencilT& KM, const AmlsOptions& opt);

    /// 最近一次求解的分割节点树节点数（诊断）。
    int64_t lastTreeNodes() const { return lastTreeNodes_; }

private:
    int64_t lastTreeNodes_ = 0;
};

} // namespace amls
