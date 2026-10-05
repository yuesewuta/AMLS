#pragma once
// AMLS 求解器协调器（5 类架构的编排层；Hyun 2020 / OOC 论文四阶段流水线）：
//   step1  MetisPartitioner      剖分 + 重排
//   step2a GaussianElimination   高斯消去（式 5–8 的消去半边）
//   step2b SubspaceCondensation  子空间缩聚（式 16–17；向 GE/EigenpairSolver get 值）
//   step3  EigenpairSolver       降阶广义特征问题（式 21）
//   step4  SubspaceCondensation::backtransform  特征向量恢复（可选，数据不释放）
#include <cstdint>
#include <map>
#include <string>

#include "core/types.hpp"

namespace amls {

class SeparatorTree;

struct AmlsOptions {
    int     nLevels           = 6;
    int64_t leafSize          = 500;
    double  thresholdFactor   = 2.0;    // η（ThresholdTruncation）
    int64_t fixedKeep         = 0;      // >0 时单趟固定截断（否则两趟：估计 λ̂ 后阈值截断）
    int64_t nev               = 100;
    int64_t denseProjectedMax = 5000;   // P12 切换点
    double  shift             = 0.0;    // K ← K + shift·M（半正定 K 的零谱保护）
    bool    twoPass           = true;   // 两趟 λ̂ 估计
    int     numThreads        = 0;      // <=0 = 硬件并发数（局部谱阶段线程池）
    bool    recoverVectors    = true;   // step4 恢复特征向量（false 时只算特征值）
    int     verbosity         = 1;
};

class AmlsSolver {
public:
    using PencilT = Pencil<DefaultTraits>;
    using Vector  = DefaultTraits::Vector;
    using DenseMatrix = DefaultTraits::DenseMatrix;

    struct Solution {
        Vector theta;       // 近似特征值（升序，nev 个）
        DenseMatrix Z;      // M-归一近似特征向量（n × nev；recoverVectors=false 时为空）
        int64_t p = 0;      // 降阶问题规模
        std::map<std::string, double> timing;  // 秒
    };

    Solution solve(const PencilT& KM, const AmlsOptions& opt);

    /// 最近一次求解的分割节点树节点数（诊断）。
    int64_t lastTreeNodes() const { return lastTreeNodes_; }

private:
    int64_t lastTreeNodes_ = 0;
};

} // namespace amls
