#pragma once
// step 2b：子空间缩聚模块（Hyun 2020 式 16–17；Gao Fig.2 的谱截断 + 投影半边）。
//
// 向高斯消去模块（GaussianElimination）与特征对求解模块（EigenpairSolver）
// get 值，三个阶段：
//
//   I.  局部谱（式 4）：K̂_ii Φ_i = M̂_ii Φ_i Λ_i，按截断保留 k_i 阶。
//       所有节点的 (K̂_ii, M̂_ii) 在消去完成后即固定、相互独立——
//       线程池并行（叶间 / 兄弟节点间天然并行，OOC 论文 §Parallel Computing）。
//
//   II. 投影装配（式 7/8 的 T = ΨΦ 结构；式 16/17 的
//       T̃ = [T_d·T_s^d, T_d·T_s^r, T_r] 主-残余组合在树递归中的实现）：
//       后序遍历，对每个节点 i
//         · 右投影（式 5+8 的 Ψ 作用，经 GE 块算子）：
//           M̄_{a,i} = (M̄_{a,i} − K̄_{a,i}·K̂_ii⁻¹·M̂_ii)·Z_i，a 为祖先
//         · 中间节点列更新：对已通过的后代列 kk，
//           M̄_{a,kk} −= K̄_{a,i}·K̂_ii⁻¹·M̄_{i,kk}
//         · 左投影：M̃_{i,kk} = Z_iᵀ·M̄_{i,kk}（完成 M̃ 的第 i 行）
//       全部乘积走 OpenBLAS gemm，转置用 CblasTrans 标志不显式转置。
//
//   III. 输出降阶 pencil：K̃ = diag(μ_i)（对角性质，P3），
//        M̃ 非对角树块表（行节点为列节点祖先，P4；对角块 = I，P8）。
//
// Φ_i（保留的局部特征向量 Z_i）与约束模态载体（GE 保留的因子与耦合块）
// 均不释放，供特征向量恢复（step 4）使用。
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "amls/eigenpair_solver.hpp"
#include "amls/gaussian_elimination.hpp"
#include "amls/separator_tree.hpp"
#include "amls/spectral_truncation.hpp"
#include "core/types.hpp"

namespace amls {

class SubspaceCondensation {
public:
    using SparseMatrix = DefaultTraits::SparseMatrix;
    using DenseMatrix  = DefaultTraits::DenseMatrix;
    using Vector       = DefaultTraits::Vector;

    /// M̃ 的非对角树块（P4）：rowNode 为 colNode 的祖先。
    struct ProjBlock {
        int64_t rowNode, colNode;
        DenseMatrix block;  // k_row × k_col
    };

    struct Result {
        int64_t p = 0;                        // 降阶问题规模
        std::vector<int64_t> projOffset;      // 每节点在投影空间的偏移
        std::vector<int64_t> k;               // 每节点保留谱分量数
        std::vector<Vector> mu;               // 保留局部特征值（K̃ 的对角，P3）
        std::vector<DenseMatrix> Z;           // 保留局部特征向量 Φ_i（n_i × k_i）
        std::vector<ProjBlock> mprojBlocks;   // M̃ 非对角块（对角块 = I，P8）
    };

    SubspaceCondensation(const GaussianElimination& ge, EigenpairSolver& eig)
        : ge_(ge), eig_(eig), N_(ge.nNodes()) {}

    /// 兼容整体入口：全部局部谱（层并行）→ 全部投影（层并行）→ 偏移。
    /// GE 不被释放，可继续 backtransform。
    Result run(const ISpectralTruncation<DefaultTraits>& truncation,
               int numThreads);

    // ---- 融合流水线 API（与 GE 逐层交织；solver 驱动）----
    /// 每趟调用一次：清空投影工作表。
    void beginPass();
    /// 第 l 层局部谱（式 4）+ 截断选择，填充 res.k/mu/Z。
    void spectralLevel(const ISpectralTruncation<DefaultTraits>& truncation,
                       int64_t l, int numThreads, Result& res);
    /// 第 l 层投影装配（式 16/17 的树递归实现）。
    void projectLevel(int64_t l, int numThreads, Result& res);
    /// 投影偏移与降阶规模（阶段 III）。
    void finalize(Result& res);

    /// 阶段计时（秒）：spectral = 并行局部谱，projection = 后序投影装配。
    const std::map<std::string, double>& lastStageTiming() const {
        return stageTiming_;
    }

    /// step 4（可选）：回代 z = L⁻ᵀ S q。Q: p × m（投影坐标）→ n × m（重排坐标）。
    DenseMatrix backtransform(const Result& r, const DenseMatrix& Q) const;

private:
    /// 阶段 II 单节点投影（层并行调用；节点间块读写不相交，仅容器
    /// 结构变更经 mapMutex_ 保护）。
    void projectNode_(int64_t i, Result& res);

    const GaussianElimination& ge_;
    EigenpairSolver& eig_;
    int64_t N_;

    // ---- 阶段 II 工作表（每趟 beginPass 清空）----
    std::unordered_map<int64_t, DenseMatrix> mbar_;
    std::unordered_map<int64_t, std::vector<int64_t>> mbarRow_;  // row → desc 列表
    std::mutex mapMutex_, outMutex_;
    // 本趟局部谱原始结果（spectralLevel 填充，选择后弃用）
    std::vector<Vector> wAll_;
    std::vector<DenseMatrix> vAll_;

    std::map<std::string, double> stageTiming_;

    int64_t key_(int64_t hi, int64_t lo) const { return hi * N_ + lo; }
};

} // namespace amls
