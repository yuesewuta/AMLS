#pragma once
// step 2b：子空间缩聚模块（约束模态 Ψ/Mtemp 形式；任务 = 依赖关系拆解表
// N7–N10）。
//
// 向高斯消去模块（GaussianElimination）与特征对求解模块（EigenpairSolver）
// get 值，四个任务：
//
//   N7  局部谱：K̂_ii·eigenvectors_i = M̂_ii·eigenvectors_i·eigenvalues_i，
//       按截断保留 keptModes_i 阶。全部节点输入在各自 N4/N5 汇合后固定、
//       相互独立——线程池/DAG 并行。
//   N8  种子：μ̃(c,a) = eigenvectors_cᵀ·Mtemp(c,a)（c 的全部祖先 a），
//       写入 MassCouplingStore（右耦合结果的独立载体，本类不再持有）。
//   N9  折叠：m(c,i) = μ̃(c,i)·eigenvectors_i，转置后产出降阶质量块
//       ReducedMassBlock{rowNode=i, colNode=c, block=(k_i × k_c)}。
//   N10 传播：μ̃(c,a) += μ̃(c,i)·u(i,a)（i 在 c→a 路径上），沿树向上。
//
// 输出降阶 pencil：K̃ = diag(eigenvalues_i)（P3），M̃ 非对角树块表
//（P4；对角块 = I，P8，局部特征向量 M-正交）。
//
// μ̃ 单写者链：种子(c) → 沿 c→a 路径的 fold 序列；树路径全序 + DAG 依赖
//（fold(i) ← fold(孩子)、fold(i) ← N8(全部后代)）保证任意时刻单写者，
// MassCouplingStore 内部无锁。
//
// 特征向量恢复（step 4）只需 u 与 eigenvectors（纯 GEMM 回代，
// z_i = eigenvectors_i·q_i + Σ_a u(i,a)·z_a）——不再需要 K̂⁻¹ 因子与耦合块。
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "amls/eigenpair_solver.hpp"
#include "amls/gaussian_elimination.hpp"
#include "amls/mass_coupling_store.hpp"
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
    struct ReducedMassBlock {
        int64_t rowNode, colNode;
        DenseMatrix block;  // keptModes_row × keptModes_col
    };

    struct Result {
        int64_t reducedSize = 0;               // 降阶问题规模 p
        std::vector<int64_t> reducedOffset;    // 每节点在投影空间的偏移
        std::vector<int64_t> keptModes;        // 每节点保留谱分量数 k_i
        std::vector<Vector> eigenvalues;       // 保留局部特征值（K̃ 对角，P3）
        std::vector<DenseMatrix> eigenvectors; // 保留局部特征向量（n_i × k_i）
        std::vector<ReducedMassBlock> reducedMassBlocks;  // M̃ 非对角块
    };

    SubspaceCondensation(const GaussianElimination& ge, EigenpairSolver& eig)
        : ge_(ge), eig_(eig), N_(ge.nNodes()) {}

    /// 兼容整体入口：全部局部谱（层并行）→ 全部投影（层并行）→ 偏移。
    /// GE 不被释放，可继续 backtransform。
    Result run(const ISpectralTruncation<DefaultTraits>& truncation,
               int numThreads);

    /// 统一 DAG：N7–N10 与 GE 任务进同一 runner（依赖边挂到 geInfo）。
    ///   N7(i) ← updatesDone(i)（叶经哨兵 ← N1）
    ///   N8(i) ← N7(i), N3(i)
    ///   fold(i) ← N1(i), N7(i), N8(i), fold(孩子), N8(全部后代)
    ///   freeMtemp(i) ← updatesDone(i), N8(i)
    ///   freeDiag(i) ← N1(i), N7(i)
    ///   freeU(i) ← fold(i)（releaseData=false 即恢复向量时不建——回代需要 u）
    /// 调用方负责：beginPass() → buildDag → ready GE 叶子 N1 → waitIdle →
    /// finalize(res)。
    void buildDag(DagRunner& dag, Result& res,
                  const ISpectralTruncation<DefaultTraits>& truncation,
                  const GaussianElimination::EliminationDagTasks& geInfo,
                  bool releaseData);

    // ---- 融合流水线 API（与 GE 逐层交织；solver 驱动）----
    /// 每趟调用一次：重建 μ̃ 存储槽位（GE 符号 pass 行模式预算列模式）。
    void beginPass();
    /// 第 l 层局部谱（N7）+ 截断选择，填充 res.keptModes/eigenvalues/
    /// eigenvectors。
    void spectralLevel(const ISpectralTruncation<DefaultTraits>& truncation,
                       int64_t l, int numThreads, Result& res);
    /// 第 l 层投影：N8 种子 + N9/N10 折叠传播（节点间块读写不相交）。
    void projectLevel(int64_t l, int numThreads, Result& res);
    /// 投影偏移与降阶规模。
    void finalize(Result& res);

    // ---- 任务函数（N7–N10；逐层路径与 DAG 路径共用同一实现）----
    /// N7：局部谱 + 截断选择（写 res.keptModes/eigenvalues/eigenvectors(i)）。
    void solveLocalEigenproblem(
        int64_t i, const ISpectralTruncation<DefaultTraits>& truncation,
        Result& res);
    /// N8：μ̃(i,a) = eigenvectors_iᵀ·Mtemp(i,a)（i 的全部祖先 a）。
    void seedMassCoupling(int64_t i, const Result& res);
    /// N9 + N10：折叠列 i 产出降阶质量块，随后沿路径传播并释放 μ̃ 列 i。
    /// keptModes_i = 0 时折叠为空操作，传播照常（链不可断）。
    void foldAndPropagate(int64_t i, Result& res);

    /// 阶段计时（秒）：spectral = 并行局部谱，projection = 种子+折叠传播。
    const std::map<std::string, double>& lastStageTiming() const {
        return stageTiming_;
    }

    /// step 4（可选）：纯 GEMM 回代 z_i = eigenvectors_i·q_i + Σ_a u(i,a)·z_a。
    /// Q: p × m（投影坐标）→ n × m（重排坐标）。需要 u 与 eigenvectors
    /// 存活（recoverVectors 时调用方不释放）。
    DenseMatrix backtransform(const Result& r, const DenseMatrix& Q) const;

private:
    const GaussianElimination& ge_;
    EigenpairSolver& eig_;
    int64_t N_;

    /// μ̃ 右耦合存储（独立载体；beginPass 预建槽位，运行期无结构变更）。
    MassCouplingStore store_;
    /// reducedMassBlocks 追加锁（折叠任务的唯一共享写点）。
    std::mutex outMutex_;

    std::map<std::string, double> stageTiming_;
};

} // namespace amls
