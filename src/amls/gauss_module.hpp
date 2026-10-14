#pragma once
// 高斯消去模块 —— 移植自 third_party/amlsPKU gauss_elimination 的列压缩路径
//（initializeColCompBlocks + 四个 colcomp 内核），子块来源换成宿主
// OriginalBlocks（稀疏 Eigen 子块，原始矩阵只读）。
//
// 架构语义（与 PKU 一致）：
//   - 全部子块槽位在 initialize 预建（运行期零 map 插入），列布局自底向上
//     一次性规划：对角满列；耦合块 rk = 原始 K(j,k) 列 ∪ 孩子 RK(c,k) 存储
//     列，rm = 原始 M(j,k) 列 ∪ 孩子 (RM∪RK)(c,k) 存储列（K、M 独立规划）。
//   - 约化块懒数值化（ensureReduced）：首次消费时从原始稀疏子块注入或零
//     初始化，之后只累加；更新只进独立槽位，原始矩阵永不写。
//   - 四个 stage（调用方按节点后序 + 孩子完成保证依赖）：
//       computeConstraint        Ψ_ij = −K̂_ii⁻¹·K̃_ij（稠密 dsytrf + opsColSolve）
//       accumulateReducedStiffness  K̃_jk += Ψ_ijᵀ·K̃_ik   (j ≤ k ∈ A_i)
//       computeMtemp             M_temp_ij = M̃_ij + M̃_ii·Ψ_ij（布局取并集）
//       accumulateReducedMass    M̃_jk += Ψ_ijᵀ·M̃_ik + M_temp_ijᵀ·Ψ_ik
//   - get/release：约化对角为满列恒等布局（dense() 即全矩阵稠密视图）；
//     Ψ/M_temp 供子空间缩聚以列压缩形式消费；getReservedColState 供 μ
//     列规划查 K 耦合块预留列（数值释放后仍可读）。
//   - per-entry mutex 只保护目标块累加（生产者按 DAG 依赖互斥，读侧无需锁）。
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "amls/original_blocks.hpp"
#include "core/types.hpp"
#include "linalg/col_ops.hpp"
#include "linalg/colcomp_mat.hpp"

namespace amls {

class GaussEliminationModule {
public:
    using SparseMatrix = DefaultTraits::SparseMatrix;
    using DenseMatrix = DefaultTraits::DenseMatrix;

    /// 槽位类别。对角/耦合为约化块，Ψ/M_temp 为方向块（不做键规约）。
    enum class Kind : std::uint8_t {
        ReducedDiagK = 0,
        ReducedDiagM,
        ReducedCoupK,
        ReducedCoupM,
        Psi,
        MTemp,
    };

    GaussEliminationModule() = default;
    GaussEliminationModule(const GaussEliminationModule&) = delete;
    GaussEliminationModule& operator=(const GaussEliminationModule&) = delete;

    /// 初始化：预建全部槽位并自底向上规划列布局（postOrder 遍历）。
    /// 失败返回 false（原因写错误日志）。
    bool initialize(const OriginalBlocks& blocks);

    // ---- 四个 stage（节点级）----
    /// Ψ_i,j = −K̂_ii⁻¹·K̃_i,j（j ∈ A_i）。依赖：全部后代的 KAccum/MAccum
    /// 完成（对角与 (i,j) 约化块定稿）。
    bool computeConstraint(int64_t nodeI);
    /// K̃_jk += Ψ_ijᵀ·K̃_ik（j ≤ k ∈ A_i）。依赖：computeConstraint(nodeI)。
    bool accumulateReducedStiffness(int64_t nodeI);
    /// M_temp_ij = M̃_ij + M̃_ii·Ψ_ij。依赖：computeConstraint(nodeI) 与
    /// 全部后代的 MAccum 完成（M̃_ij/M̃_ii 定稿）。
    bool computeMtemp(int64_t nodeI);
    /// M̃_jk += Ψ_ijᵀ·M̃_ik + M_temp_ijᵀ·Ψ_ik。依赖：computeMtemp(nodeI)。
    bool accumulateReducedMass(int64_t nodeI);

    // ---- get / release ----
    /// 约化对角块稠密只读视图（对角恒满列恒等布局，dense() 即全矩阵）。
    /// K 版需全部孩子的 accumulateReducedStiffness 完成，M 版需 MAccum。
    const DenseMatrix* getReducedDiagK(int64_t j) const;
    const DenseMatrix* getReducedDiagM(int64_t j) const;
    void releaseReducedDiagK(int64_t j);
    void releaseReducedDiagM(int64_t j);
    /// Ψ_ij（i 后代、j ∈ ancestors(i)）列压缩只读视图。
    const ColCompMat* getPsi(int64_t i, int64_t j) const;
    void releasePsi(int64_t i, int64_t j);
    /// M_temp_ij（方向块）列压缩只读视图。
    const ColCompMat* getMTemp(int64_t i, int64_t j) const;
    void releaseMTemp(int64_t i, int64_t j);
    /// K̃_ij 的预留列 state 只读视图（±(列+1) 编码）；列布局规划后即可查询，
    /// 数值释放后仍可读（μ 子块列规划查表）。
    const std::vector<int32_t>* getReservedColState(int64_t i, int64_t j) const;

    int64_t nNodes() const { return N_; }
    bool initialized() const { return blocks_ != nullptr; }

private:
    struct Key {
        Kind kind;
        int64_t i, j;
        bool operator==(const Key& o) const noexcept {
            return kind == o.kind && i == o.i && j == o.j;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            std::size_t h = static_cast<std::size_t>(k.kind);
            h = h * 31 + static_cast<std::size_t>(k.i);
            h = h * 31 + static_cast<std::size_t>(k.j);
            return h;
        }
    };
    struct Entry {
        explicit Entry(int64_t r, int64_t c) : matrix(r, c) {}
        Entry() = default;
        ColCompMat matrix;
        mutable std::mutex mtx;  // 目标块累加串行化
    };

    /// 约化块键规约：对角 (i,i)，耦合 (min, max)。
    Key reducedKey_(bool stiffness, int64_t i, int64_t j) const;
    Entry* slot_(const Key& key) const;
    /// 约化块懒数值化：Numeric 直接返回；Layout 从原始稀疏子块注入（缺失
    /// 耦合块零初始化）。desc 为行（后代）节点、anc 为列（祖先）节点。
    bool ensureReduced_(bool stiffness, int64_t desc, int64_t anc,
                        Entry& entry, ColCompMat*& out);
    const SparseMatrix& originalBlock_(bool stiffness, int64_t desc,
                                       int64_t anc) const;
    /// 原始稀疏块的实际非零列（缺失耦合块视为零块 → 空列表）。
    std::vector<int64_t> originalColumns_(bool stiffness, int64_t desc,
                                          int64_t anc) const;
    /// 自底向上列规划（对角满列 + 耦合块并集布局）。
    bool planColumns_();

    static void mergeColumns_(std::vector<int64_t>& acc,
                              const std::vector<int64_t>& add) {
        std::vector<int64_t> result;
        result.reserve(acc.size() + add.size());
        std::set_union(acc.begin(), acc.end(), add.begin(), add.end(),
                       std::back_inserter(result));
        acc.swap(result);
    }

    const OriginalBlocks* blocks_ = nullptr;
    const SeparatorTree* tree_ = nullptr;
    int64_t N_ = 0;
    bool useGemv_ = true;
    std::unordered_map<Key, std::unique_ptr<Entry>, KeyHash> slots_;
};

} // namespace amls
