#pragma once
// step 2a：高斯消去模块（约束模态 Ψ 存储形式；任务 = 依赖关系拆解表 N1–N5）。
//
// 对重排后的 (K, M) 按分割节点树后序消去。每个节点 i（祖先 a，祖先对 hi>lo）：
//   N1  u(i,a) = Ψ(i,a) = −K̂_ii⁻¹·K_{i,a}
//       一次 LDL 分解 K̂_ii、逐祖先窄右端回代（右端零列经可逆回代保持零，
//       解全程列压缩存储）；分解与回代完成后立即释放因子——后续所有阶段
//       （N3–N5、缩聚、回代恢复）都不再需要 K̂⁻¹ 算子。
//   N3  Mtemp(i,a) = M_{i,a} + M_ii·u(i,a)   （列 = u 列 ∪ M 耦合行模式）
//   N4  K̄(a,lo) += K_{a,i}·u(i,lo)            （K 单边 Schur 更新）
//   N5  M̄(a,lo) += M_{a,i}·u(i,lo) + u(i,a)ᵀ·Mtemp(i,lo)
//       （Mtemp 形式：M_ijᵀΨ + ΨᵀM_iiΨ 两项合并，三重积消失——推导见
//       papers/summary/公式.md；不再有 W/X/Y 中间量与按需回代）
//   N2  释放 (a,i) 耦合列块（N1/N3/N4/N5 完成后 = CSV 的 Free-SubBlock）
//
// 叶子与分割节点公式同构：叶耦合块为原始稀疏、分割节点耦合块为列压缩
// 稠密，经 blockops::accumulateProduct 的两个重载统一（点 3）。
// 全部块更新化简为 A += B·C 与 A += Bᵀ·C 两种规范形态（linalg/colcomp.hpp）。
//
// 消去与截断无关，两趟阈值估计只跑一次消去数学（趟间 resetBlocks 重放）。
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "amls/separator_tree.hpp"
#include "core/dag.hpp"
#include "core/types.hpp"
#include "linalg/colcomp.hpp"
#include "linalg/sparse_direct.hpp"

namespace amls {

/// 消去过程统计（测量模式启用；不影响数值）。
struct GeBlockStat {
    int64_t node;        // 被消去节点 i
    int64_t anc;         // 祖先 a
    int64_t level;       // 节点 i 的树层级（叶=0）
    int64_t rows;        // 块行数 n_a
    int64_t cols;        // 块列数 n_i
    int64_t nonzeroCols; // 耦合块 (a,i) 的非零列数 |B_i(a)|
    int64_t uNZRows;     // u(i,a) = Ψ 的显著非零行数（1e-10 阈值）
    double  kFrac;       // 耦合块显著非零元占比
    double  uFrac;       // u(i,a) 显著非零元占比
    bool    leaf;        // i 是否叶节点
};
struct GeNodeStat {
    int64_t node;
    int64_t n;           // 节点规模
    int64_t depth;       // 树深度（祖先数）
    bool    leaf;
    double  kDiagFrac;   // 对角块显著非零元占比（填充率）
};
struct GeStats {
    std::vector<GeBlockStat> blocks;
    std::vector<GeNodeStat>  nodes;
};

class GaussianElimination {
public:
    using SparseMatrix = DefaultTraits::SparseMatrix;
    using DenseMatrix  = DefaultTraits::DenseMatrix;
    using Vector       = DefaultTraits::Vector;
    using Index        = DefaultTraits::Index;
    using ColComp      = amls::ColComp;
    using ColMask      = amls::ColMask;

    /// reordered：已按分割节点树重排的 (K, M)。
    GaussianElimination(const Pencil<DefaultTraits>& reordered,
                        const SeparatorTree& tree);

    /// 后序消去。numThreads>1 时按树层级并行（同层节点读写集合不相交；
    /// 祖先块的写冲突用条带锁）。完成后可通过 get 值接口访问全部结果。
    void run(int numThreads = 1);

    /// DAG 并行：与 run() 同一消去数学，调度换成节点内子任务依赖图
    /// （core/dag.hpp）。节点就绪 = 直接孩子的 N4/N5 完成（无层 barrier）。
    void runDag(int numThreads);

    // ---- 统一 DAG（GE + 缩聚同一 runner；供融合流水线调用）----
    struct EliminationDagTasks {
        std::vector<DagTask*> n1;           // per 节点 N1（图根）
        std::vector<DagTask*> n3;           // per 节点 N3
        std::vector<DagTask*> updatesDone;  // per 节点 N4/N5 汇合哨兵
    };
    /// 只建图不运行：任务与依赖边全部 emplace 进外部 runner（不 ready）。
    /// 缩聚任务建进同一 runner 后，由调用方 ready 全部叶子 N1 并 waitIdle。
    void buildDag(DagRunner& dag, EliminationDagTasks& out);

    // ---- 逐层路径（GE + 谱 + 投影逐层交织，releaseNodeData 及时释放）----
    int64_t maxLevel() const { return maxLevel_; }
    const std::vector<int64_t>& levelNodes(int64_t l) const {
        return byLevel_[static_cast<size_t>(l)];
    }
    /// 消去第 l 层（可重入：只依赖 orig* 稀疏原块，重跑结果逐位一致）。
    void runLevel(int64_t l, int numThreads);
    /// 释放节点 i 的消去产物（diag 块、(a,i) 耦合列、Mtemp 列；
    /// keepConstraintModes=false 时连同 u 列一起释放——回代恢复需要 u，
    /// 恢复向量时调用方传 true）。逐层路径趟内调用（串行区，可 erase）。
    void releaseNodeData(int64_t i, bool keepConstraintModes);

    /// 清空全部动态数据（四张块表 + u/Mtemp 的值），供"同一 GE 对象重跑
    /// 一遍完整消去"前调用。典型场景：twoPass 趟 B 重跑且趟 A 因
    /// recoverVectors 未释放——残留旧块会让重跑变成在旧值上累加。
    /// 因子与符号表不受影响（重跑时重算/复用）。
    void resetBlocks();

    // ---- 任务函数（N1–N5 + 释放；逐层路径与 DAG 路径共用同一实现）----
    /// N1：约束模态 u(i,a) = −K̂_ii⁻¹·K_{i,a}（一次分解、逐祖先窄回代；
    /// 完成后释放因子）。依赖：全部后代的 N4/N5 完成（diag/耦合列定稿）。
    void computeConstraintModes(int64_t i);
    /// N3：Mtemp(i,a) = M_{i,a} + M_ii·u(i,a)。依赖：N1(i)。
    void buildTemporaryMass(int64_t i);
    /// N4 对角：K̄(a,a) += K_{a,i}·u(i,a)。
    void stiffnessDiagUpdate(int64_t i, int64_t a);
    /// N4 耦合：K̄(hi,lo) += K_{hi,i}·u(i,lo)。
    void stiffnessCoupUpdate(int64_t i, int64_t hi, int64_t lo);
    /// N5 对角：M̄(a,a) += M_{a,i}·u(i,a) + u(i,a)ᵀ·Mtemp(i,a)。
    void massDiagUpdate(int64_t i, int64_t a);
    /// N5 耦合：M̄(hi,lo) += M_{hi,i}·u(i,lo) + u(i,hi)ᵀ·Mtemp(i,lo)。
    void massCoupUpdate(int64_t i, int64_t hi, int64_t lo);
    /// N2：释放 (a,i) 耦合列块（值缩容；依赖：N1/N3/N4/N5(i) 全部完成）。
    void releaseCouplingColumn(int64_t i) const;
    /// 释放 diag 块（值缩容；依赖：N1(i) 与局部谱 N7(i) 完成）。
    void releaseDiagonalBlocks(int64_t i) const;
    /// 释放 Mtemp 列（值缩容；依赖：N5(i) 与 N8(i) 完成）。
    void releaseTemporaryMass(int64_t i) const;
    /// 释放 u 列（值缩容；依赖：N9/N10 折叠传播 fold(i) 完成；
    /// recoverVectors 时不调用——回代恢复需要 u）。
    void releaseConstraintModes(int64_t i) const;

    /// 调试：四个稠密块表的当前总字节数（诊断峰值内存构成）。
    double denseBytes() const;
    /// 调试：稠密块零列统计（列稀疏性数据支撑）。
    void zeroColStats(int64_t& zeroCols, int64_t& totalCols,
                      int64_t& zeroBlocks, int64_t& totalBlocks) const;

    // ---- get 值接口（供子空间缩聚 / 特征向量恢复）----
    bool isLeaf(int64_t i) const;
    int64_t offset(int64_t i) const { return off_[static_cast<size_t>(i)]; }
    int64_t size(int64_t i) const { return size_[static_cast<size_t>(i)]; }
    int64_t nNodes() const { return N_; }
    int64_t level(int64_t i) const { return level_[static_cast<size_t>(i)]; }
    const SeparatorTree& tree() const { return tree_; }
    const std::vector<int64_t>& postOrder() const { return order_; }

    /// 节点 i 的变换后对角块（分割节点稠密消去累计值；ident 恒等表，
    /// .data 即全列稠密矩阵）。叶节点请用 origKDiag/origMDiag（稀疏）。
    const ColComp& kDiag(int64_t i) const { return kDiag_.at(i); }
    const ColComp& mDiag(int64_t i) const { return mDiag_.at(i); }
    bool hasDiag(int64_t i) const { return kDiag_.count(i) != 0; }

    /// 约束模态 u(i,a) = Ψ(i,a)（列压缩；cols = a 空间非零列）。
    /// N1(i) 后可用，fold(i)（或回代恢复）后释放。
    const ColComp& constraintModes(int64_t i, int64_t a) const {
        return constraintModes_.at(modeKey_(i, a));
    }
    /// Mtemp(i,a) = M_{i,a} + M_ii·u(i,a)（列压缩，行 = i 空间全）。
    /// N3(i) 后可用，N8(i) 后释放。
    const ColComp& temporaryMass(int64_t i, int64_t a) const {
        return temporaryMass_.at(modeKey_(i, a));
    }

    /// 原始（重排后）稀疏对角块（叶节点局部谱输入）。
    const SparseMatrix& origKDiag(int64_t i) const {
        return origKDiag_.at(i);
    }
    const SparseMatrix& origMDiag(int64_t i) const {
        return origMDiag_.at(i);
    }

    /// 符号填充 pass 终态行模式（供 MassCouplingStore 预算 μ̃ 列模式）：
    /// (a,i) 块消去完成时刻的非零行 = u(i,a) / M 耦合 (i,a) 的列模式。
    const ColMask& symbolicRowK(int64_t a, int64_t i) const {
        return rowKFinal_.at(key_(a, i));
    }
    const ColMask& symbolicRowM(int64_t a, int64_t i) const {
        return rowMFinal_.at(key_(a, i));
    }

    /// 测量模式：非空时消去过程记录块结构统计（不影响数值）。
    GeStats* stats = nullptr;

    /// 消去内部阶段计时（秒）：factor / solve（N1 回代）/ mtemp（N3）/
    /// schurK（N4）/ congrM（N5），各分 .leaf / .sep 两路聚合。
    const std::map<std::string, double>& lastTiming() const { return timing_; }

    /// 细粒度性能探针（原子累加，秒；诊断各算子开销构成）。
    struct GePhaseProf {
        std::atomic<double> schurKdiag{0}, schurKcoup{0};
        std::atomic<double> massDiagTerm1{0}, massDiagTerm2{0};
        std::atomic<double> massCoupTerm1{0}, massCoupTerm2{0};
        std::atomic<double> mtempBuild{0}, solveRhs{0};
    };
    static GePhaseProf& phaseProf() {
        static GePhaseProf prof;
        return prof;
    }
    /// C++17 无 atomic<double>::fetch_add，用 CAS 循环（relaxed）。
    static void profAdd(std::atomic<double>& v, double d) {
        double cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur + d,
                                        std::memory_order_relaxed))
            ;
    }
    static void phaseProfReset() {
        GePhaseProf& prof = phaseProf();
        for (std::atomic<double>* v :
             {&prof.schurKdiag, &prof.schurKcoup, &prof.massDiagTerm1,
              &prof.massDiagTerm2, &prof.massCoupTerm1, &prof.massCoupTerm2,
              &prof.mtempBuild, &prof.solveRhs})
            v->store(0.0);
    }

private:
    const Pencil<DefaultTraits>& pencil_;
    const SeparatorTree& tree_;

    int64_t N_ = 0;
    std::vector<int64_t> order_;          // 后序
    std::vector<int64_t> off_;            // 节点在重排坐标中的偏移
    std::vector<int64_t> size_;           // 节点规模 n_i

    // 原始（重排后）稀疏子块
    std::unordered_map<int64_t, SparseMatrix> origKDiag_, origMDiag_;
    std::unordered_map<int64_t, SparseMatrix> origKCoup_, origMCoup_;

    // 消去过程动态块（列压缩；diag 为 ident 全表退化情形）
    // 四表均 mutable：统一 DAG 的按依赖序值缩容在 const 路径上亦可表述
    // （只动值不动结构）。
    mutable std::unordered_map<int64_t, ColComp> kDiag_;    // 分割节点 K̂_ii
    mutable std::unordered_map<int64_t, ColComp> kCoupD_;  // (anc, lo) K 耦合
    mutable std::unordered_map<int64_t, ColComp> mDiag_;    // 分割节点 M̂_ii
    mutable std::unordered_map<int64_t, ColComp> mCoupD_;  // (anc, lo) M 耦合
    // 约束模态 u(i,a) = Ψ 与 Mtemp(i,a)：键 = modeKey_(i,a)，precompute_
    // 预建全部键（运行期只写值，无结构变更）。
    mutable std::unordered_map<int64_t, ColComp> constraintModes_;
    mutable std::unordered_map<int64_t, ColComp> temporaryMass_;
    // 符号填充 pass 产物：各耦合块的终态列模式（= 出生表）与终态行模式
    //（行模式供 MassCouplingStore 预算 μ̃ 列模式）。消去 i 时
    // colK(hi,lo) |= rowK(lo,i)（N4 积列 = u(i,lo) 列 = K_{lo,i} 非零行）；
    // colM(hi,lo) |= rowK(lo,i) ∪ rowM(lo,i)（N5 两项）。
    std::unordered_map<int64_t, ColMask> kColFinal_, mColFinal_;
    std::unordered_map<int64_t, ColMask> rowKFinal_, rowMFinal_;
    // 符号 pass 漏列计数（诞生注入时原块列不在表中；预期恒 0）
    std::atomic<int64_t> nMiss_{0};

    // 因子（N1 内部载体：分解 → 逐祖先回代 → 立即释放；之后无消费者）。
    std::vector<std::unique_ptr<EigenSimplicialLDLTSolver<DefaultTraits>>>
        sparseFactors_;
    std::vector<std::unique_ptr<DenseLDLTSolver<DefaultTraits>>>
        denseFactors_;

    int64_t key_(int64_t hi, int64_t lo) const { return hi * N_ + lo; }
    int64_t modeKey_(int64_t i, int64_t a) const { return i * N_ + a; }

    std::map<std::string, double> timing_;

    // 树层级分组（构造时计算；叶=0）
    std::vector<std::vector<int64_t>> byLevel_;
    std::vector<int64_t> level_;          // 节点 → 层级
    int64_t maxLevel_ = 0;

    /// 节点级统计（F 任务尾部调用）。
    void recordNodeStat_(int64_t i);
    /// 目标块写锁（per-key 细粒度）：同一目标块的累加串行化，不同 key
    /// 永不互斥。锁对象惰性建（create=结构变更，走 keyLocksMutex_）。
    std::shared_mutex keyLocksMutex_;
    std::unordered_map<int64_t, std::unique_ptr<std::mutex>> keyLocks_;
    std::mutex& keyLock_(int64_t key);
    mutable std::mutex structMutex_;
    std::mutex statsMutex_;

    SparseMatrix sparseBlock_(const SparseMatrix& A, int64_t r0, int64_t r1,
                              int64_t c0, int64_t c1) const;
    void precompute_();
    void eliminateNode_(int64_t i);
    void solveWithFactor_(int64_t i, const DenseMatrix& rhs,
                          DenseMatrix& solution) const;

    // 取/建块（消去过程）
    const SparseMatrix& origK_(int64_t a, int64_t lo) const;
    const SparseMatrix& origM_(int64_t a, int64_t lo) const;

    /// 读访问：返回 (a,lo) 耦合块；不存在则诞生空表块（只读场景）。
    const ColComp& kCoup_(int64_t a, int64_t lo) const;
    const ColComp& mCoup_(int64_t a, int64_t lo) const;

    /// 写访问：返回可累加的 (a,lo) 目标块（出生表 = 符号 pass 终态列模式）。
    ColComp& kCoupTarget_(int64_t hi, int64_t lo);
    ColComp& mCoupTarget_(int64_t hi, int64_t lo);
    ColComp& kDiagTarget_(int64_t a);   // ident 全表
    ColComp& mDiagTarget_(int64_t a);

    /// 符号填充 pass：由 orig* 模式精确预算全部耦合块的终态列/行模式
    ///（标准"无精确抵消"假定；fill 语义与数值 GE 一致）。
    void symbolicFill_();
};

} // namespace amls
