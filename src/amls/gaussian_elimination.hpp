#pragma once
// step 2a：高斯消去模块（Hyun 2020 式 5–8 的消去半边；Gao TOMS'08 Fig.2 块消元）。
//
// 对重排后的 (K, M) 按分割节点树单趟后序遍历：
//   对每个节点 i：
//     1. 块分解 K̂_ii = Lᵢ Dᵢ Lᵢᵀ（叶：稀疏 LDL；分割节点：稠密 LDL）
//     2. W = K̂_ii⁻¹ M̂_ii（式 2 同余中间量；算子形式，无显式求逆——P5/P11）
//     3. 每个祖先 a：X_a = K̂_ii⁻¹ K_{i,a}，Y_a = K̂_ii⁻¹ M_{i,a}
//        （K_{i,a} = K_{a,i}ᵀ；式 5 约束模态 Ψ_{i,a} = −X_a 的实现载体）
//     4. K 单边 Schur 更新：K̄_{a,a} −= K_{a,i}·X_a；K̄_{a,a'} −= K_{a,i}·X_{a'}
//     5. M 双边同余更新（式 2 的 M⁽ⁱ⁾ 递归）：
//          M̄_{a,a}  −= K_{a,i}Y_a + (K_{a,i}Y_a)ᵀ − K_{a,i}·W·X_a
//          M̄_{a,a'} −= K_{a,i}Y_{a'} + (K_{a',i}Y_a)ᵀ − K_{a,i}·W·X_{a'}
//
// 不做截断 / 投影——那是子空间缩聚模块（式 16/17）的职责；
// 消去与截断无关，因此两趟阈值估计只需重跑缩聚，消去只做一次。
// 消去完成后全部因子与变换块保留，get 值接口供缩聚与恢复使用。
// 所有矩阵乘走 OpenBLAS gemm，转置用 CblasTrans 标志不显式转置。
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
    int64_t xNZRows;     // X_a = K̂⁻¹K_{i,a} 的显著非零行数（1e-10 阈值）
    int64_t yNZRows;     // Y_a = K̂⁻¹M_{i,a} 的显著非零行数
    double  kFrac;       // 耦合块显著非零元占比
    double  xFrac;       // X_a = K̂⁻¹K_{i,a} 显著非零元占比
    double  yFrac;       // Y_a = K̂⁻¹M_{i,a} 显著非零元占比
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

    /// 后序消去。numThreads>1 时按树层级并行（同层节点读写集合不相交，
    /// 见 cpp 头注；祖先块的写冲突用条带锁）。完成后可通过 get 值接口
    /// 访问全部结果。
    void run(int numThreads = 1);

    /// DAG 并行：与 run() 同一消去数学，调度换成节点内子任务依赖图
    /// （core/dag.hpp）。节点就绪 = 直接孩子 DONE（DONE(i)→F(parent)
    /// 边显式化），不再有层 barrier——不同子树的节点可跨层并行；
    /// 节点内：F(分解) → B(X/Y 回代, per 祖先) → C(WX, per 祖先) /
    /// D(K Schur 更新) / E(M 同余更新) → DONE。写-写冲突仍由条带锁处理。
    void runDag(int numThreads);

    // ---- 统一 DAG（GE + 缩聚同一 runner；供融合流水线调用）----
    struct GeDagTasks {
        std::vector<DagTask*> f;     // per 节点 F（图根）
        std::vector<DagTask*> done;  // per 节点 DONE（缩聚挂边的锚点）
    };
    /// 只建图不运行：任务与依赖边全部 emplace 进外部 runner（不 ready）。
    /// 缩聚任务建进同一 runner 后，由调用方 ready 全部叶子 F 并 waitIdle。
    void buildDag(DagRunner& dag, GeDagTasks& out);

    /// 只缩内存不删表项：节点 i 的 (a,i) 耦合块与 diag 块的稠密数据清零、
    /// 可选释放因子。结构（桶链）不动——可与无锁 find 并发（不动桶链，
    /// 只写已存在表项的值）；这是统一 DAG 内"边算边放"的正式释放路径，
    /// 与 releaseNode（趟间全删、串行安全）互补。const：内部缓存语义，
    /// 调用方以 DAG 依赖序保证此刻无人再读节点 i 的块。
    void shrinkNodeData(int64_t i, bool releaseFactor) const;

    // ---- 融合流水线（GE + 谱 + 投影逐层交织，releaseNode 及时释放）----
    int64_t maxLevel() const { return maxLevel_; }
    const std::vector<int64_t>& levelNodes(int64_t l) const {
        return byLevel_[static_cast<size_t>(l)];
    }
    /// 消去第 l 层（可重入：只依赖 orig* 稀疏原块，重跑结果逐位一致）。
    void runLevel(int64_t l, int numThreads);
    /// 释放节点 i 的消去产物（kDiag/mDiag/(a,i) 耦合块；保留因子与
    /// orig* 稀疏原块）。释放后 kCoupling/mCoupling/kDiag/mDiag 接口
    /// 对该节点不可用；backtransform 需要这些块，恢复向量时不得释放。
    void releaseNode(int64_t i);

    /// 清空全部动态块表（四张 ColComp 表），供"同一 GE 对象重跑一遍完整
    /// 消去"前调用。典型场景：twoPass 趟 B 重跑且趟 A 因 keepFactors
    /// 未释放——残留旧块会让重跑变成在旧值上累加（K̂ 失去正定性）。
    /// 因子与符号表不受影响（重跑时重算/复用）。
    void resetBlocks();

    /// 调试：四个稠密块表的当前总字节数（诊断峰值内存构成）。
    double denseBytes() const;
    /// 调试：稠密块零列统计（列稀疏性数据支撑）。返回全零块数/总块数，
    /// 经引用带出 零列总数/总列数（含 kDiag/mDiag）。
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

    /// 算子插件（默认 OpenBLAS 后端；注入其他实现可替换 gemm/gemv 内核）。
    const ColCompOps& ops() const { return *ops_; }
    void setOps(const ColCompOps& o) { ops_ = &o; }

    /// 节点 i 的变换后对角块（分割节点稠密消去累计值；ident 恒等表，
    /// .data 即全列稠密矩阵）。叶节点请用 origKDiag/origMDiag（稀疏）。
    const ColComp& kDiag(int64_t i) const { return kDiag_.at(i); }
    const ColComp& mDiag(int64_t i) const { return mDiag_.at(i); }
    bool hasDiag(int64_t i) const { return kDiag_.count(i) != 0; }

    /// (a, lo)：行祖先 a × 列后代 lo 的消去后耦合块（= 该对在其共同
    /// 后代全部消去完成时刻的状态，这正是缩聚右投影所需的快照）。
    /// 列压缩：.cols 为升序存储列（ident 时为空表 = 全列）。
    const ColComp& kCoupling(int64_t a, int64_t lo) const {
        return kCoup_(a, lo);
    }
    const ColComp& mCoupling(int64_t a, int64_t lo) const {
        return mCoup_(a, lo);
    }
    bool hasCoupling(int64_t a, int64_t lo) const {
        return kCoupD_.count(key_(a, lo)) != 0;
    }

    /// 原始（重排后）稀疏子块——叶节点对角块与全部原始耦合块。
    const SparseMatrix& origKDiag(int64_t i) const {
        return origKDiag_.at(i);
    }
    const SparseMatrix& origMDiag(int64_t i) const {
        return origMDiag_.at(i);
    }
    const SparseMatrix& origKCoupling(int64_t a, int64_t lo) const;
    const SparseMatrix& origMCoupling(int64_t a, int64_t lo) const;

    /// 块算子（式 5 Ψ 的实现）：X = K̂_ii⁻¹ B。B/X 稠密，列数任意。
    void solveFactor(int64_t i, const DenseMatrix& B, DenseMatrix& X) const;

    /// 测量模式：非空时消去过程记录块结构统计（不影响数值）。
    GeStats* stats = nullptr;

    /// 消去内部阶段计时（秒）：factor / solve（W、X、Y 回代）/ schurK /
    /// congrM（M 同余更新），各分 .leaf / .sep 两路聚合。
    const std::map<std::string, double>& lastTiming() const { return timing_; }

    /// 细粒度性能探针（原子累加，秒；诊断列压缩各算子开销构成）。
    struct GePhaseProf {
        std::atomic<double> schurKdiag{0}, schurKcoup{0};
        std::atomic<double> congDiagT{0}, congDiagWX{0}, congDiagT2{0};
        std::atomic<double> congCoup{0};
        std::atomic<double> solveExpandT{0}, solveRhs{0};
    };
    static GePhaseProf& phaseProf() {
        static GePhaseProf p;
        return p;
    }
    /// C++17 无 atomic<double>::fetch_add，用 CAS 循环（relaxed）。
    static void profAdd(std::atomic<double>& v, double d) {
        double cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur + d,
                                        std::memory_order_relaxed))
            ;
    }
    static void phaseProfReset() {
        GePhaseProf& p = phaseProf();
        for (std::atomic<double>* v :
             {&p.schurKdiag, &p.schurKcoup, &p.congDiagT, &p.congDiagWX,
              &p.congDiagT2, &p.congCoup, &p.solveExpandT, &p.solveRhs})
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
    // kDiag_/mDiag_ 也 mutable：统一 DAG 的 shrinkNodeData 在 const ge_
    // 上按依赖序缩容（只动值不动结构）。
    mutable std::unordered_map<int64_t, ColComp> kDiag_;    // 分割节点 K̂_ii
    mutable std::unordered_map<int64_t, ColComp> kCoupD_;  // (anc, lo) K 耦合
    mutable std::unordered_map<int64_t, ColComp> mDiag_;    // 分割节点 M̂_ii
    mutable std::unordered_map<int64_t, ColComp> mCoupD_;  // (anc, lo) M 耦合
    // 符号填充 pass 产物：各耦合块的终态列模式（= 出生表；诞生后即擦除
    // 以省内存）。消去 i 时 colK(hi,lo) |= rowK(lo,i)（K 更新列来自
    // X=K̂⁻¹K_{i,lo} 的非零列 = K_{lo,i} 的非零行；M 更新同理由 M 行驱动）。
    std::unordered_map<int64_t, ColMask> kColFinal_, mColFinal_;
    const ColCompOps* ops_ = &defaultColCompOps();
    // 符号 pass 漏列计数（诞生注入时原块列不在表中；预期恒 0）
    std::atomic<int64_t> nMiss_{0};

    // 因子（恢复与缩聚右投影的块算子载体）。mutable：shrinkNodeData 的
    // 按依赖序释放（const 语义，同块表）。
    mutable std::vector<std::unique_ptr<EigenSimplicialLDLTSolver<DefaultTraits>>>
        sparseFactors_;
    mutable std::vector<std::unique_ptr<DenseLDLTSolver<DefaultTraits>>>
        denseFactors_;

    int64_t key_(int64_t hi, int64_t lo) const { return hi * N_ + lo; }

    std::map<std::string, double> timing_;

    // 树层级分组（构造时计算；叶=0）
    std::vector<std::vector<int64_t>> byLevel_;
    std::vector<int64_t> level_;          // 节点 → 层级
    int64_t maxLevel_ = 0;

    // ---- 层并行（numThreads>1）----
    /// 节点消去的数据段：因子分解 + W/X/Y 回代（无共享写，可并行）。
    struct ElimData {
        std::vector<int64_t> anc;
        DenseMatrix W;
        std::unordered_map<int64_t, DenseMatrix> X, Y;
        // X_a / Y_a 的非零列（a 空间）：右端零列经可逆回代保持零，
        // 只需解这些列；下游 W·X 等同列跳过。缺省 = 全零解。
        std::unordered_map<int64_t, std::vector<int32_t>> xCols, yCols;
        // M̂·X 再回代的缓存（C 任务产物；DAG 路径下 D/E 任务直接取，
        // 不做 makeWX 的重复计算）。键 = 祖先 a。DAG 路径下多任务并发
        // 读写——建图阶段预填全部键（nullopt），任务阶段不触发结构变更。
        std::unordered_map<int64_t, std::optional<DenseMatrix>> WX;
        // 释放互斥：unordered_map 的 erase 会改写桶链，同桶并发 erase
        // （relX(u)/relX(u') 同 prio 同就绪、clear+erase 交叉）会竞争前驱
        // 指针 → 堆破坏（46k/12 线程随机 AV 的根因）。erase/clear 统一走锁；
        // 值的读写（B 写 X[a]、D/E 读）不触结构，无需锁。
        std::mutex relMut;
    };
    /// F 任务：K̂_ii 因式分解 + M̂_ii 缓存（d.W）。根节点此后无事。
    void factorNode_(int64_t i, ElimData& d);
    /// B 任务（per 祖先 a）：X_a / Y_a 回代（式 7a/8a 约束量）。
    void solveXYAnc_(int64_t i, ElimData& d, int64_t a, bool leaf);
    /// 旧路径整段：全部祖先的 X/Y + 块/节点统计 + solve 计时。
    void solveXY_(int64_t i, ElimData& d);
    /// C 任务：WX_a = K̂_ii⁻¹(M̂_ii·X_a)（式 8c/7c，T3 结合律；
    /// 已算过则直接取缓存）。需 X_a 与其非零列表（B 之后）。
    const DenseMatrix& makeWX_(int64_t i, ElimData& d, int64_t a) const;
    /// D/E 任务体（旧路径 applyNodeB_ 的循环体原样抽出，
    /// 含条带锁与 phaseProf 计时；不含 timing_ 段计时）。
    void schurKDiagTask_(int64_t i, ElimData& d, int64_t a, bool leaf);
    void schurKCoupTask_(int64_t i, ElimData& d, int64_t hi, int64_t lo,
                         bool leaf);
    void congDiagTask_(int64_t i, ElimData& d, int64_t a, bool leaf);
    void congCoupTask_(int64_t i, ElimData& d, int64_t hi, int64_t lo,
                       bool leaf);
    void computeNodeA_(int64_t i, ElimData& d);
    /// 更新段：Schur / 同余写入祖先块（条带锁保护写-写冲突）。
    void applyNodeB_(int64_t i, ElimData& d);
    /// DAG 统计辅助：节点级 GeNodeStat（F 任务尾部调用）。
    void recordNodeStat_(int64_t i);
    /// 目标块写锁（per-key 细粒度）：同一目标块的累加串行化，不同 key
    /// 永不互斥——256 条带哈希碰撞 + 大 dgemm 持锁曾把 12 线程扩展比
    /// 压到 1.05x（护航）。锁对象惰性建（create=结构变更，走 structMutex_），
    /// 查锁表为读多写少，用 shared_mutex。
    std::shared_mutex keyLocksMutex_;
    std::unordered_map<int64_t, std::unique_ptr<std::mutex>> keyLocks_;
    /// 取 key 的写锁（不存在则建）。返回引用——表仅在 structMutex_ 下
    /// 插入，unique_ptr 所指 mutex 生命周期随 GE 对象，引用稳定。
    std::mutex& keyLock_(int64_t key);
    mutable std::mutex structMutex_;
    std::mutex statsMutex_;

    SparseMatrix sparseBlock_(const SparseMatrix& A, int64_t r0, int64_t r1,
                              int64_t c0, int64_t c1) const;
    void precompute_();
    void eliminateNode_(int64_t i);

    // 取/建块（消去过程）
    const SparseMatrix& origK_(int64_t a, int64_t lo) const;
    const SparseMatrix& origM_(int64_t a, int64_t lo) const;

    /// 读访问：返回 (a,lo) 耦合块；不存在则诞生空表块（只读场景——
    /// 该块未收过任何更新，等价于旧实现的 Zero(n_a×n_lo) 块）。
    /// 调用方须遵守层 barrier 纪律（见 cpp 头注），表在此期间静态。
    const ColComp& kCoup_(int64_t a, int64_t lo) const;
    const ColComp& mCoup_(int64_t a, int64_t lo) const;

    /// 写访问：返回可累加的 (a,lo) 目标块。表在诞生时取自符号填充
    /// pass 的终态列模式（出生即终态，不存在运行时列插入；层 barrier
    /// 纪律保证同层无读-写并发，条带锁保证写-写串行）。
    ColComp& kCoupTarget_(int64_t hi, int64_t lo);
    ColComp& mCoupTarget_(int64_t hi, int64_t lo);
    ColComp& kDiagTarget_(int64_t a);   // ident 全表
    ColComp& mDiagTarget_(int64_t a);

    /// 符号填充 pass：由 orig* 模式精确预算全部耦合块的终态列模式
    ///（标准"无精确抵消"假定；fill 语义与数值 GE 一致）。
    void symbolicFill_();
};

} // namespace amls
