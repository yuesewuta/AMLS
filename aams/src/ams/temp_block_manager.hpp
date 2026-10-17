#pragma once
// 临时块管理器 —— 移植自 third_party/amlsPKU compress_matrix/temp_block_manager
// 的列压缩路径（createColComp / updateColComp / rightMultiplyColComp）。
//
// 职责：子空间缩聚 μ 块的创建、多跳传播、右乘提交与释放。
//   - create(i)：μ_{i,j} = Φ_iᵀ·M_temp_{i,j}（j ∈ A_i），布局 = M_temp 实际
//     有效列 ∪ K̃(x,j) 预留列（x 为 j 通往 i 方向的直接孩子；无中间跳时传
//     空）—— 预留列定理保证后续 Ψ_{x,j} 有效列可写。
//   - update(i)：μ_{d,a} += μ_{d,i}·Ψ_{i,a}（d ∈ D_i, a ∈ A_i），左块须
//     remainingUpdates==0（只读终态），目标逐块短锁内累加并倒计数。
//     remainingUpdates = depth(i) − depth(d) − 1（|Δdepth|−1 倒计数）。
//   - rightMultiply(i)：S_{d,i} = μ_{d,i}·Φ_i 稠密提交 SubspaceMassSaver。
//   - release(i)：释放 (d,i) 全部块；根节点整树释放（单树求解，门控简化）。
//
// 与 PKU 的有意偏离：
//   - 仅列压缩路径（宿主统一稠密 dsytrf 决策）；无 STANDARD/LDL 双模式。
//   - 块 key 用方向 (后代, 祖先) 不做 (min,max) 规约 —— μ 本质方向块，
//     读取只发生在该方向（PKU 规约为其稠密双向查询服务）。
//   - 任务进入/完成登记由 NodeTaskExecutor（步骤 5）持有；本模块只维护
//     节点级 done 标志与 NodeOperationGate（释放与节点任务互斥）。
//   - 根节点整树释放为单树直通实现（RootReleaseGate 的串行化保留为原子
//     状态机，无多根 map）。
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ams/eigen_pair_saver.hpp"
#include "ams/gauss_module.hpp"
#include "ams/separator_tree.hpp"
#include "ams/subspace_mass_saver.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"
#include "linalg/col_ops.hpp"
#include "linalg/colcomp_mat.hpp"

namespace aams {

class TempBlockManager {
public:
    TempBlockManager() = default;
    TempBlockManager(const TempBlockManager&) = delete;
    TempBlockManager& operator=(const TempBlockManager&) = delete;

    /// 预建全部 (后代, 祖先) 块状态与节点门控；运行期零 map 插入。
    bool initialize(const SeparatorTree& tree, GaussEliminationModule* gauss,
                    const EigenPairSaver* eigenPairSaver,
                    SubspaceMassSaver* massSaver);

    /// 左耦合创建：μ_{i,j} = 0 + Φ_iᵀ·M_temp_{i,j}。
    /// 依赖：节点 i 局部特征对已提交、gauss 四 stage（i）完成。
    bool create(int64_t nodeID);
    /// 多跳传播：μ_{d,a} += μ_{d,i}·Ψ_{i,a}。依赖：create 全后代、update
    /// 全中间节点（左块 remainingUpdates==0）。
    bool update(int64_t nodeID);
    /// 右耦合：S_{d,i} = μ_{d,i}·Φ_i 提交质量块存储。依赖：update(i) 完成。
    bool rightMultiply(int64_t nodeID);
    /// 释放节点已消费的 (d, nodeID) 块；根节点触发整树释放。
    bool release(int64_t nodeID);
    /// 清空全部临时块（不得与其他成员接口并发）。
    void clear();

    /// μ (desc, anc) 只读视图（须 Active 且数值就绪，否则返回 nullptr）；
    /// 供测试对拍与诊断。
    const ColCompMat* getMu(int64_t desc, int64_t anc) const {
        const BlockState* state = findBlock_(Key{desc, anc});
        if (state == nullptr ||
            state->lifecycle.load(std::memory_order_acquire) != Lifecycle::Active)
            return nullptr;
        return &state->value;
    }

    int64_t nNodes() const { return N_; }
    bool initialized() const { return tree_ != nullptr; }

private:
    using Key = std::pair<int64_t, int64_t>;  // (后代, 祖先) 方向键
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return static_cast<std::size_t>(k.first) * 0x9e3779b97f4a7c15ULL ^
                   static_cast<std::size_t>(k.second);
        }
    };

    enum class Lifecycle : std::uint8_t { Empty = 0, Active, Released };

    struct BlockState {
        explicit BlockState(Key k) : key(k) {}
        Key key;
        mutable std::mutex mtx;              // 数值写入串行化
        ColCompMat value;                    // μ 子块（列压缩）
        std::atomic<int64_t> remainingUpdates{-1};
        std::atomic<Lifecycle> lifecycle{Lifecycle::Empty};
    };

    /// 节点操作门控：释放请求与节点任务互斥（create 不受本节点释放限制，
    /// 其只写 (i,·) 块，与释放的 (·,i) 块不冲突，语义与 PKU 一致）。
    struct NodeGate {
        std::mutex mtx;
        std::condition_variable cv;
        std::size_t activeCalls = 0;
        std::size_t releaseRequests = 0;
        bool releaseInProgress = false;
        bool released = false;
    };

    BlockState* findBlock_(const Key& key) const;
    NodeGate* gate_(int64_t nodeID) const;
    bool enterTask_(int64_t nodeID, bool isCreate);
    void leaveTask_(int64_t nodeID);
    /// depth(i) = 祖先数；块 update 次数 = depth(i) − depth(d) − 1。
    int64_t depth_(int64_t nodeID) const {
        return static_cast<int64_t>(tree_->ancestors(nodeID).size());
    }
    bool createColComp_(int64_t nodeID);
    bool updateColComp_(int64_t nodeID);
    bool rightMultiplyColComp_(int64_t nodeID);
    bool releaseNodeBlocks_(int64_t nodeID);

    const SeparatorTree* tree_ = nullptr;
    GaussEliminationModule* gauss_ = nullptr;
    const EigenPairSaver* eigenPairSaver_ = nullptr;
    SubspaceMassSaver* massSaver_ = nullptr;
    bool useGemv_ = true;
    int64_t N_ = 0;

    std::unordered_map<Key, std::shared_ptr<BlockState>, KeyHash> blocks_;
    // 每节点待释放块 (d, node)（初始化时按 key 排序，批量加锁定序防死锁）
    std::vector<std::vector<std::shared_ptr<BlockState>>> nodeReleaseBlocks_;
    std::vector<std::unique_ptr<NodeGate>> gates_;
    // 节点任务完成标志（本模块维护，release 依赖检查用）；atomic 不可拷贝，
    // 以 unique_ptr 数目存放
    std::vector<std::unique_ptr<std::atomic<uint8_t>>> doneFlags_;  // bit0 create bit1 update bit2 right
    // 根节点整树释放状态机（单树）
    enum class RootRelease : std::uint8_t { Idle = 0, Running, Done, Failed };
    std::atomic<RootRelease> rootRelease_{RootRelease::Idle};
};

} // namespace aams
