#pragma once
// 子空间质量矩阵块存储 —— 移植自 third_party/amlsPKU
// compress_matrix/subspace_mass_matrix_saver（稠密组装路径）。
//
// 缩聚阶段 rightMultiply 按块一次性发布 S_{d,a} = μ_{d,a}·Φ_a（行 = 后代
// 模态、列 = 祖先模态，方向键 (后代, 祖先)，PKU 的 μ·Φ 语义）；组装阶段
// 按分隔树拓扑拼出全局稠密子空间质量矩阵：对角恒填 1（局部特征对
// M-正交归一约定，ΦᵀM̃Φ = I），非对角块按行/列偏移写入并对称补全。
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ams/eigen_pair_saver.hpp"
#include "ams/separator_tree.hpp"
#include "core/logging.hpp"
#include "core/types.hpp"

namespace aams {

class SubspaceMassSaver {
public:
    using DenseMatrix = DefaultTraits::DenseMatrix;

    SubspaceMassSaver() = default;
    SubspaceMassSaver(const SubspaceMassSaver&) = delete;
    SubspaceMassSaver& operator=(const SubspaceMassSaver&) = delete;

    /// 按树固化全部合法 (后代, 祖先) 块 key；运行期不再增删 key。
    bool initialize(const SeparatorTree& tree) {
        const int64_t n = tree.nNodes();
        if (n <= 0) {
            AMLS_LOG_ERROR("SubspaceMassSaver 初始化失败：空树。");
            return false;
        }
        expectedKeys_.clear();
        for (int64_t i = 0; i < n; ++i)
            for (int64_t a : tree.ancestors(i)) expectedKeys_.insert(Key{i, a});
        expectedBlockCount_ = static_cast<int64_t>(expectedKeys_.size());
        {
            std::lock_guard<std::mutex> lock(blocksMtx_);
            blocks_.clear();
            blocks_.reserve(expectedKeys_.size());
        }
        initialized_.store(true, std::memory_order_release);
        return true;
    }

    /// 移动发布一个完整块（方向：rowId 后代、colId 祖先）；同 key 只许一次。
    bool set(int64_t rowId, int64_t colId, DenseMatrix&& value) {
        if (!initialized_.load(std::memory_order_acquire)) {
            AMLS_LOG_ERROR("SubspaceMassSaver::set 失败：未初始化。");
            return false;
        }
        const Key key{rowId, colId};
        std::lock_guard<std::mutex> lock(blocksMtx_);
        if (expectedKeys_.find(key) == expectedKeys_.end()) {
            AMLS_LOG_ERROR(
                "SubspaceMassSaver::set 失败：块 ({},{}) 不是树内合法 (后代,祖先) "
                "对。",
                rowId, colId);
            return false;
        }
        if (!blocks_.emplace(key, std::move(value)).second) {
            AMLS_LOG_ERROR("SubspaceMassSaver::set 失败：块 ({},{}) 重复提交。",
                           rowId, colId);
            return false;
        }
        return true;
    }

    /// 清空已提交块（保留合法 key，可用于同一棵树的下一轮计算）。
    void clearBlocks() {
        std::lock_guard<std::mutex> lock(blocksMtx_);
        blocks_.clear();
    }

    /// 全部预期块就绪后组装全局稠密子空间质量矩阵。各节点模态数取自
    /// eigenPairSaver；对角恒填 1（M-正交归一约定）。
    bool assembleSubspaceMass(const EigenPairSaver& eigenPairSaver,
                              DenseMatrix& out) const {
        if (!initialized_.load(std::memory_order_acquire)) {
            AMLS_LOG_ERROR("assembleSubspaceMass 失败：未初始化。");
            return false;
        }
        const int64_t n = eigenPairSaver.nodeCount();
        std::vector<int64_t> modes(static_cast<size_t>(n), 0);
        std::vector<int64_t> offsets(static_cast<size_t>(n), 0);
        int64_t total = 0;
        for (int64_t i = 0; i < n; ++i) {
            EigenPairSaver::View view;
            if (!eigenPairSaver.getEigenPairView(i, view)) {
                AMLS_LOG_ERROR("assembleSubspaceMass 失败：节点 {} 特征对缺失。", i);
                return false;
            }
            modes[static_cast<size_t>(i)] =
                static_cast<int64_t>(view.eigenvalues()->size());
            offsets[static_cast<size_t>(i)] = total;
            total += modes[static_cast<size_t>(i)];
        }
        if (total == 0 && !eigenPairSaver.allowZeroEigenPairs()) {
            AMLS_LOG_ERROR("assembleSubspaceMass 失败：全部节点 0 模态。");
            return false;
        }

        std::lock_guard<std::mutex> lock(blocksMtx_);
        if (static_cast<int64_t>(blocks_.size()) != expectedBlockCount_) {
            AMLS_LOG_ERROR(
                "assembleSubspaceMass 失败：已提交块数 {} 与预期 {} 不符。",
                blocks_.size(), expectedBlockCount_);
            return false;
        }

        out = DenseMatrix::Zero(total, total);
        for (int64_t i = 0; i < total; ++i) out(i, i) = 1.0;
        for (const auto& [key, block] : blocks_) {
            const int64_t d = key.first, a = key.second;
            if (block.rows() != modes[static_cast<size_t>(d)] ||
                block.cols() != modes[static_cast<size_t>(a)]) {
                AMLS_LOG_ERROR(
                    "assembleSubspaceMass 失败：块 ({},{}) 形状 {}x{} 与模态数 "
                    "{}x{} 不符。",
                    d, a, block.rows(), block.cols(),
                    modes[static_cast<size_t>(d)], modes[static_cast<size_t>(a)]);
                return false;
            }
            out.block(offsets[static_cast<size_t>(d)],
                      offsets[static_cast<size_t>(a)], block.rows(),
                      block.cols()) = block;
            out.block(offsets[static_cast<size_t>(a)],
                      offsets[static_cast<size_t>(d)], block.cols(),
                      block.rows()) = block.transpose();
        }
        return true;
    }

private:
    using Key = std::pair<int64_t, int64_t>;
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return static_cast<std::size_t>(k.first) * 0x9e3779b97f4a7c15ULL ^
                   static_cast<std::size_t>(k.second);
        }
    };

    std::unordered_map<Key, DenseMatrix, KeyHash> blocks_;
    std::unordered_set<Key, KeyHash> expectedKeys_;
    int64_t expectedBlockCount_ = 0;
    std::atomic<bool> initialized_{false};
    mutable std::mutex blocksMtx_;
};

} // namespace aams
