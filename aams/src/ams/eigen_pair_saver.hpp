#pragma once
// 特征对存储 —— 移植自 third_party/amlsPKU compress_matrix/eigen_pair_saver。
// 只存不解：局部特征求解（宿主 EigenpairSolver）按节点提交特征对，子空间
// 缩聚与组装按只读视图消费。CAS 状态机保证单次发布；release 供恢复阶段
// 消费完特征向量后释放内存；assembleSubspaceStiffness 把各节点特征值铺到
// 子空间刚度矩阵全局对角线（ΦᵀK̃Φ = Λ 约定，非对角恒 0）。
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/logging.hpp"
#include "core/types.hpp"

namespace aams {

class EigenPairSaver {
public:
    struct Record {
        std::vector<double> eigenvalues;
        DefaultTraits::DenseMatrix eigenvectors;
    };

    /// 通过 shared_ptr 固定记录生命周期的只读视图。
    struct View {
        std::shared_ptr<const Record> record;
        const std::vector<double>* eigenvalues() const {
            return record == nullptr ? nullptr : &record->eigenvalues;
        }
        const DefaultTraits::DenseMatrix* eigenvectors() const {
            return record == nullptr ? nullptr : &record->eigenvectors;
        }
    };

    EigenPairSaver() = default;
    EigenPairSaver(const EigenPairSaver&) = delete;
    EigenPairSaver& operator=(const EigenPairSaver&) = delete;

    /// 按节点总数一次性建槽；allowZeroEigenPairs 允许 0 模态节点（零模态
    /// 分割节点）。
    bool initialize(int64_t nodeCount, bool allowZeroEigenPairs = false) {
        std::unique_lock<std::mutex> lock(lifecycleMtx_);
        std::vector<std::unique_ptr<Slot>> newSlots;
        newSlots.reserve(static_cast<size_t>(nodeCount));
        for (int64_t i = 0; i < nodeCount; ++i)
            newSlots.emplace_back(std::make_unique<Slot>());
        slots_ = std::move(newSlots);
        allowZeroEigenPairs_ = allowZeroEigenPairs;
        return true;
    }

    /// 移动发布一个节点的特征对（每个槽位只允许成功提交一次）。
    bool setEigenPair(int64_t nodeId, std::vector<double>&& eigenvalues,
                      DefaultTraits::DenseMatrix&& eigenvectors) {
        Slot* slot = findSlot_(nodeId);
        if (slot == nullptr) {
            AMLS_LOG_ERROR("EigenPairSaver::set 失败：节点 {} 越界。", nodeId);
            return false;
        }
        if (static_cast<int64_t>(eigenvalues.size()) != eigenvectors.cols()) {
            AMLS_LOG_ERROR(
                "EigenPairSaver::set 失败：节点 {} 特征值数 {} 与特征向量列数 "
                "{} 不符。",
                nodeId, eigenvalues.size(), eigenvectors.cols());
            return false;
        }
        int32_t expected = kEmpty;
        if (!slot->state.compare_exchange_strong(expected, kSetting,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
            AMLS_LOG_ERROR(
                "EigenPairSaver::set 失败：节点 {} 槽位状态 {} 不接受提交。",
                nodeId, expected);
            return false;
        }
        auto record = std::make_shared<Record>();
        record->eigenvalues = std::move(eigenvalues);
        record->eigenvectors = std::move(eigenvectors);
        {
            std::lock_guard<std::mutex> lock(slot->mtx);
            slot->record = std::move(record);
        }
        slot->state.store(kReady, std::memory_order_release);
        return true;
    }

    /// 只读视图（shared_ptr 保活）；节点未就绪返回 false。
    bool getEigenPairView(int64_t nodeId, View& out) const {
        out.record.reset();
        const Slot* slot = findSlot_(nodeId);
        if (slot == nullptr) return false;
        if (slot->state.load(std::memory_order_acquire) != kReady) return false;
        std::lock_guard<std::mutex> lock(slot->mtx);
        if (slot->state.load(std::memory_order_acquire) != kReady ||
            slot->record == nullptr)
            return false;
        out.record = slot->record;
        return true;
    }

    /// 释放节点特征对（恢复阶段消费后调用）；幂等，未就绪返回 false。
    bool release(int64_t nodeId) {
        Slot* slot = findSlot_(nodeId);
        if (slot == nullptr) return false;
        int32_t expected = kReady;
        if (!slot->state.compare_exchange_strong(expected, kReleased,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
            return false;
        std::lock_guard<std::mutex> lock(slot->mtx);
        slot->record.reset();
        return true;
    }

    /// 清空全部槽位（不得与其他接口并发）。
    void clear() {
        std::unique_lock<std::mutex> lock(lifecycleMtx_);
        slots_.clear();
        allowZeroEigenPairs_ = false;
    }

    int64_t nodeCount() const {
        return static_cast<int64_t>(slots_.size());
    }
    bool allowZeroEigenPairs() const { return allowZeroEigenPairs_; }

    /// 组装子空间稠密刚度矩阵：按节点顺序把特征值铺到全局对角线。
    /// 约定全部 setEigenPair 完成后调用；节点缺失或 0 模态超约返回 false。
    bool assembleSubspaceStiffness(DefaultTraits::DenseMatrix& out) const {
        if (slots_.empty()) {
            AMLS_LOG_ERROR("assembleSubspaceStiffness 失败：无缓存特征对。");
            return false;
        }
        int64_t total = 0;
        for (const auto& slot : slots_) {
            // 约定组装期无并发 set/release；状态原子量做发布序校验
            if (slot->state.load(std::memory_order_acquire) != kReady ||
                slot->record == nullptr) {
                AMLS_LOG_ERROR("assembleSubspaceStiffness 失败：存在未就绪节点。");
                return false;
            }
            const auto& vals = slot->record->eigenvalues;
            if (vals.empty() && !allowZeroEigenPairs_) {
                AMLS_LOG_ERROR("assembleSubspaceStiffness 失败：存在 0 模态节点。");
                return false;
            }
            total += static_cast<int64_t>(vals.size());
        }
        if (total == 0) {
            AMLS_LOG_ERROR("assembleSubspaceStiffness 失败：全部节点 0 模态。");
            return false;
        }
        out = DefaultTraits::DenseMatrix::Zero(total, total);
        int64_t offset = 0;
        for (const auto& slot : slots_) {
            for (double lambda : slot->record->eigenvalues) {
                out(offset, offset) = lambda;
                ++offset;
            }
        }
        return true;
    }

private:
    static constexpr int32_t kEmpty = 0;
    static constexpr int32_t kSetting = 1;
    static constexpr int32_t kReady = 2;
    static constexpr int32_t kReleased = 3;

    struct Slot {
        mutable std::mutex mtx;
        std::atomic<int32_t> state{kEmpty};
        std::shared_ptr<const Record> record;
    };

    Slot* findSlot_(int64_t nodeId) const {
        if (nodeId < 0 || nodeId >= static_cast<int64_t>(slots_.size()))
            return nullptr;
        return const_cast<Slot*>(slots_[static_cast<size_t>(nodeId)].get());
    }

    std::vector<std::unique_ptr<Slot>> slots_;
    mutable std::mutex lifecycleMtx_;  // initialize/clear 独占
    bool allowZeroEigenPairs_ = false;
};

} // namespace aams
