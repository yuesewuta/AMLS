#pragma once
// Abaqus 风格树聚合：在 exact 基础超节点之上，按 Abaqus AMS（PAMS）的
// Collapse2 / AggregateLeaves / 大小限制逻辑把细碎块聚合成实用的子结构块。
//
// 逆向依据（见 docs/REA版AMLS设计_树构造与Abaqus一致.md §1.2）：
//   - Collapse2:      单子节点且聚合大小 < LEAF_SN_LIMIT(120) → 折叠进父
//   - AggregateLeaves:全部子为叶且聚合大小 < LEAF_AGGR_LIMIT(20) → 聚合成叶
//   - UPPER_CUTOFF=4096 / MIN_SN_SIZE=80 等限制
// 输入输出都是 BlockForest（后序编号、祖先 id>后代、DOF 全覆盖互斥），
// 聚合后保持全部不变式（含原始消去边祖先可比）。
#include <cstdint>
#include <vector>

#include "partition/supernode_builder.hpp"

namespace amls {
namespace sym {

struct AggregationParams {
    int64_t collapseMaxDofs = 120;   // Abaqus LEAF_SN_LIMIT（0x78）：单子链折叠上限
    int64_t leafAggrMaxDofs = 20;    // Abaqus LEAF_AGGR_LIMIT（0x14）：叶子聚合上限
    int64_t maxBlockDofs   = 4096;   // Abaqus UPPER_CUTOFF（0x1000）：子结构 DOF 上限
    int64_t minBlockDofs   = 80;     // Abaqus MIN_SN_SIZE（0x50）：聚合目标下限
    int     collapseMode   = 2;      // Abaqus SN_COLLAPSE
    bool    collapseL2R    = false;  // Abaqus COLLAPSE_L2R
    bool    aggregateLeaves = true;  // Abaqus AggregateLeaves
    // 单子链折叠: 0=不折叠,1=无条件,2=大小<collapseMaxDofs（Abaqus 默认行为）
};

/// 对基础块森林做 Abaqus 风格聚合。返回聚合后的森林（重新后序编号）。
BlockForest aggregateAbaqusStyle(const BlockForest& base,
                                 const AggregationParams& p = {});

/// 与 Abaqus Dump 对齐的诊断：块大小分布 / 单子链数 / 最大深度。
struct AggregationStats {
    int64_t blocks = 0, roots = 0, leaves = 0, singleChildren = 0, maxDepth = 0;
    std::vector<std::pair<int64_t, int64_t>> sizeHistogram;  // {size, count} 升序
};
AggregationStats aggregationStats(const BlockForest& forest);

/// 打印聚合统计（stderr，对齐 Abaqus 树形诊断）。
void dumpAggregationStats(const BlockForest& forest, const char* tag);

}  // namespace sym
}  // namespace amls