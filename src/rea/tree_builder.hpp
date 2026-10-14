#pragma once
// Abaqus 风格树构造管线（REA-AMLS）：
//   METIS_NodeND 排序 → 符号分析(etree+columnCount) → exact 基础超节点
//   → Abaqus 聚合(Collapse2+AggregateLeaves+Cutoff) → SeparatorTree。
//
// 对应 Abaqus PAMS 链路 [R]（见 docs/REA版AMLS设计_树构造与Abaqus一致.md）：
//   GenerateOrder → BuildTreeAfterMetis → CreateFundamentalSuperNodes
//   → Collapse2/AggregateLeaves → ComputeSubTreesNew。
#include <cstdint>
#include <string>
#include <vector>

#include "amls/separator_tree.hpp"
#include "partition/partitioner.hpp"

namespace amls {
namespace rea {

struct AbaqusTreeOptions {
    // ---- METIS 排序 ----
    int metisSeed = 42;

    // ---- 基础超节点 ----
    int64_t maxBlockDofs = 4096;   // Abaqus UPPER_CUTOFF（dat "MAXIMUM SIZE OF A
                                   // SUBSTRUCTURE 4096"）

    // ---- Abaqus 聚合参数 ----
    int64_t collapseMaxDofs = 120;   // LEAF_SN_LIMIT
    int64_t leafAggrMaxDofs = 20;    // LEAF_AGGR_LIMIT
    int64_t minBlockDofs   = 80;     // MIN_SN_SIZE
    int     collapseMode   = 2;      // SN_COLLAPSE
    bool    aggregateLeaves = true;  // AggregateLeaves

    // ---- 多根处理 ----
    // Abaqus 支持森林（numberOfTrees），但 SeparatorTree/solver 需单根。
    // 多根时把全部根合成一个超根（DOF = 各根之和）。superRootDofs 为超根
    // 的 DOF 数实际上限（仅用于 reserve，不影响正确性）；0 表示"多根报错"。
    int64_t superRootDofs = 0;
};

/// 与 Abaqus 一致的树构造。A = |K|+|M| 结构并集。
/// 返回 SeparatorTree（原始坐标 indices）+ perm[new]=old。
IPartitioner<DefaultTraits>::Result buildAbaqusTree(
    const DefaultTraits::SparseMatrix& A, const AbaqusTreeOptions& opt);

}  // namespace rea
}  // namespace amls