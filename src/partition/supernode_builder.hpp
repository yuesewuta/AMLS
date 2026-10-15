#pragma once
// 超节点/受限消去块构建：从符号分析元数据构造 AMLS 消去块树。
//
// 参考设计文档 §4.5：exact/fundamental 风格分组 + 硬拆分（maxBlockDofs）+ 
// 保留单子链 + 森林 roots。输出：
//   blocks[]: EliminationBlock {begin,end,parent,children}
//   blockOfColumn[j]: 列 j 所属块
//   roots[]: 真实根块列表（不合并多根）
//
// 分组判据（设计文档 §4.3）：
//   parent(j)=j+1 且 columnCount(j)=columnCount(j+1)+1  → 同 exact supernode
//   附加 scalarChildCount[j+1]<=1（fundamental 风格, 不跨标量树分支）
#include <cstdint>
#include <vector>

#include "partition/symbolic_analyzer.hpp"

namespace amls {
namespace sym {

struct EliminationBlock {
    int64_t begin = -1;        // 最终重排列坐标 [begin, end)
    int64_t end = -1;
    int64_t parent = -1;       // -1 = 真实根
    std::vector<int64_t> children;
};

struct BlockForest {
    std::vector<EliminationBlock> blocks;
    std::vector<int64_t> blockOfColumn;   // 列 j → 块 id
    std::vector<int64_t> roots;
    std::vector<int64_t> postorder;       // 祖先 id 恒大于后代 id
    int64_t maxOwnedDofs = 0;
};

/// 从符号分析构造受限消去块树。
/// maxBlockDofs: 每块自身 DOF 上限（硬拆分阈值，设计文档默认 1000）。
/// maxChainNodes: 单个 exact 块最多合并的连续列（节点）数，≤0 = 不限。
///   默认 4（Abaqus fca 实测其基础超节点尺寸支撑为 {3,6,9,12} DOF，
///   dp=3 时即 ≤4 节点）：既复现 Abaqus 的均匀小块分布，又避免因子
///   填充在规则网格区产生的长等结构链（cant_10 曾出现 363-DOF 巨链）。
/// 保留单子链（不收缩）、不合并多根（森林）。
BlockForest buildBlockForest(const SymbolicAnalysis& sa,
                             int64_t maxBlockDofs = 1000,
                             int64_t maxChainNodes = 4);

/// 统计: 块数/最大块/叶数/最大深度/单子节点数（诊断）。
void dumpBlockForestStats(const BlockForest& forest);

}  // namespace sym
}  // namespace amls
