#pragma once
// 符号分析：对给定排列的对称矩阵做符号 Cholesky/LDLᵀ 分析，产出
// 消去树 (etree)、填充后列计数 (columnCount) 等超节点识别所需元数据。
//
// 参考设计文档 §4.3：从 NodeND 排列出发，对 pattern(K)∪pattern(M) 的并集
// 结构做符号分解（只追踪非零位置，不计算数值）。输出：
//   permFinal / inversePerm  排列
//   parentColumn[j]          标量消去树父列（-1 = 根）
//   columnCount[j]           L(:,j) 含对角的结构非零数
//   componentRoots           结构连通分量根
//
// 实现：用 Eigen SimplicialLDLT::analyzePattern + matrixL() 提取 L 结构，
// 不引入 CHOLMOD 依赖（设计文档 §4.4 的 CHOLMOD 为可选项）。
#include <cstdint>
#include <vector>

#include <Eigen/Sparse>

#include "core/types.hpp"

namespace aams {
namespace sym {

struct SymbolicAnalysis {
    std::vector<int64_t> perm;         // perm[new] = old
    std::vector<int64_t> inversePerm;  // iperm[old] = new
    std::vector<int64_t> parentColumn; // 标量 etree 父列, -1 = 根
    std::vector<int64_t> columnCount;  // L(:,j) 结构非零数 (含对角)
    std::vector<int64_t> componentRoots;  // 每连通分量一个根列
};

/// 对 A 的结构（对称）做符号分析，使用给定排列 permND。
/// A: 对称稀疏矩阵（只用结构，忽略数值）；permND: perm[new]=old。
SymbolicAnalysis analyzeSymmetrized(
    const DefaultTraits::SparseMatrix& A,
    const std::vector<int64_t>& permND);

/// 从 Eigen SimplicialLDLT 的 matrixL() 提取 etree 与列计数。
struct FactorMeta {
    std::vector<int64_t> parentColumn;
    std::vector<int64_t> columnCount;
    std::vector<int64_t> componentRoots;
};
FactorMeta extractFactorMeta(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>& L, int64_t n);

}  // namespace sym
}  // namespace aams
