#pragma once
// MatrixMarket 读写（设计文档 §4.2）
#include <string>

#include "core/types.hpp"

namespace aams {

/// 读 MatrixMarket `coordinate` 格式;`symmetric` 标志展开为全矩阵。
/// 实数矩阵;pattern/integer 按 double 读。重复坐标累加。
template <typename T>
typename T::SparseMatrix readMatrixMarket(const std::string& path);

/// 写 coordinate + symmetric 格式（只写下三角）。
template <typename T>
void writeMatrixMarket(const std::string& path,
                       const typename T::SparseMatrix& A);

} // namespace aams
