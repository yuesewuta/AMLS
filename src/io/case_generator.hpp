#pragma once
// 谱已知的代数百科算例生成器（设计文档 §4.2 / 验证方案 §2.1）。
// 全部为离散图 Laplacian 型算例,与任何物理来源无关。
#include <cstdint>

#include "core/types.hpp"

namespace amls {

struct PencilPair {
    DefaultTraits::SparseMatrix K;
    DefaultTraits::SparseMatrix M;
};

/// 一维链：K = 三对角 [-1, 2, -1]（Dirichlet 端,n×n,n 个特征值）。
/// M = I 或一致（consistent）三对角变体 [1/6, 2/3, 1/6]。
/// 闭式谱（M=I）:λ_j = 2 − 2cos(jπ/(n+1)), j = 1..n。
/// 闭式谱（一致 M）:λ_j = 6(1−cos θⱼ)/(2+cos θⱼ), θⱼ = jπ/(n+1)。
PencilPair pathGraph(int64_t n, bool consistentMass = false);

/// 一维链闭式谱（升序,j=1..n）。
DefaultTraits::Vector pathGraphSpectrum(int64_t n, bool consistentMass = false);

/// 二维规则网格：nx×ny 个内点,5 点 stencil,K 为块三对角,边界 Dirichlet。
/// M = I。
/// 闭式谱:λ_jk = 2−2cos(jπ/(nx+1)) + 2−2cos(kπ/(ny+1))。
PencilPair gridGraph2d(int64_t nx, int64_t ny);

/// 二维网格闭式谱（升序）。
DefaultTraits::Vector gridGraph2dSpectrum(int64_t nx, int64_t ny);

/// 三维规则网格：nx×ny×nz 个内点，7 点 stencil，边界 Dirichlet。M = I。
/// 闭式谱：λ = Σ(2−2cos(jπ/(n_α+1)))。供 50 万 DOF 级测试（解析参考解）。
PencilPair gridGraph3d(int64_t nx, int64_t ny, int64_t nz);

/// 三维网格闭式谱（升序）。
DefaultTraits::Vector gridGraph3dSpectrum(int64_t nx, int64_t ny,
                                          int64_t nz);

} // namespace amls
