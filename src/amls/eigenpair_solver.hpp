#pragma once
// 特征对求解模块（step 3 的求解内核；Hyun 2020 式 4/13/21 的数值实现）。
//
//   1) 局部子问题：K̂_ii z = w M̂_ii z（式 4 子结构正则模态 / 式 13 界面子空间）
//      —— 稠密 dsygvx/dsygvd 与 SIL 移位反演 Lanczos 双内核，按铅笔稀疏度
//      自动选择（实测 crossover：稀疏 nnz/n²≲15% 且 n≳200 时 SIL 快 5–20×）。
//   2) 降阶问题：K̃ q = w M̃ q（式 21）
//
// 线程安全：solveLocal* 无共享状态，可在线程池中并发调用；
// 并发阶段应将 OpenBLAS 线程数设为 1（见 solver 编排），避免过度订阅。
#include <cstdint>

#include "amls/shift_invert_lanczos.hpp"
#include "core/types.hpp"
#include "linalg/dense_eigensolver.hpp"

namespace amls {

class EigenpairSolver {
public:
    using DenseMatrix = DefaultTraits::DenseMatrix;
    using Vector      = DefaultTraits::Vector;
    using Scalar      = DefaultTraits::Scalar;

    explicit EigenpairSolver(IDenseEigensolver<DefaultTraits>& dense)
        : dense_(dense) {}

    /// SIL 开关：0=自动（按稀疏度），1=强制 SIL，-1=强制稠密。
    int silMode = 0;

    /// 全部特征对，升序；K/M 被覆盖。
    void solveLocalAll(DenseMatrix& K, DenseMatrix& M, Vector& w,
                       DenseMatrix& V) const {
        dense_.solveAll(K, M, w, V);
    }

    /// 半开区间 (lo, hi] 内全部特征对（按值），升序；K/M 被覆盖。
    void solveLocalInterval(DenseMatrix& K, DenseMatrix& M, Scalar lo,
                            Scalar hi, Vector& w, DenseMatrix& V) const {
        dense_.solveInterval(K, M, lo, hi, w, V);
    }

    /// 索引区间 [il, iu]（1-based，含端点）特征对，升序；K/M 被覆盖。
    /// il>1 时 SIL 单移位无法定位，自动回退稠密。
    /// 已知局限：SIL（单向量 Lanczos）对简并特征值只返回一份副本
    /// （实测 bcsstk18 的 729 重 λ=1 簇被吞）——因此 solveLocalInterval
    /// 恒走稠密 dsygvx（Sturm 计数二分，对多重性安全）；本函数仅用于
    /// 趟 A 的 λ̂ 估计，阈值估计偏大属安全方向，故保留 SIL 加速。
    void solveLocalIndex(DenseMatrix& K, DenseMatrix& M, int64_t il,
                         int64_t iu, Vector& w, DenseMatrix& V) const {
        const int64_t n = K.rows();
        if (il == 1 && useSil_(K, M)) {
            const int64_t k = iu - il + 1;
            if (k < (n * 4) / 5) {
                sil::solveIndex(K, M, il, iu, w, V);
                if (w.size() == static_cast<int64_t>(k))
                    return;
            }
        }
        dense_.solveIndex(K, M, il, iu, w, V);
    }

private:
    /// 稀疏度判据：K 的 nnz/n² < 0.15 且 n ≥ 200（实测 201–500 桶 SIL 7.3×，
    /// 稠密填充块持平 → 稀疏度才是选择变量，不是纯规模）。
    bool useSil_(const DenseMatrix& K, const DenseMatrix& M) const {
        if (silMode > 0)
            return true;
        if (silMode < 0)
            return false;
        const int64_t n = K.rows();
        if (n < 200)
            return false;
        int64_t nnz = 0;
        for (int64_t j = 0; j < K.size(); ++j)
            if (K.data()[j] != 0.0)
                ++nnz;
        return static_cast<double>(nnz) <
               0.15 * static_cast<double>(n) * static_cast<double>(n);
    }

    IDenseEigensolver<DefaultTraits>& dense_;
};

} // namespace amls
