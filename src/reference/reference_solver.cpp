#include <reference/reference_solver.hpp>

#include <stdexcept>

#include <algorithm>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include "linalg/dense_eigensolver.hpp"

namespace amls {

ReferenceSolver::Vector
ReferenceSolver::smallestEigenvalues(const SparseMatrix& K,
                                     const SparseMatrix& M, int64_t nev,
                                     DenseMatrix* V) {
    if (nev <= 0)
        throw std::invalid_argument("nev 必须为正");
    const int64_t n = K.rows();
    nev = std::min<int64_t>(nev, n);

    if (n <= denseMax_) {
        // 金标准：稠密全谱
        DenseMatrix Kd(K), Md(M);
        Vector w;
        DenseMatrix Veig;
        LapackDenseEigensolver<DefaultTraits> lapack;
        lapack.solveAll(Kd, Md, w, Veig);
        Vector out = w.head(nev);
        if (V)
            *V = Veig.leftCols(nev);
        return out;
    }

    // Spectra 移位反演:σ=0（K 需 SPD）,求 ν=1/λ 最大 ↔ λ 最小。
    // Spectra 内部使用 int 索引的稀疏矩阵,此处转一份（参考解路径,一次性开销）。
    Eigen::SparseMatrix<double> Ki(K.rows(), K.cols());
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(K.nonZeros()));
        for (int64_t c = 0; c < K.outerSize(); ++c)
            for (SparseMatrix::InnerIterator it(K, c); it; ++it)
                trips.emplace_back(static_cast<int>(it.row()),
                                   static_cast<int>(it.col()), it.value());
        Ki.setFromTriplets(trips.begin(), trips.end());
    }
    Eigen::SparseMatrix<double> Mi(M.rows(), M.cols());
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(M.nonZeros()));
        for (int64_t c = 0; c < M.outerSize(); ++c)
            for (SparseMatrix::InnerIterator it(M, c); it; ++it)
                trips.emplace_back(static_cast<int>(it.row()),
                                   static_cast<int>(it.col()), it.value());
        Mi.setFromTriplets(trips.begin(), trips.end());
    }
    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;
    OpType op(Ki, Mi);
    BOpType bop(Mi);
    Spectra::SymGEigsShiftSolver<OpType, BOpType,
                                 Spectra::GEigsMode::ShiftInvert>
        eigs(op, bop, static_cast<int>(nev),
             std::min<int>(static_cast<int>(nev) * 3 + 20, static_cast<int>(n)),
             0.0);
    eigs.init();
    const int nconv = eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-12);
    if (eigs.info() != Spectra::CompInfo::Successful || nconv < nev)
        throw std::runtime_error("Spectra 移位反演未收敛（" +
                                 std::to_string(nconv) + "/" +
                                 std::to_string(nev) + "）");
    // Spectra 在 ShiftInvert 模式下按变换后 λ 的 LargestMagn 输出（降序）;
    // 收敛集即最靠近 σ 的最小 λ,此处统一按升序重排。
    Vector vals = eigs.eigenvalues().head(nev);
    DenseMatrix vecs = eigs.eigenvectors().leftCols(nev);
    std::vector<int> ord(static_cast<size_t>(nev));
    for (int i = 0; i < static_cast<int>(nev); ++i)
        ord[static_cast<size_t>(i)] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return vals[a] < vals[b];
    });
    Vector out(nev);
    DenseMatrix vs(vecs.rows(), nev);
    for (int64_t j = 0; j < nev; ++j) {
        out[j] = vals[ord[static_cast<size_t>(j)]];
        vs.col(j) = vecs.col(ord[static_cast<size_t>(j)]);
    }
    if (V)
        *V = vs;
    return out;
}

} // namespace amls
