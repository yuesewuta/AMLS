// 大矩阵参考解工具（SLEPc 思路的本仓库等价实现，Windows 无 conda petsc4py
// 故自建）：Krylov-Schur（Spectra）+ 移位反演接 METIS 排序的稀疏 LDLT。
// scipy LOBPCG+AMG 在 20万–100万 阶上收敛过慢（dK 5 小时未完成），
// 本工具一次分解 + 每次迭代一次稀疏三角回代，分钟级完成。
//
// 输出耦合部分的前 nev 阶特征值（升序，每行一个）；解耦自由度（仅对角
// 非零的行列）在此剥离——其特征值 lambda = K_ii 由 make_reference.py
// 解析并回（两者用同一判据，多重性才完整）。
//
// 用法: reference_tool --K K.mtx [--nev 200] [--ncv 0] [--out out.txt]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Spectra/SymEigsShiftSolver.h>

#include "io/matrix_market.hpp"
#include "metis.h"

using Clock = std::chrono::steady_clock;
using Sparse64 = amls::DefaultTraits::SparseMatrix;
using Sparse32 = Eigen::SparseMatrix<double, 0, int>;
using VectorXd = Eigen::VectorXd;

namespace {

double secondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/// METIS 嵌套剖分排序（Eigen OrderingType 接口）。工程内 metis 编译为
/// 64 位 idx_t，与 Eigen 5 自带 MetisSupport（按 StorageIndex 直传）不
/// 兼容，故自写转换层。
struct MetisNdOrdering {
    using StorageIndex = int;
    using PermutationType =
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;

    template <typename MatrixType>
    void operator()(const MatrixType& mat, PermutationType& perm) {
        const int n = static_cast<int>(mat.rows());
        std::vector<idx_t> xadj(static_cast<size_t>(n) + 1, 0);
        std::vector<idx_t> adjncy;
        adjncy.reserve(static_cast<size_t>(mat.nonZeros()));
        for (int c = 0; c < n; ++c) {
            for (typename MatrixType::InnerIterator it(mat, c); it; ++it) {
                if (it.row() == c)
                    continue;   // METIS 不要自环
                adjncy.push_back(static_cast<idx_t>(it.row()));
            }
            xadj[static_cast<size_t>(c) + 1] =
                static_cast<idx_t>(adjncy.size());
        }
        std::vector<idx_t> p(static_cast<size_t>(n)), ip(static_cast<size_t>(n));
        idx_t nIdx = static_cast<idx_t>(n);
        if (METIS_NodeND(&nIdx, xadj.data(), adjncy.data(), nullptr, nullptr,
                         p.data(), ip.data()) != METIS_OK)
            throw std::runtime_error("METIS_NodeND 失败");
        perm.resize(n);
        for (int i = 0; i < n; ++i)
            perm.indices()[i] = static_cast<int>(p[static_cast<size_t>(i)]);
    }
};

using Ldl = Eigen::SimplicialLDLT<Sparse32, Eigen::Lower, MetisNdOrdering>;

/// 与 make_reference.py 相同的对称化规则。注意 readMatrixMarket 对
/// symmetric 头已经镜像展开，直接返回即可；general 全存储上下三角各存
/// 一份（thermomech_dK），取 (A+Aᵀ)/2 消除装配舍入差。
Sparse64 symmetrize(const Sparse64& A, bool headerSymmetric) {
    if (headerSymmetric)
        return A;
    Sparse64 T(A.transpose());
    double scale = 0.0, asym = 0.0;
    for (int64_t c = 0; c < A.outerSize(); ++c) {
        for (Sparse64::InnerIterator it(A, c); it; ++it) {
            scale = std::max(scale, std::abs(it.value()));
            asym = std::max(asym, std::abs(it.value() - T.coeff(it.row(), c)));
        }
    }
    if (asym / (scale == 0.0 ? 1.0 : scale) > 1e-6)
        throw std::runtime_error("general 存储但数值不对称");
    return static_cast<Sparse64>((A + T) * 0.5);
}

/// 剥离仅对角非零的行/列,返回耦合子矩阵；keep 为保留行到原索引的映射。
Sparse32 stripDecoupled(const Sparse64& A, std::vector<int>& keep) {
    const int64_t n = A.rows();
    std::vector<int64_t> offDiag(static_cast<size_t>(n), 0);
    for (int64_t c = 0; c < A.outerSize(); ++c)
        for (Sparse64::InnerIterator it(A, c); it; ++it)
            if (it.row() != c) {
                ++offDiag[static_cast<size_t>(it.row())];
                ++offDiag[static_cast<size_t>(c)];
            }
    keep.clear();
    for (int64_t i = 0; i < n; ++i)
        if (offDiag[static_cast<size_t>(i)] > 0)
            keep.push_back(static_cast<int>(i));
    std::vector<int> newIdx(static_cast<size_t>(n), -1);
    for (size_t j = 0; j < keep.size(); ++j)
        newIdx[static_cast<size_t>(keep[j])] = static_cast<int>(j);
    std::vector<Eigen::Triplet<double, int>> trips;
    trips.reserve(static_cast<size_t>(A.nonZeros()));
    for (int64_t c = 0; c < A.outerSize(); ++c)
        for (Sparse64::InnerIterator it(A, c); it; ++it) {
            const int r = newIdx[static_cast<size_t>(it.row())];
            const int cc = newIdx[static_cast<size_t>(it.col())];
            if (r >= 0 && cc >= 0)
                trips.emplace_back(r, cc, it.value());
        }
    Sparse32 B(static_cast<int>(keep.size()), static_cast<int>(keep.size()));
    B.setFromTriplets(trips.begin(), trips.end());
    return B;
}

struct LdlShiftSolve {
    using Scalar = double;
    const Ldl* ldl = nullptr;
    int n = 0;

    int rows() const { return n; }
    void set_shift(const double&) const {}   // 分解已按同一 sigma 预先完成
    void perform_op(const double* x, double* y) const {
        Eigen::Map<const VectorXd> xs(x, n);
        Eigen::Map<VectorXd> ys(y, n);
        ys = ldl->solve(xs);
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string kFile, outFile;
    int64_t nev = 200;
    int ncvArg = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc)
                throw std::runtime_error("参数缺值: " + s);
            return argv[i];
        };
        if (s == "--K") kFile = next();
        else if (s == "--nev") nev = std::stoll(next());
        else if (s == "--ncv") ncvArg = std::stoi(next());
        else if (s == "--out") outFile = next();
        else throw std::runtime_error("未知参数: " + s);
    }
    if (kFile.empty())
        throw std::runtime_error("用法: reference_tool --K K.mtx [--nev N] "
                                 "[--ncv N] [--out out.txt]");

    auto t0 = Clock::now();
    const bool headerSymmetric = [kFile] {
        std::ifstream in(kFile);
        std::string line;
        std::getline(in, line);
        for (char& ch : line) ch = static_cast<char>(std::tolower(ch));
        return line.find("symmetric") != std::string::npos;
    }();
    Sparse64 K = amls::readMatrixMarket<amls::DefaultTraits>(kFile);
    K = symmetrize(K, headerSymmetric);
    std::cout << "[ref] 读入 " << K.rows() << " 阶 nnz=" << K.nonZeros()
              << " (" << secondsSince(t0) << "s)\n"
              << std::flush;

    std::vector<int> keep;
    Sparse32 Kc = stripDecoupled(K, keep);
    std::cout << "[ref] 剥离解耦自由度 "
              << (K.rows() - static_cast<int64_t>(keep.size()))
              << " 个,耦合部分 " << Kc.rows() << " 阶\n"
              << std::flush;

    // sigma = -1e-8·scale（与 make_reference.py 一致；半正定 K 保持 PD）
    double scale = 0.0;
    for (int c = 0; c < Kc.outerSize(); ++c)
        for (Sparse32::InnerIterator it(Kc, c); it; ++it)
            scale = std::max(scale, std::abs(it.value()));
    if (scale < 1.0) scale = 1.0;
    const double sigma = -1e-8 * scale;

    auto t1 = Clock::now();
    Ldl ldl;
    Sparse32 Kshift = Kc;
    for (int j = 0; j < Kshift.outerSize(); ++j)
        for (Sparse32::InnerIterator it(Kshift, j); it; ++it)
            if (it.row() == j) it.valueRef() -= sigma;
    ldl.compute(Kshift);
    if (ldl.info() != Eigen::Success)
        throw std::runtime_error("LDLT 分解失败");
    const double memGB =
        static_cast<double>(ldl.matrixL().nestedExpression().nonZeros()) *
        (8.0 + sizeof(int)) / 1e9;
    std::cout << "[ref] LDLT(METIS) 完成: nnz(L)="
              << ldl.matrixL().nestedExpression().nonZeros() << " 约"
              << memGB << "GB (" << secondsSince(t1) << "s)\n"
              << std::flush;

    LdlShiftSolve op;
    op.ldl = &ldl;
    op.n = Kc.rows();
    const int ncv = ncvArg > 0
                        ? ncvArg
                        : std::min<int>(static_cast<int>(Kc.rows()),
                                        static_cast<int>(nev) * 3 + 20);
    auto t2 = Clock::now();
    Spectra::SymEigsShiftSolver<LdlShiftSolve> eigs(op, static_cast<int>(nev),
                                                   ncv, sigma);
    eigs.init();
    const int nconv = eigs.compute(Spectra::SortRule::LargestMagn, 2000, 1e-11);
    if (eigs.info() != Spectra::CompInfo::Successful || nconv < nev)
        throw std::runtime_error("Spectra 未收敛: " + std::to_string(nconv) +
                                 "/" + std::to_string(nev));
    auto vals = eigs.eigenvalues();
    std::sort(vals.data(), vals.data() + vals.size());
    std::cout << "[ref] Krylov-Schur 完成 (" << secondsSince(t2)
              << "s), mu 范围 [" << vals[0] << ", " << vals[vals.size() - 1]
              << "]\n";

    std::ostream* os = &std::cout;
    std::ofstream file;
    if (!outFile.empty()) {
        file.open(outFile);
        os = &file;
    }
    for (int i = 0; i < nev; ++i)
        (*os) << std::setprecision(17) << vals[i] << "\n";
    std::cout << "[ref] 完成,共 " << nev << " 阶 -> "
              << (outFile.empty() ? "stdout" : outFile) << " (总 "
              << secondsSince(t0) << "s)\n";
    return 0;
}
