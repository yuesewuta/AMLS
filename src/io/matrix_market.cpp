#include "io/matrix_market.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Sparse>

namespace amls {

namespace {

struct Header {
    bool symmetric = false;
    bool pattern   = false;
    bool integer   = false;
};

Header parseBanner(std::istream& in) {
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("readMatrixMarket: 空文件");
    std::istringstream ss(line);
    std::string banner, object, format, field, symmetry;
    ss >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket")
        throw std::runtime_error("readMatrixMarket: 缺少 banner");
    if (object != "matrix")
        throw std::runtime_error("readMatrixMarket: 仅支持 matrix 对象");
    if (format != "coordinate")
        throw std::runtime_error("readMatrixMarket: 仅支持 coordinate 格式");
    if (symmetry != "symmetric" && symmetry != "general")
        throw std::runtime_error("readMatrixMarket: 仅支持 symmetric/general");
    if (field != "real" && field != "double" && field != "pattern" &&
        field != "integer")
        throw std::runtime_error("readMatrixMarket: 仅支持 real/integer/pattern");
    Header h;
    h.symmetric = (symmetry == "symmetric");
    h.pattern   = (field == "pattern");
    h.integer   = (field == "integer");
    return h;
}

} // namespace

template <typename T>
typename T::SparseMatrix readMatrixMarket(const std::string& path) {
    using Scalar = typename T::Scalar;
    using Index  = typename T::Index;

    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("readMatrixMarket: 无法打开 " + path);

    Header h = parseBanner(in);

    std::string line;
    // 跳过注释与空行,读尺寸行
    Index nrows = -1, ncols = -1, nnz = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%')
            continue;
        std::istringstream ss(line);
        ss >> nrows >> ncols >> nnz;
        break;
    }
    if (nrows < 0)
        throw std::runtime_error("readMatrixMarket: 缺少尺寸行");

    std::vector<Eigen::Triplet<Scalar, Index>> trips;
    trips.reserve(static_cast<size_t>(nnz) * (h.symmetric ? 2 : 1));

    Index i, j;
    double v;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%')
            continue;
        std::istringstream ss(line);
        ss >> i >> j;
        if (h.pattern) {
            v = 1.0;
        } else {
            ss >> v;
        }
        --i;  // 1-based -> 0-based
        --j;
        trips.emplace_back(i, j, static_cast<Scalar>(v));
        if (h.symmetric && i != j)
            trips.emplace_back(j, i, static_cast<Scalar>(v));
    }

    typename T::SparseMatrix A(nrows, ncols);
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

template <typename T>
void writeMatrixMarket(const std::string& path,
                       const typename T::SparseMatrix& A) {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("writeMatrixMarket: 无法写 " + path);
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    out << A.rows() << " " << A.cols() << " " << A.nonZeros() << "\n";
    out.precision(17);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename T::SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col())
                out << it.row() + 1 << " " << it.col() + 1 << " "
                    << it.value() << "\n";
        }
    }
}

template typename DefaultTraits::SparseMatrix
readMatrixMarket<DefaultTraits>(const std::string&);
template void writeMatrixMarket<DefaultTraits>(
    const std::string&, const DefaultTraits::SparseMatrix&);

} // namespace amls
