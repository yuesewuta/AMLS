#include "io/matrix_market.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Sparse>

namespace amls {

namespace {

struct Header {
    bool symmetric = false;
    bool pattern   = false;
};

Header parseBanner(std::istream& in) {
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("readMatrixMarket: 空文件");
    std::istringstream ss(line);
    std::string banner, object, format, field, symmetry, extra;
    if (!(ss >> banner >> object >> format >> field >> symmetry) || (ss >> extra))
        throw std::runtime_error("readMatrixMarket: banner 格式错误");
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
    return h;
}

// MatrixMarket uses ASCII whitespace; avoid locale lookups in the numeric hot path.
bool isSpace(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
           c == '\f' || c == '\v';
}

class NumericLine {
public:
    explicit NumericLine(const std::string& line)
        : cursor_(line.data()), end_(line.data() + line.size()) {}

    bool ignored() {
        skipSpace();
        return cursor_ == end_ || *cursor_ == '%';
    }

    template <typename Number>
    Number read() {
        skipSpace();
        // Unlike stream extraction, from_chars does not accept a leading '+'.
        if (cursor_ != end_ && *cursor_ == '+') {
            ++cursor_;
            if (cursor_ == end_ || *cursor_ == '+' || *cursor_ == '-')
                throw std::runtime_error("readMatrixMarket: 非法数值符号");
        }
        Number value{};
        const auto result = std::from_chars(cursor_, end_, value);
        if (result.ec != std::errc{} ||
            (result.ptr != end_ && !isSpace(*result.ptr)))
            throw std::runtime_error("readMatrixMarket: 缺少、非法或溢出的数值");
        cursor_ = result.ptr;
        return value;
    }

    void finish() {
        skipSpace();
        if (cursor_ != end_)
            throw std::runtime_error("readMatrixMarket: 行中存在多余字段");
    }

private:
    void skipSpace() {
        while (cursor_ != end_ && isSpace(*cursor_))
            ++cursor_;
    }

    const char* cursor_;
    const char* end_;
};

} // namespace

template <typename T>
typename T::SparseMatrix readMatrixMarket(const std::string& path) {
    using Scalar = typename T::Scalar;
    using Index  = typename T::Index;

    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("readMatrixMarket: 无法打开 " + path);

    const Header h = parseBanner(in);

    std::string line;
    // Skip comments and blank lines, then strictly parse the dimensions.
    Index nrows = -1, ncols = -1, nnz = -1;
    while (std::getline(in, line)) {
        NumericLine numbers(line);
        if (numbers.ignored())
            continue;
        nrows = numbers.read<Index>();
        ncols = numbers.read<Index>();
        nnz = numbers.read<Index>();
        numbers.finish();
        break;
    }
    if (nrows < 0 || ncols < 0 || nnz < 0)
        throw std::runtime_error("readMatrixMarket: 缺少尺寸行或尺寸/nnz 为负");
    // Eigen allocates ncols + 1 outer indices and, for triplets, nrows + 1
    // outer indices in a temporary transposed matrix.
    if (nrows >= (std::numeric_limits<Eigen::Index>::max)() ||
        ncols >= (std::numeric_limits<Eigen::Index>::max)())
        throw std::runtime_error("readMatrixMarket: 尺寸超出 Eigen 索引范围");
    if (h.symmetric && nrows != ncols)
        throw std::runtime_error("readMatrixMarket: symmetric 矩阵必须为方阵");
    if (nnz != 0 && (nrows == 0 || ncols == 0))
        throw std::runtime_error("readMatrixMarket: 空矩阵不能包含非零元记录");

    std::vector<Eigen::Triplet<Scalar, Index>> trips;
    const size_t expansion = h.symmetric ? 2 : 1;
    // Check before multiplying or reserving, including Eigen's stored nnz index.
    if (static_cast<uintmax_t>(nnz) > trips.max_size() / expansion ||
        nnz > (std::numeric_limits<Index>::max)() / static_cast<Index>(expansion))
        throw std::runtime_error("readMatrixMarket: nnz 超出可存储范围");
    trips.reserve(static_cast<size_t>(nnz) * expansion);

    Index entries = 0;
    while (std::getline(in, line)) {
        NumericLine numbers(line);
        if (numbers.ignored())
            continue;
        if (entries == nnz)
            throw std::runtime_error("readMatrixMarket: 记录数超过声明的 nnz");
        Index i = numbers.read<Index>();
        Index j = numbers.read<Index>();
        // Check 1-based indices before subtraction, including INT64_MIN.
        if (i <= 0 || i > nrows || j <= 0 || j > ncols)
            throw std::runtime_error("readMatrixMarket: 行/列索引越界");
        const double v = h.pattern ? 1.0 : numbers.read<double>();
        if (!std::isfinite(v))
            throw std::runtime_error("readMatrixMarket: 数值必须有限");
        numbers.finish();
        --i;  // 1-based -> 0-based
        --j;
        trips.emplace_back(i, j, static_cast<Scalar>(v));
        if (h.symmetric && i != j)
            trips.emplace_back(j, i, static_cast<Scalar>(v));
        ++entries;
    }
    if (in.bad())
        throw std::runtime_error("readMatrixMarket: 文件读取失败 " + path);
    if (entries != nnz)
        throw std::runtime_error("readMatrixMarket: 记录数少于声明的 nnz");

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
    typename T::Index lowerNnz = 0;
    for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
        for (typename T::SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col())
                ++lowerNnz;
        }
    }
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    out << A.rows() << " " << A.cols() << " " << lowerNnz << "\n";
    out.precision(17);
    for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
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
