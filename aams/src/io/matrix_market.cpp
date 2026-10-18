#include "io/matrix_market.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <Eigen/Sparse>

namespace aams {

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
    NumericLine(const char* begin, const char* end)
        : cursor_(begin), end_(end) {}

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

// 数据区按行边界切片并行解析：getline+from_chars 逐行是纯 CPU 热点
// （cant_31 parse K=4.5s），分片后拼接 triplet 时保持文件顺序，使
// Eigen setFromTriplets 的重复元求和顺序与串行版逐位一致。
template <typename Index, typename Scalar>
struct ChunkResult {
    std::vector<Eigen::Triplet<Scalar, Index>> trips;
    Index entries = 0;
    std::string error;
};

template <typename Index, typename Scalar>
ChunkResult<Index, Scalar> parseChunk(const char* begin, const char* end,
                                      Index nrows, Index ncols, bool pattern) {
    ChunkResult<Index, Scalar> res;
    try {
        const char* p = begin;
        while (p != end) {
            const char* nl = static_cast<const char*>(
                memchr(p, '\n', static_cast<size_t>(end - p)));
            const char* lineEnd = nl ? nl : end;
            NumericLine numbers(p, lineEnd);
            p = nl ? nl + 1 : end;
            if (numbers.ignored()) continue;
            Index i = numbers.read<Index>();
            Index j = numbers.read<Index>();
            // Check 1-based indices before subtraction, including INT64_MIN.
            if (i <= 0 || i > nrows || j <= 0 || j > ncols)
                throw std::runtime_error("readMatrixMarket: 行/列索引越界");
            const double v = pattern ? 1.0 : numbers.read<double>();
            if (!std::isfinite(v))
                throw std::runtime_error("readMatrixMarket: 数值必须有限");
            numbers.finish();
            --i;  // 1-based -> 0-based
            --j;
            res.trips.emplace_back(i, j, static_cast<Scalar>(v));
            ++res.entries;
        }
    } catch (const std::exception& e) {
        res.error = e.what();
    }
    return res;
}

} // namespace

template <typename T>
typename T::SparseMatrix readMatrixMarket(const std::string& path) {
    using Scalar = typename T::Scalar;
    using Index  = typename T::Index;

    // 整体读入后并行解析：getline 逐行是纯 CPU 串行热点（cant_31 K
    // 文件数百 MB，parse 占 ~4.5s），分片并行可压到 ~1s。
    const bool prof = std::getenv("AAMS_PROFILE") != nullptr;
    auto clockNow = [] { return std::chrono::steady_clock::now(); };
    auto tp0 = clockNow();
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("readMatrixMarket: 无法打开 " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff fsize = in.tellg();
    if (fsize < 0)
        throw std::runtime_error("readMatrixMarket: 无法确定文件大小 " + path);
    std::string buf;
    buf.resize(static_cast<size_t>(fsize));
    in.seekg(0, std::ios::beg);
    if (fsize > 0) in.read(buf.data(), fsize);
    if (in.bad())
        throw std::runtime_error("readMatrixMarket: 文件读取失败 " + path);
    auto tp1 = clockNow();

    const char* fileBegin = buf.data();
    const char* fileEnd = fileBegin + buf.size();
    const char* firstNl = static_cast<const char*>(
        memchr(fileBegin, '\n', buf.size()));
    std::istringstream bannerStream(std::string(
        fileBegin, firstNl ? firstNl : fileEnd));
    const Header h = parseBanner(bannerStream);

    auto nextLineStart = [&](const char* from) {
        const char* nl = static_cast<const char*>(
            memchr(from, '\n', static_cast<size_t>(fileEnd - from)));
        return nl ? nl + 1 : fileEnd;
    };

    const char* p = fileBegin;
    // Skip comments and blank lines, then strictly parse the dimensions.
    Index nrows = -1, ncols = -1, nnz = -1;
    for (;;) {
        const char* lineEnd = nextLineStart(p);
        NumericLine numbers(p, lineEnd);
        p = lineEnd;
        if (numbers.ignored()) {
            if (p == fileEnd)
                throw std::runtime_error("readMatrixMarket: 缺少尺寸行");
            continue;
        }
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

    // 数据区 [p, fileEnd) 按行边界切片并行解析；分片数约等于核数
    // （≥1MB/片，避免空转），行边界切分保证不撕裂记录。
    const size_t bytes = static_cast<size_t>(fileEnd - p);
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    size_t nChunks = std::min<size_t>(hw, bytes / (1u << 20) + 1);
    if (nChunks < 1) nChunks = 1;
    std::vector<const char*> bounds(nChunks + 1, fileEnd);
    bounds[0] = p;
    for (size_t c = 1; c < nChunks; ++c) {
        const char* target = p + bytes * c / nChunks;
        const char* nl = static_cast<const char*>(
            memchr(target, '\n', static_cast<size_t>(fileEnd - target)));
        bounds[c] = nl ? nl + 1 : fileEnd;
    }
    std::vector<ChunkResult<Index, Scalar>> results(nChunks);
    std::vector<std::thread> workers;
    const size_t nWorkers =
        std::min<size_t>(static_cast<size_t>(hw), nChunks);
    for (size_t w = 0; w < nWorkers; ++w)
        workers.emplace_back([&, w] {
            for (size_t c = w; c < nChunks; c += nWorkers)
                results[c] = parseChunk<Index, Scalar>(
                    bounds[c], bounds[c + 1], nrows, ncols, h.pattern);
        });
    for (auto& w : workers) w.join();
    for (const auto& r : results)
        if (!r.error.empty()) throw std::runtime_error(r.error);

    // 按文件顺序拼接（重复元求和顺序与串行版逐位一致）；对称阵在此
    // 展开成对角镜像。分片 triplet 随取随释放，压低峰值内存。
    trips.reserve(static_cast<size_t>(nnz) * expansion);
    Index entries = 0;
    for (auto& r : results) {
        for (const auto& t : r.trips) {
            trips.push_back(t);
            if (h.symmetric && t.row() != t.col())
                trips.emplace_back(t.col(), t.row(), t.value());
        }
        entries += r.entries;
        std::vector<Eigen::Triplet<Scalar, Index>>().swap(r.trips);
    }
    std::string().swap(buf);  // 数据已进 trips，释放文件缓冲
    auto tp2 = clockNow();
    if (entries > nnz)
        throw std::runtime_error("readMatrixMarket: 记录数超过声明的 nnz");
    if (entries != nnz)
        throw std::runtime_error("readMatrixMarket: 记录数少于声明的 nnz");

    typename T::SparseMatrix A(nrows, ncols);
    A.setFromTriplets(trips.begin(), trips.end());
    if (prof)
        std::fprintf(stderr,
                     "[prof] mm read=%.2fs parse+concat=%.2fs "
                     "setFromTriplets=%.2fs\n",
                     std::chrono::duration<double>(tp1 - tp0).count(),
                     std::chrono::duration<double>(tp2 - tp1).count(),
                     std::chrono::duration<double>(clockNow() - tp2).count());
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

} // namespace aams
