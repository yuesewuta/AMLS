#include "ams/original_blocks.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "core/logging.hpp"

namespace aams {

OriginalBlocks::OriginalBlocks(const Pencil<DefaultTraits>& reordered,
                               const SeparatorTree& tree)
    : tree_(tree) {
    const int64_t n = reordered.K.rows();
    if (reordered.M.rows() != n || reordered.K.cols() != n ||
        reordered.M.cols() != n || !tree_.validate(n)) {
        AMLS_LOG_ERROR(
            "OriginalBlocks 构造失败：pencil 形状或树校验不通过 (K={}x{}, M={}x{})。",
            reordered.K.rows(), reordered.K.cols(), reordered.M.rows(),
            reordered.M.cols());
        throw std::invalid_argument("OriginalBlocks: invalid pencil or tree");
    }
    N_ = tree_.nNodes();

    idx_.resize(static_cast<size_t>(N_));
    nodeOf_.assign(static_cast<size_t>(n), -1);
    localOf_.assign(static_cast<size_t>(n), -1);
    for (int64_t i = 0; i < N_; ++i) {
        idx_[static_cast<size_t>(i)] = tree_.node(i).indices;
        auto& list = idx_[static_cast<size_t>(i)];
        std::sort(list.begin(), list.end());
        for (size_t p = 0; p < list.size(); ++p) {
            const int64_t g = list[p];
            if (g < 0 || g >= n || nodeOf_[static_cast<size_t>(g)] != -1) {
                AMLS_LOG_ERROR(
                    "OriginalBlocks 构造失败：节点 {} 索引越界或重复 (g={})。",
                    i, g);
                throw std::invalid_argument(
                    "OriginalBlocks: node indices out of range or duplicated");
            }
            nodeOf_[static_cast<size_t>(g)] = i;
            localOf_[static_cast<size_t>(g)] = static_cast<int64_t>(p);
        }
    }

    // DFS 进入/离开区间（迭代）：祖先 a 的区间严格包含后代 d 的区间。
    tin_.assign(static_cast<size_t>(N_), -1);
    tout_.assign(static_cast<size_t>(N_), -1);
    int64_t timer = 0;
    std::vector<std::pair<int64_t, size_t>> stack;
    const int64_t root = tree_.root();
    tin_[static_cast<size_t>(root)] = timer++;
    stack.emplace_back(root, 0);
    while (!stack.empty()) {
        auto& [id, ci] = stack.back();
        const std::vector<int64_t>& ch = tree_.node(id).children;
        if (ci < ch.size()) {
            const int64_t c = ch[ci++];
            tin_[static_cast<size_t>(c)] = timer++;
            stack.emplace_back(c, 0);
        } else {
            tout_[static_cast<size_t>(id)] = timer++;
            stack.pop_back();
        }
    }

    // 预建全部键（含无结构耦合的空对，形状固定），再按非零计数填充。
    for (int64_t i = 0; i < N_; ++i) {
        const int64_t ni = size(i);
        kDiag_.emplace(key_(i, i), SparseMatrix(ni, ni));
        mDiag_.emplace(key_(i, i), SparseMatrix(ni, ni));
        for (int64_t a : tree_.ancestors(i)) {
            kCoup_.emplace(key_(i, a), SparseMatrix(ni, size(a)));
            mCoup_.emplace(key_(i, a), SparseMatrix(ni, size(a)));
        }
    }
    nPairs_ = static_cast<int64_t>(kCoup_.size());
    extract_(reordered.K, kDiag_, kCoup_);
    extract_(reordered.M, mDiag_, mCoup_);
}

int64_t OriginalBlocks::size(int64_t i) const {
    return static_cast<int64_t>(idx_[static_cast<size_t>(i)].size());
}

const std::vector<int64_t>& OriginalBlocks::indices(int64_t i) const {
    return idx_[static_cast<size_t>(i)];
}

const OriginalBlocks::SparseMatrix& OriginalBlocks::diagK(int64_t i) const {
    return kDiag_.at(key_(i, i));
}
const OriginalBlocks::SparseMatrix& OriginalBlocks::diagM(int64_t i) const {
    return mDiag_.at(key_(i, i));
}

const OriginalBlocks::SparseMatrix& OriginalBlocks::subK(int64_t i,
                                                         int64_t a) const {
    if (!isAncestor_(a, i))
        throw std::out_of_range("OriginalBlocks::subK: a is not an ancestor of i");
    return kCoup_.at(key_(i, a));
}
const OriginalBlocks::SparseMatrix& OriginalBlocks::subM(int64_t i,
                                                         int64_t a) const {
    if (!isAncestor_(a, i))
        throw std::out_of_range("OriginalBlocks::subM: a is not an ancestor of i");
    return mCoup_.at(key_(i, a));
}

bool OriginalBlocks::hasCouplingK(int64_t i, int64_t a) const {
    auto it = kCoup_.find(key_(i, a));
    return it != kCoup_.end() && it->second.nonZeros() > 0;
}
bool OriginalBlocks::hasCouplingM(int64_t i, int64_t a) const {
    auto it = mCoup_.find(key_(i, a));
    return it != mCoup_.end() && it->second.nonZeros() > 0;
}

void OriginalBlocks::extract_(
    const SparseMatrix& A,
    std::unordered_map<int64_t, SparseMatrix>& diag,
    std::unordered_map<int64_t, SparseMatrix>& coup) {
    if (!A.isCompressed()) {
        AMLS_LOG_ERROR("OriginalBlocks::extract_ 失败：输入矩阵未压缩。");
        throw std::invalid_argument("OriginalBlocks: input matrix not compressed");
    }

    // pass 1：分类计数（diag nc==nr；耦合取后代为行节点方向）。
    std::unordered_map<int64_t, int64_t> count;
    count.reserve(diag.size() + coup.size());
    for (int64_t c = 0; c < A.cols(); ++c) {
        const int64_t nc = nodeOf_[static_cast<size_t>(c)];
        for (SparseMatrix::InnerIterator it(A, c); it; ++it) {
            const int64_t r = it.row();
            const int64_t nr = nodeOf_[static_cast<size_t>(r)];
            int64_t key;
            if (nr == nc) {
                key = key_(nc, nc);
            } else if (isAncestor_(nc, nr)) {
                key = key_(nr, nc);  // 行 ∈ nr（后代），列 ∈ nc（祖先）
            } else if (isAncestor_(nr, nc)) {
                key = key_(nc, nr);  // 对称镜像：行 ∈ nc（后代），列 ∈ nr
            } else {
                AMLS_LOG_ERROR(
                    "OriginalBlocks::extract_ 失败：不可比节点间出现耦合 "
                    "(row dof {} -> node {}, col dof {} -> node {})。",
                    r, nr, c, nc);
                throw std::invalid_argument(
                    "OriginalBlocks: coupling between incomparable nodes");
            }
            ++count[key];
        }
    }
    for (const auto& [key, cnt] : count) {
        auto dit = diag.find(key);
        if (dit != diag.end()) {
            dit->second.reserve(cnt);
            continue;
        }
        auto cit = coup.find(key);
        if (cit == coup.end()) {
            AMLS_LOG_ERROR("OriginalBlocks::extract_ 失败：计数键 {} 无预建槽位。",
                           key);
            throw std::logic_error("OriginalBlocks: counted key has no slot");
        }
        cit->second.reserve(cnt);
    }

    // pass 2：按全局列升序插入 → 块内外层（列）升序，insert 均摊 O(1)。
    for (int64_t c = 0; c < A.cols(); ++c) {
        const int64_t nc = nodeOf_[static_cast<size_t>(c)];
        for (SparseMatrix::InnerIterator it(A, c); it; ++it) {
            const int64_t r = it.row();
            const int64_t nr = nodeOf_[static_cast<size_t>(r)];
            SparseMatrix* target;
            int64_t lr, lc;
            if (nr == nc) {
                target = &diag.at(key_(nc, nc));
                lr = localOf_[static_cast<size_t>(r)];
                lc = localOf_[static_cast<size_t>(c)];
            } else if (isAncestor_(nc, nr)) {
                target = &coup.at(key_(nr, nc));
                lr = localOf_[static_cast<size_t>(r)];
                lc = localOf_[static_cast<size_t>(c)];
            } else {
                target = &coup.at(key_(nc, nr));  // 镜像项：转置位置
                lr = localOf_[static_cast<size_t>(c)];
                lc = localOf_[static_cast<size_t>(r)];
            }
            target->insert(lr, lc) = it.value();
        }
    }
    // 按坐标去重（首见保留）：全对称 pencil 的镜像项在耦合块中与直接项
    // 落入同一坐标（转置规约后恒同位），insert 会产生重复存储；单侧存储
    // 则由镜像分支补全。对称矩阵两向值恒相等，取首见对两种约定均正确。
    auto finalize = [](SparseMatrix& mat) {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(mat.nonZeros()));
        for (int64_t c = 0; c < mat.cols(); ++c)
            for (SparseMatrix::InnerIterator it(mat, c); it; ++it)
                trips.emplace_back(it.row(), c, it.value());
        std::sort(trips.begin(), trips.end(),
                  [](const Eigen::Triplet<double>& a,
                     const Eigen::Triplet<double>& b) {
                      if (a.col() != b.col()) return a.col() < b.col();
                      return a.row() < b.row();
                  });
        const auto uniqueEnd = std::unique(
            trips.begin(), trips.end(),
            [](const Eigen::Triplet<double>& a, const Eigen::Triplet<double>& b) {
                return a.row() == b.row() && a.col() == b.col();
            });
        mat.setFromTriplets(trips.begin(), uniqueEnd);
    };
    for (auto& [key, mat] : diag) finalize(mat);
    for (auto& [key, mat] : coup) finalize(mat);
}

} // namespace aams
