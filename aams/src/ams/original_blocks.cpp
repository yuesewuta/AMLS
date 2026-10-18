#include "ams/original_blocks.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <utility>

#include "core/logging.hpp"
#include "linalg/gemm_cols.hpp"

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

    // 抽取在 extract_ 内部完成（计数 → 按需建槽 → 填充）；K 与 M 的
    // map 互不相交，两线程并行。
    {
        colcomp_detail::ThreadGroup tg(1);
        auto work = [&] {
            blas::pinThisThread();
            extract_(reordered.M, mDiag_, mCoup_, "M");
        };
        tg.start(work);
        extract_(reordered.K, kDiag_, kCoup_, "K");
        tg.wait();
    }
    nPairs_ = static_cast<int64_t>(kCoup_.size());
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

const OriginalBlocks::SparseMatrix* OriginalBlocks::findSubK(
    int64_t i, int64_t a) const {
    if (!isAncestor_(a, i))
        throw std::out_of_range(
            "OriginalBlocks::findSubK: a is not an ancestor of i");
    const auto it = kCoup_.find(key_(i, a));
    return it == kCoup_.end() ? nullptr : &it->second;
}
const OriginalBlocks::SparseMatrix* OriginalBlocks::findSubM(
    int64_t i, int64_t a) const {
    if (!isAncestor_(a, i))
        throw std::out_of_range(
            "OriginalBlocks::findSubM: a is not an ancestor of i");
    const auto it = mCoup_.find(key_(i, a));
    return it == mCoup_.end() ? nullptr : &it->second;
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
    std::unordered_map<int64_t, SparseMatrix>& coup,
    const char* tag) {
    if (!A.isCompressed()) {
        AMLS_LOG_ERROR("OriginalBlocks::extract_ 失败：输入矩阵未压缩。");
        throw std::invalid_argument("OriginalBlocks: input matrix not compressed");
    }
    static const bool phLog = std::getenv("AAMS_RED_CLOSE_LOG") != nullptr;
    const auto ph0 = std::chrono::steady_clock::now();

    // 列 → 节点归属：块 key_(i,·) 只从节点 i 的列接收非零（直连与镜像
    // 皆然），故按节点划分列区间即可无锁并行 pass1/pass2。
    std::vector<std::vector<int64_t>> nodeCols(
        static_cast<size_t>(N_));
    for (int64_t c = 0; c < A.cols(); ++c)
        nodeCols[static_cast<size_t>(nodeOf_[static_cast<size_t>(c)])]
            .push_back(c);

    // pass 1：分类计数（diag nc==nr；耦合取后代为行节点方向）。
    // 同列内行按全局升序且各节点 DOF 为连续区间 → 行节点成段出现，
    // 单槽缓存（nr → key）把哈希查找从每非零一次降到每段一次。
    // 各线程独立计数，最后合并（块键集固定，无竞争）。
    const int nthr = [&] {
        if (const char* e = std::getenv("AAMS_OB_NTHR"))
            return std::max(1, std::atoi(e));
        return static_cast<int>(std::min<size_t>(
            8, std::max<size_t>(1, N_ / 64)));
    }();
    std::vector<std::unordered_map<int64_t, int64_t>> counts(
        static_cast<size_t>(std::max(1, nthr)));
    {
        colcomp_detail::ThreadGroup tg(
            static_cast<size_t>(std::max(0, nthr - 1)));
        auto countRange = [&](int t) {
            blas::pinThisThread();
            auto& count = counts[static_cast<size_t>(t)];
            count.reserve(diag.size() + coup.size());
            for (int64_t i = t; i < N_; i += nthr) {
                for (int64_t c : nodeCols[static_cast<size_t>(i)]) {
                    const int64_t nc = i;
                    int64_t lastNr = -1, lastKey = -1;
                    for (SparseMatrix::InnerIterator it(A, c); it; ++it) {
                        const int64_t r = it.row();
                        const int64_t nr =
                            nodeOf_[static_cast<size_t>(r)];
                        if (nr != lastNr) {
                            lastNr = nr;
                            if (nr == nc) {
                                lastKey = key_(nc, nc);
                            } else if (isAncestor_(nc, nr)) {
                                lastKey = key_(nr, nc);
                            } else if (isAncestor_(nr, nc)) {
                                lastKey = key_(nc, nr);
                            } else {
                                AMLS_LOG_ERROR(
                                    "OriginalBlocks::extract_ 失败：不可比节点"
                                    "间出现耦合 (row dof {} -> node {}, col dof "
                                    "{} -> node {})。",
                                    r, nr, c, nc);
                                throw std::invalid_argument(
                                    "OriginalBlocks: coupling between "
                                    "incomparable nodes");
                            }
                        }
                        ++count[lastKey];
                    }
                }
            }
        };
        for (int t = 0; t < nthr; ++t) {
            if (t == nthr - 1) countRange(t);
            else tg.start([=] { countRange(t); });
        }
        tg.wait();
    }
    // 合并到独立 total（保持 counts[t] 为纯线程本地计数，供 pass2 预留）。
    std::unordered_map<int64_t, int64_t> total;
    for (size_t t = 0; t < counts.size(); ++t)
        for (const auto& [key, cnt] : counts[t]) total[key] += cnt;
    // 按需建槽：对角块恒建；耦合块仅在计数非零时建（无结构耦合的祖先对
    // 不建槽，findSub* 返回 nullptr → 消去模块对缺失块走零初始化路径）。
    // total 恰为收到非零的键集（每个非零都计数），建槽后必能承接 pass2。
    for (int64_t i = 0; i < N_; ++i)
        diag.emplace(key_(i, i), SparseMatrix(size(i), size(i)));
    for (const auto& [key, cnt] : total) {
        const int64_t d = key / N_, a = key - d * N_;
        if (d == a) continue;
        coup.emplace(key, SparseMatrix(size(d), size(a)));
    }
    for (const auto& [key, cnt] : total) {
        auto dit = diag.find(key);
        if (dit != diag.end()) {
            dit->second.reserve(cnt);
            continue;
        }
        auto cit = coup.find(key);
        if (cit != coup.end()) cit->second.reserve(cnt);
    }
    const auto ph1 = std::chrono::steady_clock::now();

    // pass 2（分线程暂存）：耦合块 key_(d,a) 同时收到节点 a 列（直连，
    // 行∈d）与节点 d 列（镜像）的非零，d/a 可能分属不同线程 → 不能并行
    // 直写共享块。各线程写独立暂存 map（touching 记录 (key,线程)），
    // finalize 阶段按块合并。每节点列升序 → 暂存块内列升序。
    std::vector<std::unordered_map<int64_t, SparseMatrix>> tDiag(
        static_cast<size_t>(nthr)),
        tCoup(static_cast<size_t>(nthr));
    std::vector<std::pair<int64_t, int>> touched;
    {
        colcomp_detail::ThreadGroup tg(
            static_cast<size_t>(std::max(0, nthr - 1)));
        auto insertRange = [&](int t) {
            blas::pinThisThread();
            auto& diagT = tDiag[static_cast<size_t>(t)];
            auto& coupT = tCoup[static_cast<size_t>(t)];
            // 暂存块按需建槽：默认构造是 0x0，insert 会越界 → 建槽时给足尺寸。
            auto stageDiag = [&](int64_t key, int64_t node) -> SparseMatrix& {
                auto it = diagT.find(key);
                if (it == diagT.end())
                    it = diagT
                             .emplace(key,
                                      SparseMatrix(size(node), size(node)))
                             .first;
                return it->second;
            };
            auto stageCoup = [&](int64_t key, int64_t rd,
                                 int64_t rc) -> SparseMatrix& {
                auto it = coupT.find(key);
                if (it == coupT.end())
                    it = coupT
                             .emplace(key, SparseMatrix(size(rd), size(rc)))
                             .first;
                return it->second;
            };
            // 线程 t 的计数与暂存同分区：按计数预建槽位并 reserve，
            // 恢复旧实现的预留插入路径。
            for (const auto& [key, cnt] :
                 counts[static_cast<size_t>(t)]) {
                const int64_t d = key / N_, a = key - d * N_;
                SparseMatrix& s =
                    d == a ? stageDiag(key, d) : stageCoup(key, d, a);
                s.reserve(cnt);
            }
            for (int64_t i = t; i < N_; i += nthr) {
                for (int64_t c : nodeCols[static_cast<size_t>(i)]) {
                    SparseMatrix* lastTgt = nullptr;
                    int64_t lastNr = -1;
                    int64_t lastKey = -1;
                    bool lastMirror = false;
                    for (SparseMatrix::InnerIterator it(A, c); it; ++it) {
                        const int64_t r = it.row();
                        const int64_t nr =
                            nodeOf_[static_cast<size_t>(r)];
                        if (nr != lastNr) {
                            lastNr = nr;
                            if (nr == i) {
                                lastKey = key_(i, i);
                                lastTgt = &stageDiag(lastKey, i);
                                lastMirror = false;
                            } else if (isAncestor_(i, nr)) {
                                lastKey = key_(nr, i);
                                lastTgt = &stageCoup(lastKey, nr, i);
                                lastMirror = false;
                            } else if (isAncestor_(nr, i)) {
                                lastKey = key_(i, nr);  // 镜像项
                                lastTgt = &stageCoup(lastKey, i, nr);
                                lastMirror = true;
                            } else {
                                AMLS_LOG_ERROR(
                                    "OriginalBlocks::extract_ 失败：不可比节点"
                                    "间出现耦合 (row dof {} -> node {}, col dof "
                                    "{} -> node {})。",
                                    r, nr, c, i);
                                throw std::invalid_argument(
                                    "OriginalBlocks: coupling between "
                                    "incomparable nodes");
                            }
                        }
                        if (lastMirror)
                            lastTgt->insert(
                                localOf_[static_cast<size_t>(c)],
                                localOf_[static_cast<size_t>(r)]) = it.value();
                        else
                            lastTgt->insert(
                                localOf_[static_cast<size_t>(r)],
                                localOf_[static_cast<size_t>(c)]) = it.value();
                    }
                }
            }
        };
        for (int t = 0; t < nthr; ++t) {
            if (t == nthr - 1) insertRange(t);
            else tg.start([=] { insertRange(t); });
        }
        tg.wait();
    }
    for (int t = 0; t < nthr; ++t) {
        for (const auto& [key, mat] : tDiag[static_cast<size_t>(t)])
            touched.emplace_back(key, t);
        for (const auto& [key, mat] : tCoup[static_cast<size_t>(t)])
            touched.emplace_back(key, t);
    }
    std::sort(touched.begin(), touched.end(),
              [](const std::pair<int64_t, int>& a,
                 const std::pair<int64_t, int>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });
    // 合并 + 去重：耦合块可能来自两个线程（直连 + 镜像），diag 块单一
    // 来源。touched 按 key 排序，equal_range 取该块全部来源（线程序固定
    // → 求值确定）。同坐标跨来源重复（对称两向值恒等），取首见。
    // 各块互不相交，按块并行。
    auto finalize = [&](int64_t key, SparseMatrix& mat) {
        std::vector<Eigen::Triplet<double>> trips;
        const int64_t d = key / N_, a = key - d * N_;
        const bool isDiag = d == a;
        const auto lb = std::lower_bound(
            touched.begin(), touched.end(), key,
            [](const std::pair<int64_t, int>& p, int64_t k) {
                return p.first < k;
            });
        const auto ub = std::upper_bound(
            touched.begin(), touched.end(), key,
            [](int64_t k, const std::pair<int64_t, int>& p) {
                return k < p.first;
            });
        for (auto itp = lb; itp != ub; ++itp) {
            const auto& staged =
                isDiag ? tDiag[static_cast<size_t>(itp->second)]
                       : tCoup[static_cast<size_t>(itp->second)];
            const auto found = staged.find(key);
            if (found == staged.end()) continue;
            const SparseMatrix& src = found->second;
            for (int64_t c = 0; c < src.cols(); ++c)
                for (SparseMatrix::InnerIterator it(src, c); it; ++it)
                    trips.emplace_back(it.row(), c, it.value());
        }
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
    std::vector<std::pair<int64_t, SparseMatrix*>> mats;
    mats.reserve(diag.size() + coup.size());
    for (auto& [key, mat] : diag) mats.emplace_back(key, &mat);
    for (auto& [key, mat] : coup) mats.emplace_back(key, &mat);
    const auto ph2 = std::chrono::steady_clock::now();
    {
        const int nthr = static_cast<int>(std::min<size_t>(
            8, std::max<size_t>(1, mats.size() / 64)));
        if (nthr <= 1) {
            for (auto& [key, m] : mats) finalize(key, *m);
        } else {
            std::atomic<size_t> next{0};
            colcomp_detail::ThreadGroup tg(static_cast<size_t>(nthr - 1));
            for (int t = 0; t < nthr; ++t) {
                auto work = [&] {
                    blas::pinThisThread();
                    for (;;) {
                        const size_t i = next.fetch_add(1);
                        if (i >= mats.size()) break;
                        finalize(mats[i].first, *mats[i].second);
                    }
                };
                if (t == nthr - 1) work();
                else tg.start(work);
            }
            tg.wait();
        }
    }
    if (phLog) {
        const auto ph3 = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "[obext] %s pass1=%.1fms pass2=%.1fms fin(%zu)=%.1fms\n",
                     tag,
                     std::chrono::duration<double, std::milli>(ph1 - ph0).count(),
                     std::chrono::duration<double, std::milli>(ph2 - ph1).count(),
                     mats.size(),
                     std::chrono::duration<double, std::milli>(ph3 - ph2).count());
    }
}

} // namespace aams
