#include "ams/tree_builder.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <metis.h>

#include "partition/supernode_builder.hpp"
#include "partition/symbolic_analyzer.hpp"
#include "ams/aggregation.hpp"

namespace aams {
namespace rea {

namespace {

// 与 metis_partitioner.cpp extractGraphFast 一致的无向 CSR 提取（idx_t 类型，
// 必须匹配 METIS_NodeND 的 idx_t* 参数；否则指针步进错乱导致段错误）。
struct Csr {
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
};
Csr extractGraph(const DefaultTraits::SparseMatrix& A) {
    const int64_t n = A.rows();
    DefaultTraits::SparseMatrix compressed = A;
    compressed.makeCompressed();
    const auto& C = compressed;
    // 行度计数（off-diag (r,c) 贡献 c 到 r 的邻接）
    std::vector<idx_t> rdeg(static_cast<size_t>(n) + 1, 0);
    for (int64_t c = 0; c < n; ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(C, c); it; ++it)
            if (it.row() != c)
                ++rdeg[static_cast<size_t>(it.row()) + 1];
    for (int64_t i = 0; i < n; ++i)
        rdeg[static_cast<size_t>(i) + 1] += rdeg[static_cast<size_t>(i)];
    std::vector<idx_t> radj(static_cast<size_t>(rdeg.back()));
    std::vector<idx_t> cur(rdeg.begin(), rdeg.end() - 1);
    for (int64_t c = 0; c < n; ++c)
        for (DefaultTraits::SparseMatrix::InnerIterator it(C, c); it; ++it)
            if (it.row() != c)
                radj[static_cast<size_t>(cur[static_cast<size_t>(it.row())]++)] =
                    static_cast<idx_t>(c);
    // adj(v) = 列 v 行号 ∪ radj[v]，两侧各自升序 → 归并去重
    Csr g;
    g.xadj.assign(static_cast<size_t>(n) + 1, 0);
    for (int64_t v = 0; v < n; ++v) {
        const int64_t cb = C.outerIndexPtr()[v], ce = C.outerIndexPtr()[v + 1];
        const idx_t rb = rdeg[static_cast<size_t>(v)],
                      re = rdeg[static_cast<size_t>(v) + 1];
        int64_t cnt = 0;
        int64_t p = cb, q = rb;
        while (p < ce || q < re) {
            idx_t val;
            const idx_t colVal = p < ce ? C.innerIndexPtr()[p] : idx_t(-1);
            const idx_t rowVal = q < re ? radj[static_cast<size_t>(q)] : idx_t(-1);
            if (q >= re || (p < ce && colVal < rowVal)) { val = colVal; ++p; }
            else if (p >= ce || rowVal < colVal)        { val = rowVal; ++q; }
            else                                        { val = colVal; ++p; ++q; }
            if (val == static_cast<idx_t>(v)) continue;   // 去自环
            g.adjncy.push_back(val);
            ++cnt;
        }
        g.xadj[static_cast<size_t>(v) + 1] =
            g.xadj[static_cast<size_t>(v)] + static_cast<idx_t>(cnt);
    }
    return g;
}

// 原子类图：classOf 把 dof 映射到原子类（网格节点/超节点原子），边双向收集
// 后逐顶点去重。classOf 不要求连续布局（c/dp 只是 classOf 的特例），因此
// 置换后的矩阵同样可提取。
Csr extractClassGraph(const DefaultTraits::SparseMatrix& A,
                      const std::vector<int32_t>& classOf, int64_t nv) {
    DefaultTraits::SparseMatrix compressed = A;
    compressed.makeCompressed();
    std::vector<std::vector<idx_t>> adj(static_cast<size_t>(nv));
    for (int64_t c = 0; c < compressed.rows(); ++c) {
        const int64_t nc = classOf[static_cast<size_t>(c)];
        for (DefaultTraits::SparseMatrix::InnerIterator it(compressed, c); it;
             ++it) {
            const int64_t r = it.row();
            if (r == c) continue;
            const int64_t nr = classOf[static_cast<size_t>(r)];
            if (nr == nc) continue;
            adj[static_cast<size_t>(nr)].push_back(static_cast<idx_t>(nc));
            adj[static_cast<size_t>(nc)].push_back(static_cast<idx_t>(nr));
        }
    }
    Csr g;
    g.xadj.assign(static_cast<size_t>(nv) + 1, 0);
    for (int64_t v = 0; v < nv; ++v) {
        auto& a = adj[static_cast<size_t>(v)];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        g.adjncy.insert(g.adjncy.end(), a.begin(), a.end());
        g.xadj[static_cast<size_t>(v) + 1] =
            static_cast<idx_t>(g.adjncy.size());
    }
    return g;
}

// 连续 dp 分组布局检测：n % dp == 0 且组内耦合边覆盖率 ≥99%。
// 不要求组内两两全耦合：Abaqus 导出的 mtx 会丢弃数值极小的跨向耦合
// （cant_20 中 1495/8820 节点缺 (0,1)/(0,2) 对、7 组全缺），缺失的耦合
// 由符号分析阶段的消元填充补回，不影响 3-DOF 超节点形成。
bool detectGroupedLayout(const DefaultTraits::SparseMatrix& A, int64_t dp) {
    const int64_t n = A.rows();
    if (n % dp != 0) return false;
    std::vector<uint8_t> hasEdge(static_cast<size_t>(n / dp), 0);
    for (int64_t c = 0; c < n; ++c) {
        const int64_t t = c / dp;
        if (hasEdge[static_cast<size_t>(t)]) continue;
        for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
            if (it.row() / dp == t && it.row() != c) {
                hasEdge[static_cast<size_t>(t)] = 1;
                break;
            }
    }
    int64_t nEmpty = 0;
    for (uint8_t e : hasEdge)
        if (!e) ++nEmpty;
    // 容忍 ≤1% 的孤立组（导出丢值），仍有组内边的组 ≥99% 即认定分组布局
    return nEmpty * 100 <= static_cast<int64_t>(hasEdge.size());
}

// ---- twin-clique 代数原子检测（置换不变）----
//
// 原子类 = 外部邻域一致的顶点类：对类 C 内任意 u,v，N(u)\C == N(v)\C。
// 这是图论 true/false twin 的推广——不要求类内两两耦合（Abaqus 导出会丢弃
// 数值极小的组内耦合，cant_20 中 1495/8820 组缺 (0,1)/(0,2) 对、7 组全缺），
// 只要求类的对外连接模式一致；缺失的组内耦合由符号分析阶段的消元填充补回。
// 判据完全由图结构决定，不依赖"连续 dp 个 DOF 一组"的布局先验，对 PAPᵀ
// 置换不变。
//
// 算法（只依赖无向图，零几何/布局假设）：
//  1. 近孪生边：|N(u)ΔN(v)| ≤ 2(cmax-1) 且 |N(u)∩N(v)| ≥ min(deg)-(cmax-1)
//     的顶点对并入 union-find。完美类内部任意两点的对称差只有 {u,v}，
//     组内缺边只会让对称差增大 2·缺边数，容差仍覆盖。
//  2. 连通分量即候选类（size>cmax 的判为误并，拆散为单例）。
//  3. 分量校验：逐类验证外部邻域严格一致，失败分量拆散为单例（保守）。
//  4. 均匀性：size≥2 类的宽度一致记 c、覆盖 ≥99% 才认定原子宽度 c；
//     少数单例作为 size-1 类保留（METIS 顶点加权），否则退回 dof 级。
struct TwinAtomInfo {
    bool ok = false;
    bool uniform = false;              // 全部类等宽且无单例
    int64_t width = 1;                 // 等宽类宽（uniform 时 ≥2，否则无意义）
    std::vector<int32_t> classOf;      // dof → 类 id（按类最小成员排序）
    std::vector<int32_t> classSize;    // 类 → 成员数（含 size-1 单例类）
    int64_t nClasses = 0;
    int64_t nSingleton = 0;            // size-1 类个数
};

TwinAtomInfo detectTwinAtoms(const Csr& g, int64_t n, bool dump) {
    TwinAtomInfo info;
    if (n < 2) return info;
    constexpr int64_t cmax = 6;        // 覆盖实体 3 / 板壳 6 自由度节点
    // 近孪生对称差容差。cant_31 全边实测标尺（delta_probe）：同三元组边
    // |N(u)ΔN(v)| ≤ 16，跨三元组边 ≥ 14——重叠区 14–16：真实长尾仅 ~11 对，
    // 跨节点假边 ~200 条。两个阈值分离：
    //   合并 13：并查集阶段排除全部跨节点假边（损失 ≤~10 个真实对，由
    //            链式传递/挂靠回收）；
    //   验证 16：组件已确保纯真（假边进不来），外部一致性校验放宽到真实
    //            上界，回收 c 与类对称差 14–16 的第三个成员。
    const int64_t limMerge = 13;
    const int64_t limVerify = 16;
    const int64_t lim = limMerge;   // 阶段 1 合并容差（下文挂靠/聚类用 limVerify）

    // 邻接升序防御整理（归并式成对测试要求有序）
    Csr gg;
    const Csr* gp = &g;
    bool sorted = true;
    for (int64_t v = 0; v < n && sorted; ++v)
        for (idx_t p = g.xadj[v] + 1; p < g.xadj[v + 1]; ++p)
            if (g.adjncy[p - 1] >= g.adjncy[p]) { sorted = false; break; }
    if (!sorted) {
        gg.xadj.assign(static_cast<size_t>(n) + 1, 0);
        gg.adjncy.reserve(g.adjncy.size());
        for (int64_t v = 0; v < n; ++v) {
            const idx_t b = g.xadj[v], e = g.xadj[v + 1];
            std::vector<idx_t> seg(g.adjncy.begin() + b, g.adjncy.begin() + e);
            std::sort(seg.begin(), seg.end());
            seg.erase(std::unique(seg.begin(), seg.end()), seg.end());
            gg.adjncy.insert(gg.adjncy.end(), seg.begin(), seg.end());
            gg.xadj[static_cast<size_t>(v) + 1] =
                static_cast<idx_t>(gg.adjncy.size());
        }
        gp = &gg;
    }
    const Csr& G = *gp;

    // ---- 公共工具：有序邻接比较 / 外部邻域提取 / 类校验 ----
    auto sameAdj = [&](int32_t u, int32_t v) {
        idx_t a = G.xadj[static_cast<size_t>(u)];
        const idx_t ae = G.xadj[static_cast<size_t>(u) + 1];
        idx_t b = G.xadj[static_cast<size_t>(v)];
        const idx_t be = G.xadj[static_cast<size_t>(v) + 1];
        while (a < ae && b < be) {
            if (G.adjncy[a] == G.adjncy[b]) { ++a; ++b; }
            else return false;
        }
        return a == ae && b == be;
    };
    std::vector<int32_t> stamp(static_cast<size_t>(n), -1);
    int32_t round = 0;
    std::vector<int32_t> ext0, ext1;
    // 外部邻域：N(u) 去掉类成员（邻接有序 ⇒ 输出有序）
    auto extOf = [&](int32_t u, const std::vector<int32_t>& members,
                     std::vector<int32_t>& out) {
        ++round;
        for (const int32_t m : members)
            stamp[static_cast<size_t>(m)] = round;
        out.clear();
        for (idx_t p = G.xadj[static_cast<size_t>(u)];
             p < G.xadj[static_cast<size_t>(u) + 1]; ++p)
            if (stamp[static_cast<size_t>(G.adjncy[p])] != round)
                out.push_back(G.adjncy[p]);
    };
    // 有序集合对称差 ≤ tol（早停）
    auto diffLE = [&](const std::vector<int32_t>& a,
                      const std::vector<int32_t>& b, int64_t tol) {
        size_t i = 0, j = 0;
        int64_t d = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j]) { ++i; ++j; }
            else if (a[i] < b[j]) { ++d; ++i; }
            else                  { ++d; ++j; }
            if (d > tol) return false;
        }
        d += static_cast<int64_t>(a.size() - i) +
             static_cast<int64_t>(b.size() - j);
        return d <= tol;
    };
    // 类校验：成员外部邻域对首成员的对称差 ≤ lim
    auto verifyClass = [&](const std::vector<int32_t>& members) {
        const int64_t sz = static_cast<int64_t>(members.size());
        if (sz < 2 || sz > cmax) return false;
        extOf(members[0], members, ext0);
        for (int64_t t = 1; t < sz; ++t) {
            extOf(members[static_cast<size_t>(t)], members, ext1);
            if (!diffLE(ext0, ext1, limVerify)) return false;
        }
        return true;
    };

    std::vector<int32_t> assigned(static_cast<size_t>(n), 0);
    std::vector<int32_t> classIdx(static_cast<size_t>(n), -1);  // dof → 类序号
    std::vector<std::vector<int32_t>> classes;   // 校验通过的 size≥2 类
    // 类外部邻域指纹桶：游离单例反查候选类（见阶段 2）
    std::vector<uint64_t> zob(static_cast<size_t>(n));
    {
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (auto& z : zob) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            z = s;
        }
    }
    std::unordered_map<uint64_t, std::vector<int32_t>> extBucket;
    std::vector<uint64_t> classEh;
    std::vector<int> classOrigin;   // 诊断：0=精确桶 1=边分量 2=挂靠增长 3=单例聚类
    auto classRehash = [&](int32_t cid) {
        const auto& mem = classes[static_cast<size_t>(cid)];
        extOf(mem[0], mem, ext0);
        uint64_t h = 0;
        for (const int32_t x : ext0) h ^= zob[static_cast<size_t>(x)];
        classEh[static_cast<size_t>(cid)] = h;
        extBucket[h].push_back(cid);
    };

    // ---- 阶段 0：精确孪生桶（Zobrist XOR 邻域指纹）----
    // 捕获完全同邻域的非邻接 dof：组内耦合被导出全丢的组（cant_20 有 7 组）
    // 成员互不相连，边测试永远够不到。N(u)==N(v) ⟺ 指纹桶相同
    // （64 位指纹只作候选，逐对精确复核）。
    {
        std::vector<uint64_t> fh(static_cast<size_t>(n), 0);
        for (int64_t u = 0; u < n; ++u)
            for (idx_t p = G.xadj[static_cast<size_t>(u)];
                 p < G.xadj[static_cast<size_t>(u) + 1]; ++p)
                fh[static_cast<size_t>(u)] ^= zob[static_cast<size_t>(G.adjncy[p])];
        std::unordered_map<uint64_t, std::vector<int32_t>> buckets;
        for (int64_t u = 0; u < n; ++u)
            buckets[fh[static_cast<size_t>(u)]].push_back(
                static_cast<int32_t>(u));
        for (auto& kv : buckets) {
            auto& bm = kv.second;
            if (bm.size() < 2 || bm.size() > static_cast<size_t>(cmax)) continue;
            bool same = true;
            for (size_t t = 1; t < bm.size() && same; ++t)
                same = sameAdj(bm[0], bm[t]);
            if (!same) continue;
            classes.push_back(std::move(bm));
            classEh.push_back(0);
            classOrigin.push_back(0);
            const int32_t cid = static_cast<int32_t>(classes.size() - 1);
            for (const int32_t u : classes.back()) {
                assigned[static_cast<size_t>(u)] = 1;
                classIdx[static_cast<size_t>(u)] = cid;
            }
            classRehash(cid);
        }
    }

    // ---- 阶段 1：近孪生边 + 并查集 + 分量校验 ----

    // 并查集
    std::vector<int32_t> parent(static_cast<size_t>(n));
    for (int64_t v = 0; v < n; ++v)
        parent[static_cast<size_t>(v)] = static_cast<int32_t>(v);
    auto find = [&](int32_t x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] =
                parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    auto unite = [&](int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[static_cast<size_t>(std::max(a, b))] = std::min(a, b);
    };

    // 近孪生边测试：对称差早停 + 交集下限（已入类的 dof 不再参与）
    for (int64_t u = 0; u < n; ++u) {
        if (assigned[static_cast<size_t>(u)]) continue;
        const idx_t ub = G.xadj[static_cast<size_t>(u)];
        const idx_t ue = G.xadj[static_cast<size_t>(u) + 1];
        const int64_t du = ue - ub;
        for (idx_t p = ub; p < ue; ++p) {
            const int64_t v = G.adjncy[p];
            if (v <= u) continue;   // 每条无向边只测一次
            if (assigned[static_cast<size_t>(v)]) continue;
            const idx_t vb = G.xadj[static_cast<size_t>(v)];
            const idx_t ve = G.xadj[static_cast<size_t>(v) + 1];
            const int64_t dv = ve - vb;
            if (du > dv + 2 * cmax || dv > du + 2 * cmax) continue;
            int64_t delta = 0, inter = 0;
            idx_t a = ub, b = vb;
            while (a < ue && b < ve && delta <= lim) {
                const idx_t x = G.adjncy[a], y = G.adjncy[b];
                if (x == y) { ++inter; ++a; ++b; }
                else if (x < y) { ++delta; ++a; }
                else            { ++delta; ++b; }
            }
            if (delta > lim) continue;
            delta += (ue - a) + (ve - b);
            if (delta > lim) continue;
            if (inter < std::min(du, dv) - (cmax - 1)) continue;
            unite(static_cast<int32_t>(u), static_cast<int32_t>(v));
        }
    }

    // 分量收集（CSR 两遍，只收未入类的 dof）
    std::vector<int32_t> compOf(static_cast<size_t>(n), -1);
    int64_t nc = 0;
    for (int64_t v = 0; v < n; ++v) {
        if (assigned[static_cast<size_t>(v)]) continue;
        const int32_t r = find(static_cast<int32_t>(v));
        if (compOf[static_cast<size_t>(r)] < 0)
            compOf[static_cast<size_t>(r)] = static_cast<int32_t>(nc++);
        compOf[static_cast<size_t>(v)] = compOf[static_cast<size_t>(r)];
    }
    std::vector<int64_t> cStart(static_cast<size_t>(nc) + 1, 0);
    for (int64_t v = 0; v < n; ++v) {
        if (assigned[static_cast<size_t>(v)]) continue;
        ++cStart[static_cast<size_t>(compOf[static_cast<size_t>(v)]) + 1];
    }
    for (int64_t i = 0; i < nc; ++i)
        cStart[static_cast<size_t>(i) + 1] += cStart[static_cast<size_t>(i)];
    std::vector<int32_t> cList(
        static_cast<size_t>(n -
                            static_cast<int64_t>(std::count(
                                assigned.begin(), assigned.end(), 1))),
        0);
    {
        std::vector<int64_t> cur(cStart.begin(), cStart.end() - 1);
        for (int64_t v = 0; v < n; ++v) {
            if (assigned[static_cast<size_t>(v)]) continue;
            cList[static_cast<size_t>(
                cur[static_cast<size_t>(compOf[static_cast<size_t>(v)])]++)] =
                static_cast<int32_t>(v);
        }
    }

    // 分量校验：外部邻域对称差 ≤ lim；失败分量拆散为游离 dof（阶段 2 再试挂靠）
    int64_t dissolved = 0, nBig = 0, nExtFail = 0;
    std::map<int64_t, int64_t> compHist;         // 分量规模直方图（诊断）
    for (int64_t ci = 0; ci < nc; ++ci) {
        const int64_t b = cStart[static_cast<size_t>(ci)];
        const int64_t e = cStart[static_cast<size_t>(ci) + 1];
        const int64_t sz = e - b;
        ++compHist[sz <= 12 ? sz : 13];
        if (sz == 1) continue;
        std::vector<int32_t> cand(cList.begin() + b, cList.begin() + e);
        if (verifyClass(cand)) {
            classes.push_back(std::move(cand));
            classEh.push_back(0);
            classOrigin.push_back(1);
            const int32_t cid = static_cast<int32_t>(classes.size() - 1);
            for (const int32_t u : classes.back()) {
                assigned[static_cast<size_t>(u)] = 1;
                classIdx[static_cast<size_t>(u)] = cid;
            }
            classRehash(cid);
        } else {
            dissolved += sz;
            if (sz > cmax) ++nBig; else ++nExtFail;
        }
    }

    // ---- 阶段 2：游离 dof 挂靠 ----
    // 阶段 0/1 后仍未入类的 dof（典型：组内两条边都被导出丢弃的真类单例，
    // 与类成员互不相连、外部指纹也因非对称丢边而不同）。候选类按来源合并
    // 去重，再按外部对称差 |E_u Δ E_C0| 升序逐一尝试校验挂靠：
    //  a) 距离 1：邻居已入类；
    //  b) 距离 2：邻居 e 的邻居已入类——真类成员与 u 共享整个外部邻域，
    //     只要任一条 a–e 边幸存即被提名（不要求 u 与类成员相邻）；
    //  c) 外部指纹桶：N(u) 的 Zobrist 指纹恰好命中某类的类外指纹。
    int64_t dbgOk = 0, dbgFail = 0, dbgNoCand = 0, dbgDist2Only = 0;
    std::vector<int32_t> extU;
    for (int64_t u = 0; u < n; ++u) {
        if (assigned[static_cast<size_t>(u)]) continue;
        ++round;
        std::vector<int32_t> candClasses;
        std::vector<int> candFlags;   // bit0=距离1 bit1=指纹桶 bit2=距离2
        auto pushCand = [&](int32_t cid, int flag) {
            if (cid < 0) return;
            for (size_t k = 0; k < candClasses.size(); ++k)
                if (candClasses[k] == cid) { candFlags[k] |= flag; return; }
            candClasses.push_back(cid);
            candFlags.push_back(flag);
        };
        // 指纹先算（extOf 会推进 round，不能夹在 pushCand 中间）
        extOf(static_cast<int32_t>(u), {}, extU);
        uint64_t hu = 0;
        for (const int32_t x : extU) hu ^= zob[static_cast<size_t>(x)];
        // a) 距离 1 + b) 距离 2
        for (idx_t p = G.xadj[static_cast<size_t>(u)];
             p < G.xadj[static_cast<size_t>(u) + 1]; ++p) {
            const int32_t e = G.adjncy[p];
            if (classIdx[static_cast<size_t>(e)] >= 0)
                pushCand(classIdx[static_cast<size_t>(e)], 1);
            for (idx_t q = G.xadj[static_cast<size_t>(e)];
                 q < G.xadj[static_cast<size_t>(e) + 1]; ++q) {
                const int32_t x = G.adjncy[q];
                if (x != u && classIdx[static_cast<size_t>(x)] >= 0)
                    pushCand(classIdx[static_cast<size_t>(x)], 4);
            }
        }
        // c) 外部指纹桶
        auto it = extBucket.find(hu);
        if (it != extBucket.end())
            for (const int32_t cid : it->second) pushCand(cid, 2);
        if (candClasses.empty()) { ++dbgNoCand; continue; }

        // 排序：与 u 的外部对称差越小越可能是真类
        std::vector<std::pair<int64_t, int32_t>> ranked;
        ranked.reserve(candClasses.size());
        for (size_t k = 0; k < candClasses.size(); ++k) {
            const int32_t cid = candClasses[k];
            if (static_cast<int64_t>(classes[static_cast<size_t>(cid)].size()) +
                    1 > cmax)
                continue;
            extOf(classes[static_cast<size_t>(cid)][0],
                  classes[static_cast<size_t>(cid)], ext1);
            int64_t d = 0;
            size_t i = 0, j = 0;
            while (i < extU.size() && j < ext1.size()) {
                if (extU[i] == ext1[j]) { ++i; ++j; }
                else if (extU[i] < ext1[j]) { ++d; ++i; }
                else { ++d; ++j; }
            }
            d += static_cast<int64_t>(extU.size() - i) +
                 static_cast<int64_t>(ext1.size() - j);
            ranked.emplace_back(d, cid);
        }
        std::sort(ranked.begin(), ranked.end());

        bool ok = false;
        bool onlyDist2 = true;
        for (const auto& rc : ranked) {
            const int32_t cid = rc.second;
            std::vector<int32_t> trial(classes[static_cast<size_t>(cid)]);
            // 保持成员升序（extOf/binary_search 依赖）
            trial.insert(std::lower_bound(trial.begin(), trial.end(),
                                          static_cast<int32_t>(u)),
                         static_cast<int32_t>(u));
            if (!verifyClass(trial)) continue;
            classes[static_cast<size_t>(cid)] = std::move(trial);
            assigned[static_cast<size_t>(u)] = 1;
            classIdx[static_cast<size_t>(u)] = cid;
            classRehash(cid);
            ok = true;
            for (size_t k = 0; k < candClasses.size(); ++k)
                if (candClasses[k] == cid) { onlyDist2 = candFlags[k] == 4; break; }
            break;
        }
        if (ok) {
            ++dbgOk;
            if (onlyDist2) ++dbgDist2Only;
        } else {
            ++dbgFail;
        }
    }
    if (dump)
        std::fprintf(stderr,
                     "[rea-tree] twin-clique 挂靠: 成功=%lld 其中仅距离2=%lld "
                     "失败=%lld 无候选=%lld\n",
                     (long long)dbgOk, (long long)dbgDist2Only,
                     (long long)dbgFail, (long long)dbgNoCand);

    // ---- 阶段 3：游离单例聚类 ----
    // 组内耦合全缺且外部指纹互有差异的三元组（cant_20 式全缺组）成员均为
    // 单例，无类可挂靠。剩余单例数量极少（~0.4%），直接两两比对邻域对称
    // 差 ≤ lim 者并集，再逐分量 verifyClass。
    {
        std::vector<int32_t> singles;
        for (int64_t v = 0; v < n; ++v)
            if (!assigned[static_cast<size_t>(v)])
                singles.push_back(static_cast<int32_t>(v));
        if (singles.size() >= 2) {
            // 并查集（复用 parent，先重置）
            for (const int32_t v : singles)
                parent[static_cast<size_t>(v)] = v;
            for (size_t i = 0; i < singles.size(); ++i)
                for (size_t j = i + 1; j < singles.size(); ++j) {
                    // 邻域对称差 ≤ lim → 同类候选（Zobrist 预筛不适用：
                    // 差异源自丢值，指纹必不同；单例量少直接比对）
                    int64_t d = 0;
                    idx_t a = G.xadj[static_cast<size_t>(singles[i])];
                    const idx_t ae =
                        G.xadj[static_cast<size_t>(singles[i]) + 1];
                    idx_t b = G.xadj[static_cast<size_t>(singles[j])];
                    const idx_t be =
                        G.xadj[static_cast<size_t>(singles[j]) + 1];
                    while (a < ae && b < be && d <= lim) {
                        if (G.adjncy[a] == G.adjncy[b]) { ++a; ++b; }
                        else if (G.adjncy[a] < G.adjncy[b]) { ++d; ++a; }
                        else { ++d; ++b; }
                    }
                    if (d > limVerify) continue;
                    d += (ae - a) + (be - b);
                    if (d <= limVerify)
                        unite(singles[i], singles[j]);
                }
            // 分量 → verifyClass → 类
            std::map<int32_t, std::vector<int32_t>> groups;
            for (const int32_t v : singles)
                groups[find(v)].push_back(v);
            for (auto& kv : groups) {
                auto& mem = kv.second;
                if (mem.size() < 2 ||
                    static_cast<int64_t>(mem.size()) > cmax) continue;
                std::sort(mem.begin(), mem.end());
                if (verifyClass(mem)) {
                    classes.push_back(std::move(mem));
                    classEh.push_back(0);
                    classOrigin.push_back(3);
                    const int32_t cid =
                        static_cast<int32_t>(classes.size() - 1);
                    for (const int32_t u : classes.back()) {
                        assigned[static_cast<size_t>(u)] = 1;
                        classIdx[static_cast<size_t>(u)] = cid;
                    }
                    classRehash(cid);
                }
            }
        }
    }

    // ---- 均匀性 + 覆盖率 ----
    int64_t covered = 0;
    int64_t width = 0;
    {
        std::map<int64_t, int64_t> hist;
        for (const auto& cls : classes) {
            ++hist[static_cast<int64_t>(cls.size())];
            covered += static_cast<int64_t>(cls.size());
        }
        int64_t singles = 0;
        for (int64_t v = 0; v < n; ++v)
            if (!assigned[static_cast<size_t>(v)]) ++singles;
        // 单例 ≤1% 才认定原子结构（与 detectGroupedLayout 的容忍度一致）
        const bool covOk = singles * 100 <= n;
        if (dump) {
            std::string hs;
            for (const auto& kv : hist)
                hs += " " + std::to_string(kv.first) + ":" +
                      std::to_string(kv.second);
            std::string cs;
            for (const auto& kv : compHist)
                cs += " " + (kv.first <= 12 ? std::to_string(kv.first)
                                            : std::string(">12")) +
                      ":" + std::to_string(kv.second);
            std::fprintf(stderr,
                         "[rea-tree] twin-clique: 类宽%s 单例=%lld "
                         "类覆盖=%.3f%% 拆散=%lld（过大=%lld 外部不一致=%lld）"
                         "分量规模%s\n",
                         hs.c_str(), (long long)singles,
                         100.0 * static_cast<double>(covered) /
                             static_cast<double>(n),
                         (long long)dissolved, (long long)nBig,
                         (long long)nExtFail, cs.c_str());
        }
        if (!hist.empty() && covOk) {
            info.ok = true;
            info.uniform = hist.size() == 1 && singles == 0;
            if (info.uniform) width = hist.begin()->first;
        }
    }
    if (!info.ok) return info;

    // 类按最小成员排序赋 id（连续布局下还原为 d/c 的自然顺序）
    std::sort(classes.begin(), classes.end(),
              [](const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
                  return x.front() < y.front();
              });
    info.classOf.assign(static_cast<size_t>(n), -1);
    info.classSize.clear();
    int32_t nextId = 0;
    int64_t singles = 0;
    for (const auto& cls : classes) {
        for (const int32_t u : cls)
            info.classOf[static_cast<size_t>(u)] = nextId;
        info.classSize.push_back(static_cast<int32_t>(cls.size()));
        ++nextId;
    }
    for (int64_t v = 0; v < n; ++v) {
        if (info.classOf[static_cast<size_t>(v)] < 0) {
            info.classOf[static_cast<size_t>(v)] = nextId++;
            info.classSize.push_back(1);
            ++singles;
        }
    }
    if (dump && n % 3 == 0) {
        // 诊断（借用布局口径）：连续三元组被拆散的形态统计
        std::map<std::string, int64_t> pat;
        for (int64_t t = 0; t < n; t += 3) {
            const int32_t c0 = info.classOf[static_cast<size_t>(t)];
            const int32_t c1 = info.classOf[static_cast<size_t>(t + 1)];
            const int32_t c2 = info.classOf[static_cast<size_t>(t + 2)];
            if (c0 == c1 && c0 == c2) continue;
            const int s0 = info.classSize[static_cast<size_t>(c0)];
            const int s1 = info.classSize[static_cast<size_t>(c1)];
            const int s2 = info.classSize[static_cast<size_t>(c2)];
            pat["(" + std::to_string(s0) + "," + std::to_string(s1) + "," +
                std::to_string(s2) + ")"]++;
        }
        std::string ps;
        for (const auto& kv : pat)
            ps += " " + kv.first + ":" + std::to_string(kv.second);
        std::fprintf(stderr, "[rea-tree] twin-clique 拆散三元组形态:%s\n",
                     ps.c_str());
        // size-2 类的成员来源：同三元组对 / 跨三元组对（诊断用布局口径）
        int64_t sameP = 0, crossP = 0;
        for (const auto& cls : classes) {
            if (cls.size() != 2) continue;
            ++(((cls[0] / 3) == (cls[1] / 3)) ? sameP : crossP);
        }
        std::fprintf(stderr,
                     "[rea-tree] twin-clique size2类: 同三元组=%lld "
                     "跨三元组=%lld\n", (long long)sameP, (long long)crossP);
        std::map<int, std::map<int, int64_t>> stageHist;
        for (size_t k = 0; k < classes.size(); ++k)
            ++stageHist[classOrigin[k]]
                       [static_cast<int>(classes[k].size())];
        for (const auto& kv : stageHist) {
            std::string h2;
            for (const auto& kv2 : kv.second)
                h2 += " " + std::to_string(kv2.first) + ":" +
                      std::to_string(kv2.second);
            std::fprintf(stderr, "[rea-tree] twin-clique 阶段%d类:%s\n",
                         kv.first, h2.c_str());
        }
    }
    info.width = width;
    info.nClasses = nextId;
    info.nSingleton = singles;
    return info;
}


}  // namespace

IPartitioner<DefaultTraits>::Result buildAbaqusTree(
    const DefaultTraits::SparseMatrix& A, const AbaqusTreeOptions& opt) {
    using Result = IPartitioner<DefaultTraits>::Result;
    const int64_t n = A.rows();
    Result res;
    if (n == 0) return res;
    const bool prof = std::getenv("AAMS_PROFILE") != nullptr;
    auto clockNow = [] { return std::chrono::steady_clock::now(); };
    auto profMark = [&](const char* what, std::chrono::steady_clock::time_point t0) {
        if (prof)
            std::fprintf(stderr, "[prof] tree %s=%.2fs\n", what,
                         std::chrono::duration<double>(clockNow() - t0).count());
    };
    auto ts = clockNow();

    // ---- 1. 原子检测 + 网格节点图收缩 + METIS_NodeND 排列 ----
    // Abaqus 在网格节点图上跑 NodeND（stf 块以 3-DOF 为主），DOF 排列 =
    // 节点排列按类展开成 perm[new]=old。dp<=0 时优先 twin-clique 代数原子
    // 检测（置换不变，判据 = 外部邻域一致）；检测未认定原子时退回旧的
    // 连续分组布局先验（dp=3），再不行 dof 级。
    const bool dump = std::getenv("AMLS_DUMP_TREE") != nullptr;
    int dp = opt.dofsPerNode;
    std::vector<int32_t> classOf;    // dof → 原子类 id
    std::vector<int32_t> classSize;  // 类 → 成员数
    int64_t nv = 0;
    bool uniformNodes = false;       // 全类等宽（METIS 免顶点权重）
    auto identityClasses = [&](int64_t w) {
        classOf.resize(static_cast<size_t>(n));
        for (int64_t d = 0; d < n; ++d)
            classOf[static_cast<size_t>(d)] = static_cast<int32_t>(d / w);
        classSize.assign(static_cast<size_t>(n / w),
                         static_cast<int32_t>(w));
        nv = n / w;
        uniformNodes = true;
    };
    Csr gd;   // dof 级图（dp==1 时直接作 NodeND 输入）
    if (dp > 1) {
        // 显式指定宽度：沿用连续分组契约（n 整除 + 组内耦合覆盖率 ≥99%）
        if (!(n % dp == 0 && detectGroupedLayout(A, dp))) {
            throw std::runtime_error(
                "rea::buildAbaqusTree: dofsPerNode=" + std::to_string(dp) +
                " 布局校验失败（n 不整除或组内 dofs 未两两耦合）");
        }
        identityClasses(dp);
    } else if (opt.dofsPerNode == 1) {
        identityClasses(1);   // 显式 dof 级，不做原子检测
    } else {
        gd = extractGraph(A);
        profMark("extract-dof", ts);
        TwinAtomInfo atoms;
        if (std::getenv("AAMS_NO_TWIN") == nullptr)
            atoms = detectTwinAtoms(gd, n, dump);
        if (atoms.ok) {
            // dp 仅作阈值换算与日志；混合类宽时统一走加权路径
            dp = atoms.uniform ? static_cast<int>(atoms.width) : 1;
            if (!atoms.uniform) {
                // 还原等宽类宽供换算（若全部 size≥2 类等宽），否则用均值
                int64_t wsum = 0;
                for (const int32_t s : atoms.classSize) wsum += s;
                dp = static_cast<int>(std::max<int64_t>(
                    1, wsum / std::max<int64_t>(atoms.nClasses, 1)));
            }
            classOf = std::move(atoms.classOf);
            classSize = std::move(atoms.classSize);
            nv = atoms.nClasses;
            uniformNodes = atoms.uniform;
            if (dump)
                std::fprintf(stderr,
                             "[rea-tree] twin-clique 原子检测: %s 类数=%lld "
                             "单例=%lld\n",
                             atoms.uniform
                                 ? ("等宽 dp=" + std::to_string(atoms.width))
                                       .c_str()
                                 : "混合类宽",
                             (long long)atoms.nClasses,
                             (long long)atoms.nSingleton);
        } else {
            dp = (n % 3 == 0 && detectGroupedLayout(A, 3)) ? 3 : 1;
            identityClasses(dp);
            if (dump)
                std::fprintf(stderr,
                             "[rea-tree] twin-clique 未认定原子，退回分组布局 "
                             "dp=%d\n", dp);
        }
    }
    if (dump)
        std::fprintf(stderr, "[rea-tree] 节点收缩宽度 dp=%d (nv=%lld)\n",
                     dp, (long long)nv);
    const int64_t nodeUnit =
        uniformNodes ? dp : std::max<int64_t>(1, n / std::max<int64_t>(nv, 1));
    profMark("detect", ts);
    Csr g = dp > 1 ? extractClassGraph(A, classOf, nv) : std::move(gd);
    auto te = clockNow();
    profMark("extract", te);
    idx_t nvtxs = static_cast<idx_t>(nv);
    std::vector<idx_t> mperm(static_cast<size_t>(nv)), iperm(static_cast<size_t>(nv));
    std::vector<idx_t> vwgt;
    if (!uniformNodes) {
        vwgt.resize(static_cast<size_t>(nv));
        for (int64_t v = 0; v < nv; ++v)
            vwgt[static_cast<size_t>(v)] =
                static_cast<idx_t>(classSize[static_cast<size_t>(v)]);
    }
    idx_t opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = opt.metisSeed;
    const int r = METIS_NodeND(&nvtxs, g.xadj.data(), g.adjncy.data(),
                               vwgt.empty() ? nullptr : vwgt.data(), opts,
                               mperm.data(), iperm.data());
    if (r != METIS_OK)
        throw std::runtime_error("rea::buildAbaqusTree: METIS_NodeND 失败");
    profMark("metis", clockNow());
    // DOF 排列：节点新序按类展开（类成员升序连续放置）。nodeOff[p] = 新序
    // 前 p 个节点的 dof 数（step3 块展开复用）；等宽连续时退化为 dp*p。
    std::vector<int64_t> mStart(static_cast<size_t>(nv) + 1, 0),
        mList(static_cast<size_t>(n));
    for (int64_t d = 0; d < n; ++d)
        ++mStart[static_cast<size_t>(classOf[static_cast<size_t>(d)]) + 1];
    for (int64_t v = 0; v < nv; ++v)
        mStart[static_cast<size_t>(v) + 1] += mStart[static_cast<size_t>(v)];
    {
        std::vector<int64_t> cur(mStart.begin(), mStart.end() - 1);
        for (int64_t d = 0; d < n; ++d)   // dof 升序扫描 ⇒ 类成员升序
            mList[static_cast<size_t>(
                cur[static_cast<size_t>(classOf[static_cast<size_t>(d)])]++)] = d;
    }
    std::vector<int64_t> nodeOff(static_cast<size_t>(nv) + 1, 0);
    for (int64_t pos = 0; pos < nv; ++pos)
        nodeOff[static_cast<size_t>(pos) + 1] =
            nodeOff[static_cast<size_t>(pos)] +
            classSize[static_cast<size_t>(mperm[static_cast<size_t>(pos)])];
    std::vector<int64_t> permND(static_cast<size_t>(n));
    for (int64_t pos = 0; pos < nv; ++pos) {
        const int64_t v = mperm[static_cast<size_t>(pos)];
        const int64_t base = nodeOff[static_cast<size_t>(pos)];
        const int64_t mb = mStart[static_cast<size_t>(v)];
        const int64_t msz = classSize[static_cast<size_t>(v)];
        for (int64_t d = 0; d < msz; ++d)
            permND[static_cast<size_t>(base + d)] =
                mList[static_cast<size_t>(mb + d)];
    }

    // ---- 2. 符号分析（BuildTreeAfterMetis ↣ etree + columnCount）----
    // 在原子类层做符号分析：A_n = Rᵀ A R（类内 dof 块求和；R 为类→dof 聚合，
    // A SPD ⇒ A_n SPD）。exact 超节点以原子类为基本单元，块大小均为类宽的
    // 倍数——与 Abaqus stf 分布（cant_20: 3:6739/6:1033/9:70/12:48/18:5/24:3，
    // 全为 3 的倍数）一致。若在 dof 层做消除分析，导出 mtx 丢失的组内耦合
    // 会让三重奏被 exact 检测拆开。
    sym::SymbolicAnalysis sa;
    auto t2 = clockNow();
    auto t2a = t2;
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(A.nonZeros()));
        for (int64_t c = 0; c < n; ++c)
            for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it)
                trips.emplace_back(classOf[static_cast<size_t>(it.row())],
                                   classOf[static_cast<size_t>(c)],
                                   it.value());
        profMark("step2-trips", t2a);
        t2a = clockNow();
        DefaultTraits::SparseMatrix An(nv, nv);
        An.setFromTriplets(trips.begin(), trips.end());
        profMark("step2-sft", t2a);
        t2a = clockNow();
        std::vector<int64_t> nodePerm(static_cast<size_t>(nv));
        for (int64_t v = 0; v < nv; ++v)
            nodePerm[static_cast<size_t>(v)] = mperm[static_cast<size_t>(v)];
        if (dump)
            std::fprintf(stderr,
                         "[rea-tree] step2 analyze (node level, nv=%lld)\n",
                         (long long)nv);
        sa = sym::analyzeSymmetrized(An, nodePerm);
    }
    if (dump)
        std::fprintf(stderr, "[rea-tree] step2 done etree size=%lld\n",
                     (long long)sa.parentColumn.size());
    profMark("step2-analyze", t2);
    auto t3 = clockNow();

    // ---- 3. exact 基础超节点（CreateFundamentalSuperNodes ↣ buildBlockForest）----
    if (dump)
        std::fprintf(stderr, "[rea-tree] step3 base blocks\n");
    // 块上限按节点数换算（maxBlockDofs 为 dof 单位；非等宽时按平均类宽）
    sym::BlockForest base = sym::buildBlockForest(
        sa, std::max<int64_t>(1, opt.maxBlockDofs / nodeUnit));
    // 节点块 [b,e) → dof 块 [nodeOff[b], nodeOff[e))（节点排列展开后连续）；
    // blockOfColumn 同步重建为 dof 层（聚合 Pass 以它取 nTotal）。
    for (auto& blk : base.blocks) {
        blk.begin = nodeOff[static_cast<size_t>(blk.begin)];
        blk.end = nodeOff[static_cast<size_t>(blk.end)];
    }
    base.blockOfColumn.assign(static_cast<size_t>(n), -1);
    for (int64_t b = 0; b < static_cast<int64_t>(base.blocks.size()); ++b)
        for (int64_t j = base.blocks[static_cast<size_t>(b)].begin;
             j < base.blocks[static_cast<size_t>(b)].end; ++j)
            base.blockOfColumn[static_cast<size_t>(j)] = b;
    if (std::getenv("AMLS_DUMP_TREE")) {
        std::fprintf(stderr, "[rea-tree] step3 done base blocks=%lld\n",
                     (long long)base.blocks.size());
        sym::dumpAggregationStats(base, "base");
    }

    // ---- 4. Abaqus 聚合（Collapse2 + AggregateLeaves + Cutoff）----
    sym::AggregationParams ap;
    ap.collapseMaxDofs = opt.collapseMaxDofs;
    ap.leafAggrMaxDofs = opt.leafAggrMaxDofs;
    ap.maxBlockDofs   = opt.maxBlockDofs;
    ap.minBlockDofs   = opt.minBlockDofs;
    ap.collapseMode   = opt.collapseMode;
    ap.aggregateLeaves = opt.aggregateLeaves;
    if (std::getenv("AMLS_DUMP_TREE"))
        std::fprintf(stderr, "[rea-tree] step4 aggregate\n");
    sym::BlockForest forest = sym::aggregateAbaqusStyle(base, ap);
    profMark("step3+4-blocks", t3);
    auto t5 = clockNow();
    if (std::getenv("AMLS_DUMP_TREE"))
        sym::dumpAggregationStats(forest, "aggregated");

    // ---- 5. 块森林 → SeparatorTree（原始坐标）----
    // 多根处理：SeparatorTree 只支持单根（postOrder/validate 依赖）。
    // Abaqus 支持森林（numberOfTrees），但后续 solver 需要单根；
    // 这里把多个根合并为一个超根。超根是纯结构合并点，indices 必须为空——
    // 各根的 DOF 仍归各根节点所有，若超根再收一遍会造成 DOF 重复归属。
    // opt.superRootDofs>0 仅是显式授权开关（0 = 拒绝合成，直接报错）。
    int64_t nb = static_cast<int64_t>(forest.blocks.size());
    bool needSuperRoot = static_cast<int64_t>(forest.roots.size()) > 1;
    if (needSuperRoot && opt.superRootDofs <= 0)
        throw std::runtime_error(
            "rea::buildAbaqusTree: 多根森林需要 superRootDofs>0（合成单根）");
    std::vector<int64_t> nodeOfBlock(static_cast<size_t>(nb), -1);
    // 先建普通块节点
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        std::vector<int64_t> idx;
        idx.reserve(static_cast<size_t>(blk.end - blk.begin));
        for (int64_t j = blk.begin; j < blk.end; ++j)
            idx.push_back(permND[static_cast<size_t>(j)]);
        const NodeRole role =
            blk.children.empty() ? NodeRole::Leaf : NodeRole::Separator;
        nodeOfBlock[static_cast<size_t>(b)] =
            res.tree.addNode(role, std::move(idx));
    }
    // 父/子关系
    for (int64_t b = 0; b < nb; ++b) {
        const auto& blk = forest.blocks[static_cast<size_t>(b)];
        if (blk.children.empty()) continue;
        std::vector<int64_t> kids;
        kids.reserve(blk.children.size());
        for (const int64_t c : blk.children)
            kids.push_back(nodeOfBlock[static_cast<size_t>(c)]);
        for (const int64_t c : kids)
            res.tree.setParent(c, nodeOfBlock[static_cast<size_t>(b)]);
        res.tree.setChildren(nodeOfBlock[static_cast<size_t>(b)], std::move(kids));
    }
    // 多根合成超根（空 indices 的结构合并点）
    if (needSuperRoot) {
        std::vector<int64_t> rootKids;
        rootKids.reserve(forest.roots.size());
        for (const int64_t r : forest.roots)
            rootKids.push_back(nodeOfBlock[static_cast<size_t>(r)]);
        const int64_t superId =
            res.tree.addNode(NodeRole::Root, std::vector<int64_t>{});
        for (const int64_t c : rootKids)
            res.tree.setParent(c, superId);
        res.tree.setChildren(superId, std::move(rootKids));
    }
    res.tree.relabelRoot();

    // ---- 6. 后序排列 perm[new]=old（重排坐标 → 原始坐标）----
    // nd.indices 已是原始坐标（step5 存的就是 permND[列]），perm 直接是
    // 后序节点 indices 的顺接；再套一次 permND 会双重映射。
    const std::vector<int64_t> order = res.tree.postOrder();
    res.perm.clear();
    res.perm.reserve(static_cast<size_t>(n));
    for (const int64_t id : order) {
        const TreeNode& nd = res.tree.node(static_cast<size_t>(id));
        for (const int64_t j : nd.indices)
            res.perm.push_back(j);
    }

    // ---- 7. 原始 K 全边祖先可比自检 ----
    // A 的边 (r,c) 与节点 indices 同为原始坐标，直接用原始 dof 索引 dofBlock。
    if (std::getenv("AMLS_DUMP_TREE") != nullptr) {
        auto isAncestor = [&](int64_t a, int64_t b) -> bool {
            for (int64_t x = b; x != -1;
                 x = res.tree.node(static_cast<size_t>(x)).parent)
                if (x == a) return true;
            return false;
        };
        std::vector<int64_t> dofBlock(static_cast<size_t>(n), -1);
        for (int64_t id = 0; id < res.tree.nNodes(); ++id)
            for (const int64_t d : res.tree.node(static_cast<size_t>(id)).indices)
                dofBlock[static_cast<size_t>(d)] = id;
        int64_t nBad = 0;
        for (int64_t c = 0; c < A.outerSize(); ++c)
            for (DefaultTraits::SparseMatrix::InnerIterator it(A, c); it; ++it) {
                const int64_t rr = it.row();
                if (rr == c) continue;
                const int64_t br = dofBlock[static_cast<size_t>(rr)];
                const int64_t bc = dofBlock[static_cast<size_t>(c)];
                if (br == bc) continue;
                if (!isAncestor(br, bc) && !isAncestor(bc, br)) ++nBad;
            }
        std::fprintf(stderr, "[rea-tree] K全边不可比计数=%lld\n",
                     (long long)nBad);
    }
    profMark("step5-7-tree", t5);
    return res;
}

}  // namespace rea
}  // namespace aams