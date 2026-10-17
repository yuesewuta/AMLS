#include "ams/separator_tree.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace aams {

int64_t SeparatorTree::addNode(NodeRole role, std::vector<int64_t> indices) {
    TreeNode nd;
    nd.id = static_cast<int64_t>(nodes_.size());
    nd.role = role;
    nd.indices = std::move(indices);
    nodes_.push_back(std::move(nd));
    return nodes_.back().id;
}

void SeparatorTree::setChildren(int64_t id, int64_t left, int64_t right) {
    nodes_.at(id).children = {left, right};
}

void SeparatorTree::setChildren(int64_t id, std::vector<int64_t> children) {
    nodes_.at(id).children = std::move(children);
}

void SeparatorTree::setParent(int64_t id, int64_t parent) {
    nodes_.at(id).parent = parent;
}

void SeparatorTree::postOrderRec(int64_t id, std::vector<int64_t>& out) const {
    const TreeNode& nd = nodes_.at(id);
    for (int64_t c : nd.children)
        postOrderRec(c, out);
    out.push_back(id);
}

std::vector<int64_t> SeparatorTree::postOrder() const {
    std::vector<int64_t> out;
    out.reserve(nodes_.size());
    postOrderRec(root(), out);
    return out;
}

void SeparatorTree::descendantsRec(int64_t id,
                                   std::vector<int64_t>& out) const {
    for (int64_t c : nodes_.at(id).children) {
        out.push_back(c);
        descendantsRec(c, out);
    }
}

std::vector<int64_t> SeparatorTree::descendants(int64_t id) const {
    std::vector<int64_t> out;
    descendantsRec(id, out);
    return out;
}

std::vector<int64_t> SeparatorTree::ancestors(int64_t id) const {
    std::vector<int64_t> out;
    for (int64_t p = nodes_.at(id).parent; p != -1; p = nodes_.at(p).parent)
        out.push_back(p);
    return out;
}

int64_t SeparatorTree::root() const {
    for (const auto& nd : nodes_)
        if (nd.parent == -1)
            return nd.id;
    return -1;
}

void SeparatorTree::relabelRoot() {
    const int64_t r = root();
    if (r >= 0 && nodes_[static_cast<size_t>(r)].role == NodeRole::Separator)
        nodes_[static_cast<size_t>(r)].role = NodeRole::Root;
}

bool SeparatorTree::validate(int64_t nTotal) const {
    if (nodes_.empty())
        return false;
    // 恰一个根
    int nRoots = 0;
    for (const auto& nd : nodes_)
        if (nd.parent == -1)
            ++nRoots;
    if (nRoots != 1)
        return false;
    // 每个非根节点恰有一条入边，并且所有节点从根可达。
    std::vector<int64_t> incoming(nodes_.size(), 0);
    for (const auto& nd : nodes_) {
        if (nd.role == NodeRole::Leaf && !nd.children.empty()) return false;
        // 支持单子链：非叶节点可恰有 1 子（消去块树允许单子链）。
        // 若角色为 Leaf 则必须无子；Separator/Root 可有 >=1 子。
        for (int64_t c : nd.children) {
            if (c < 0 || c >= nNodes() || nodes_[static_cast<size_t>(c)].parent != nd.id)
                return false;
            if (++incoming[static_cast<size_t>(c)] != 1) return false;
        }
    }
    for (const auto& nd : nodes_)
        if (incoming[static_cast<size_t>(nd.id)] != (nd.parent == -1 ? 0 : 1))
            return false;
    std::vector<bool> seen(nodes_.size(), false);
    std::vector<int64_t> stack{root()};
    int64_t visited = 0;
    while (!stack.empty()) {
        const int64_t id = stack.back();
        stack.pop_back();
        if (seen[static_cast<size_t>(id)]) return false;
        seen[static_cast<size_t>(id)] = true;
        ++visited;
        for (const int64_t c : nodes_[static_cast<size_t>(id)].children)
            stack.push_back(c);
    }
    if (visited != nNodes()) return false;
    // 索引：不相交且并集 = [0, nTotal)
    std::vector<int64_t> all;
    all.reserve(static_cast<size_t>(nTotal));
    for (const auto& nd : nodes_)
        all.insert(all.end(), nd.indices.begin(), nd.indices.end());
    if (static_cast<int64_t>(all.size()) != nTotal)
        return false;
    std::sort(all.begin(), all.end());
    for (int64_t i = 0; i < nTotal; ++i)
        if (all[static_cast<size_t>(i)] != i)
            return false;
    return true;
}

} // namespace aams
