#include "amls/separator_tree.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace amls {

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
    // 父子对称
    for (const auto& nd : nodes_) {
        if (nd.role == NodeRole::Leaf && !nd.children.empty())
            return false;
        if (nd.role != NodeRole::Leaf && nd.children.size() != 2)
            return false;
        for (int64_t c : nd.children)
            if (nodes_.at(c).parent != nd.id)
                return false;
    }
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
    // 无环（索引总数匹配 + 父子对称已足够,树形由构造保证）
    return true;
}

} // namespace amls
