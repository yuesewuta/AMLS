#pragma once
// 分割节点树（设计文档 §4.4）——系统的唯一组织概念。
// 每个节点恰为三种角色之一：叶节点 / 分割节点 / 根节点。
#include <cstdint>
#include <vector>

namespace amls {

enum class NodeRole { Leaf, Separator, Root };

struct TreeNode {
    int64_t id = -1;
    NodeRole role = NodeRole::Leaf;
    std::vector<int64_t> indices;   // 本节点持有的原始行列索引
    std::vector<int64_t> children;  // 分割节点恰有 2 子；叶节点为空
    int64_t parent = -1;            // 根节点为 -1
};

class SeparatorTree {
public:
    SeparatorTree() = default;

    int64_t addNode(NodeRole role, std::vector<int64_t> indices);
    void setChildren(int64_t id, int64_t left, int64_t right);
    void setParent(int64_t id, int64_t parent);

    int64_t nNodes() const { return static_cast<int64_t>(nodes_.size()); }
    const TreeNode& node(int64_t id) const { return nodes_.at(id); }

    /// 消除顺序 = 后序（叶→根）,设计文档 §2。
    std::vector<int64_t> postOrder() const;
    /// 祖先路径（父→根方向,不含自身）。
    std::vector<int64_t> ancestors(int64_t id) const;
    /// 后代（子树内,不含自身）。
    std::vector<int64_t> descendants(int64_t id) const;
    /// 根节点 id。
    int64_t root() const;

    /// 将根节点角色修正为 Root（三角色标注完整化）。
    void relabelRoot();

    /// 一致性校验：索引不相交、并集完备、父子关系对称、三角色合法。
    bool validate(int64_t nTotal) const;

private:
    std::vector<TreeNode> nodes_;
    void postOrderRec(int64_t id, std::vector<int64_t>& out) const;
    void descendantsRec(int64_t id, std::vector<int64_t>& out) const;
};

} // namespace amls
