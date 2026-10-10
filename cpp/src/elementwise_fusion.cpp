#include <unordered_map>
#include <vector>

#include "../include/fused_elementwise.h"
#include "../include/graph_builder.h"
#include "../include/pass.h"

static bool fusible_op(OpCode op) {
    return op == OpCode::ADD || op == OpCode::SUB || op == OpCode::MUL ||
           op == OpCode::DIV || op == OpCode::UNARY;
}

static bool same_layout(const Node& a, const Node& b) {
    return a.shape == b.shape && a.strides == b.strides && a.offset == b.offset;
}

struct FusedInstruction {
    FusedOp op;
    int64_t lhs;
    int64_t rhs;
    int64_t attr;
};

class ExpressionBuilder {
   public:
    ExpressionBuilder(const IR& ir, const std::vector<int>& use_count, int root)
        : ir_(ir), use_count_(use_count), root_(root) {}

    int64_t build(int node_id, const Node* consumer) {
        const Node& node = ir_.nodes[node_id];
        const bool can_inline = fusible_op(node.op) &&
                                (node_id == root_ ||
                                 (consumer != nullptr && use_count_[node_id] == 1 &&
                                  same_layout(node, *consumer)));
        if (!can_inline) return leaf(node_id);

        int64_t lhs = -1;
        int64_t rhs = -1;
        if (node.inputs.size() == 2) {
            lhs = build(node.inputs[0], &node);
            rhs = build(node.inputs[1], &node);
        } else if (node.inputs.size() != 1) {
            return leaf(node_id);
        } else {
            lhs = build(node.inputs[0], &node);
        }

        FusedInstruction instruction;
        instruction.lhs = lhs;
        instruction.rhs = rhs;
        instruction.attr = (node.op == OpCode::UNARY) ? node.args[0] : 0;
        switch (node.op) {
            case OpCode::ADD:
                instruction.op = FusedOp::ADD;
                break;
            case OpCode::SUB:
                instruction.op = FusedOp::SUB;
                break;
            case OpCode::MUL:
                instruction.op = FusedOp::MUL;
                break;
            case OpCode::DIV:
                instruction.op = FusedOp::DIV;
                break;
            case OpCode::UNARY:
                instruction.op = FusedOp::UNARY;
                break;
            default:
                return leaf(node_id);
        }
        instructions_.push_back(instruction);
        return fused_temp_ref(static_cast<int64_t>(instructions_.size() - 1));
    }

    const std::vector<int>& leaves() const { return leaves_; }
    const std::vector<FusedInstruction>& instructions() const { return instructions_; }

   private:
    int64_t leaf(int node_id) {
        auto it = leaf_ids_.find(node_id);
        if (it != leaf_ids_.end()) return it->second;
        const int64_t ref = static_cast<int64_t>(leaves_.size());
        leaf_ids_.emplace(node_id, ref);
        leaves_.push_back(node_id);
        return ref;
    }

    const IR& ir_;
    const std::vector<int>& use_count_;
    int root_;
    std::unordered_map<int, int64_t> leaf_ids_;
    std::vector<int> leaves_;
    std::vector<FusedInstruction> instructions_;
};

void ElementwiseFusionPass::run(IR& ir) {
    if (ir.output_index < 0 || ir.output_index >= static_cast<int>(ir.nodes.size())) return;

    std::vector<int> use_count(ir.nodes.size(), 0);
    for (const Node& node : ir.nodes) {
        for (int input : node.inputs) ++use_count[input];
    }

    const int root = ir.output_index;
    if (!fusible_op(ir.nodes[root].op)) return;

    ExpressionBuilder expression(ir, use_count, root);
    const int64_t output_ref = expression.build(root, nullptr);
    if (expression.instructions().size() < 2) return;

    Node fused;
    fused.op = OpCode::FUSED_ELEMENTWISE;
    fused.inputs = expression.leaves();
    fused.shape = ir.nodes[root].shape;
    fused.strides = ir.nodes[root].strides;
    fused.offset = ir.nodes[root].offset;
    fused.args = {kFusedEncodingVersion,
                  static_cast<int64_t>(expression.instructions().size()), output_ref};
    for (const FusedInstruction& instruction : expression.instructions()) {
        fused.args.push_back(static_cast<int64_t>(instruction.op));
        fused.args.push_back(instruction.lhs);
        fused.args.push_back(instruction.rhs);
        fused.args.push_back(instruction.attr);
    }

    GraphBuilder builder(ir);
    const int fused_id = builder.append(std::move(fused));
    builder.replace(root, fused_id);
}
