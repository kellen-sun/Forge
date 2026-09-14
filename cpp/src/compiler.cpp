#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../include/array_handle.h"
#include "../include/compiler.h"
#include "../include/ir_utils.h"
#include "../include/pass.h"

void optimize_graph(IR& ir) {
    PassManager pm;
    pm.addPass(std::make_unique<ShapeInferPass>());
    pm.addPass(std::make_unique<CanonicalizePass>());
    pm.addPass(std::make_unique<DCEPass>());
    pm.run(ir);
}
// Could generate Fused Kernels, with special OpCodes
// Description of which fusedkernel for the OpCode given in the OpCodes "Arg" parameter
// Read about MLIR & TVM as options here instead of doing it here
// Optimization ideas:
// 1. dead code elimination
// 2. fold constants, (3 + 4) known at compile time. or
// 3. common sub expression elimination
// ---> those all fall under LVN analysis
// 4. reroll a loop, say they did: for i in len(A): A[i] + B[i] -> just A+ B yk
// 5. Fusion, combine nodes into "blocks" that run in the same "way" (elementwise easiest)
// 6. loop fusion. like two for i in range(100) can be put together

static KernelConfig ghost_config() {
    KernelConfig c;
    c.grid = {1, 1, 1};
    c.group = {1, 1, 1};
    return c;
}

static KernelConfig dispatch_config(const std::string& name, uint64_t numel) {
    KernelConfig c;
    c.name = name;
    uint64_t n = std::max<uint64_t>(numel, 1);
    uint64_t tg = std::min<uint64_t>(256, n);
    c.grid = {n, 1, 1};
    c.group = {tg, 1, 1};
    return c;
}

static const char* bin_symbol(OpCode op) {
    switch (op) {
        case OpCode::ADD:
            return "+";
        case OpCode::SUB:
            return "-";
        case OpCode::MUL:
            return "*";
        case OpCode::DIV:
            return "/";
        default:
            return nullptr;
    }
}

static const char* bin_name(OpCode op) {
    switch (op) {
        case OpCode::ADD:
            return "add";
        case OpCode::SUB:
            return "sub";
        case OpCode::MUL:
            return "mul";
        case OpCode::DIV:
            return "div";
        default:
            return nullptr;
    }
}

static void emit_linear_index(std::ostringstream& src, const std::string& idx_name,
                              const std::string& linear, const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides) {
    src << "long " << idx_name << "=0;";
    if (shape.empty()) return;
    src << "{uint remaining=" << linear << ";";
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        src << "{uint c=remaining%" << (uint64_t)shape[i] << "u;" << idx_name << "+=long(c)*"
            << strides[i] << "L;remaining/=" << (uint64_t)shape[i] << "u;}";
    }
    src << "}";
}

static std::string f32_literal(float v) {
    std::ostringstream o;
    o.setf(std::ios::scientific);
    o.precision(9);
    o << v << 'f';
    return o.str();
}

void generateKernels(Graph& graph) {
    graph.configs.clear();
    graph.shader_source.clear();
    graph.configs.reserve(graph.nodes.size());

    std::ostringstream body;
    bool any_kernel = false;
    bool random_helpers_emitted = false;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const Node& node = graph.nodes[i];
        uint64_t numel = numel_from_shape(node.shape);

        switch (node.op) {
            case OpCode::INPUT:
            case OpCode::VIEW:
            case OpCode::RESHAPE:
            case OpCode::TRANSPOSE:
                graph.configs.push_back(ghost_config());
                break;

            case OpCode::CONSTANT: {
                if (node.args.empty()) {
                    throw std::runtime_error("generateKernels: CONSTANT missing value");
                }
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                std::string name = "op_" + std::to_string(i) + "_const";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],uint gid [[thread_position_in_grid]]){"
                     << "Out[gid]=" << f32_literal(decode_f32(node.args[0])) << ";}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::ZEROS: {
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                std::string name = "op_" + std::to_string(i) + "_zeros";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],uint gid [[thread_position_in_grid]]){"
                     << "Out[gid]=0.0f;}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::RAND:
            case OpCode::RANDN: {
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                if (!random_helpers_emitted) {
                    body << "uint forge_hash(uint seed){"
                            "seed=(seed^61u)^(seed>>16);seed*=9u;seed=seed^(seed>>4);"
                            "seed*=0x27d4eb2du;seed=seed^(seed>>15);return seed;}"
                            "float forge_rand_uniform(uint gid,uint base_seed){"
                            "return float(forge_hash(base_seed+gid))/4294967295.0f;}"
                            "float forge_rand_normal(uint gid,uint base_seed){"
                            "float u1=max(forge_rand_uniform(gid*2u,base_seed),1e-7f);"
                            "float u2=forge_rand_uniform(gid*2u+1u,base_seed);"
                            "float r=sqrt(-2.0f*log(u1));"
                            "float theta=2.0f*3.14159265359f*u2;return r*cos(theta);}";
                    random_helpers_emitted = true;
                }
                const bool normal = node.op == OpCode::RANDN;
                const std::string name =
                    "op_" + std::to_string(i) + (normal ? "_randn" : "_rand");
                const char* function = normal ? "forge_rand_normal" : "forge_rand_uniform";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],constant uint& seed [[buffer(1)]],"
                     << "uint gid [[thread_position_in_grid]]){"
                     << "Out[gid]=" << function << "(gid,seed);}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::ADD:
            case OpCode::SUB:
            case OpCode::MUL:
            case OpCode::DIV: {
                if (node.inputs.size() != 2) {
                    throw std::runtime_error("generateKernels: binary op expects 2 inputs");
                }
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                const Node& a = graph.nodes[node.inputs[0]];
                const Node& b = graph.nodes[node.inputs[1]];
                auto strides_a = get_bcast_strides(a.shape, a.strides, node.shape);
                auto strides_b = get_bcast_strides(b.shape, b.strides, node.shape);
                std::string name = "op_" + std::to_string(i) + "_" + bin_name(node.op);
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],const device float* A [[buffer(1)]],"
                     << "const device float* B [[buffer(2)]],uint gid [[thread_position_in_grid]]){";
                emit_linear_index(body, "idx_a", "gid", node.shape, strides_a);
                emit_linear_index(body, "idx_b", "gid", node.shape, strides_b);
                body << "Out[gid]=A[idx_a]" << bin_symbol(node.op) << "B[idx_b];}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::UNARY: {
                if (node.inputs.size() != 1 || node.args.empty()) {
                    throw std::runtime_error("generateKernels: UNARY expects 1 input and kind");
                }
                const int64_t kind = node.args[0];
                if (kind < 0 || kind >= kUnaryCount) {
                    throw std::runtime_error("generateKernels: unknown UNARY kind");
                }
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                const Node& src = graph.nodes[node.inputs[0]];
                auto strides_s = get_bcast_strides(src.shape, src.strides, node.shape);
                const char* uname = kUnaryNames[kind];
                std::string name = "op_" + std::to_string(i) + "_" + uname;
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],const device float* A [[buffer(1)]],"
                     << "uint gid [[thread_position_in_grid]]){";
                emit_linear_index(body, "idx_a", "gid", node.shape, strides_s);
                body << "Out[gid]=" << uname << "(A[idx_a]);}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::COPY: {
                if (node.inputs.size() != 1) {
                    throw std::runtime_error("generateKernels: COPY expects 1 input");
                }
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                const Node& src = graph.nodes[node.inputs[0]];
                auto strides_s = get_bcast_strides(src.shape, src.strides, node.shape);
                std::string name = "op_" + std::to_string(i) + "_copy";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],const device float* A [[buffer(1)]],"
                     << "uint gid [[thread_position_in_grid]]){";
                emit_linear_index(body, "idx_a", "gid", node.shape, strides_s);
                body << "Out[gid]=A[idx_a];}\n";
                graph.configs.push_back(dispatch_config(name, numel));
                any_kernel = true;
                break;
            }

            case OpCode::UPDATE: {
                if (node.inputs.size() != 2 || node.args.size() < 1 ||
                    (node.args.size() - 1) % 2 != 0) {
                    throw std::runtime_error("generateKernels: UPDATE has invalid target metadata");
                }
                if (storage_root(graph.nodes, node.inputs[0]) ==
                    storage_root(graph.nodes, node.inputs[1])) {
                    throw std::runtime_error(
                        "generateKernels: UPDATE rejects potentially overlapping RHS and destination");
                }
                const size_t rank = (node.args.size() - 1) / 2;
                std::vector<int64_t> target_shape(node.args.begin(), node.args.begin() + rank);
                std::vector<int64_t> target_strides(node.args.begin() + rank,
                                                    node.args.begin() + 2 * rank);
                const int64_t target_offset = node.args.back();
                const Node& rhs = graph.nodes[node.inputs[1]];
                const auto rhs_strides =
                    get_bcast_strides(rhs.shape, rhs.strides, target_shape);
                const uint64_t target_numel = numel_from_shape(target_shape);
                if (target_numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                const int64_t target_delta = target_offset - node.offset;
                const std::string name = "op_" + std::to_string(i) + "_update";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],const device float* A [[buffer(1)]],"
                     << "const device float* B [[buffer(2)]],uint gid "
                        "[[thread_position_in_grid]]){";
                emit_linear_index(body, "idx_target", "gid", target_shape, target_strides);
                emit_linear_index(body, "idx_rhs", "gid", target_shape, rhs_strides);
                body << "Out[idx_target+" << target_delta << "L]=B[idx_rhs];}\n";
                graph.configs.push_back(dispatch_config(name, target_numel));
                any_kernel = true;
                break;
            }

            case OpCode::SUM: {
                if (node.inputs.size() != 1 || node.args.empty()) {
                    throw std::runtime_error("generateKernels: SUM expects 1 input and args");
                }
                if (numel == 0) {
                    graph.configs.push_back(ghost_config());
                    break;
                }
                const Node& src = graph.nodes[node.inputs[0]];
                std::string name = "op_" + std::to_string(i) + "_sum";
                body << "kernel void " << name
                     << "(device float* Out [[buffer(0)]],const device float* A [[buffer(1)]],";
                if (node.args.size() == 1) {
                    const uint64_t in_numel = std::max<uint64_t>(numel_from_shape(src.shape), 1);
                    const uint64_t tg = std::min<uint64_t>(256, in_numel);
                    body << "uint tid [[thread_index_in_threadgroup]],"
                         << "uint simd_lane [[thread_index_in_simdgroup]],"
                         << "uint simd_id [[simdgroup_index_in_threadgroup]],"
                         << "uint tg_size [[threads_per_threadgroup]]){"
                         << "threadgroup float scratch[32];float t=0;"
                         << "for(uint i=tid;i<" << in_numel << "u;i+=tg_size){";
                    emit_linear_index(body, "idx", "i", src.shape, src.strides);
                    body << "t+=A[idx];}t=simd_sum(t);if(simd_lane==0)scratch[simd_id]=t;"
                         << "threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){"
                         << "float s=0;uint ns=(tg_size+31u)/32u;for(uint k=0;k<ns;++k)s+=scratch[k];"
                         << "Out[0]=s;}}\n";
                    graph.configs.push_back(dispatch_config(name, tg));
                } else {
                    body << "uint gid [[thread_position_in_grid]]){";
                    int64_t axis = node.args[0];
                    const int64_t rank = static_cast<int64_t>(src.shape.size());
                    if (axis < 0) axis += rank;
                    std::vector<int64_t> squeezed;
                    squeezed.reserve(src.shape.size());
                    for (int64_t d = 0; d < rank; ++d) {
                        if (d != axis) squeezed.push_back(src.shape[static_cast<size_t>(d)]);
                    }
                    const uint64_t axis_size =
                        src.shape.empty() ? 1 : static_cast<uint64_t>(src.shape[static_cast<size_t>(axis)]);
                    const int64_t axis_stride =
                        src.strides.empty() ? 0 : src.strides[static_cast<size_t>(axis)];
                    body << "long base=0;{uint remaining=gid;";
                    int s = static_cast<int>(squeezed.size()) - 1;
                    for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
                        if (d == axis) continue;
                        body << "{uint c=remaining%" << (uint64_t)squeezed[s] << "u;base+=long(c)*"
                             << src.strides[d] << "L;remaining/=" << (uint64_t)squeezed[s] << "u;}";
                        --s;
                    }
                    body << "}float t=0;for(uint j=0;j<" << axis_size
                         << "u;++j)t+=A[base+long(j)*" << axis_stride << "L];Out[gid]=t;}\n";
                    graph.configs.push_back(dispatch_config(name, numel));
                }
                any_kernel = true;
                break;
            }

            default:
                throw std::runtime_error("generateKernels: unsupported opcode " +
                                         std::to_string(static_cast<int>(node.op)));
        }
    }

    if (any_kernel) {
        graph.shader_source = "#include <metal_stdlib>\nusing namespace metal;\n" + body.str();
    }
}
// Generates one huge string of all the kernel functions back to back
// if the op requires no gpu kernel (INPUT, VIEW, etc -> call it "no op"),
// we dont generate any string
// for each node in the graph, we save the associated kernel's name, in the config.name
// if it's a no-op, keep a "ghost" empty config
// so that at the end: configs.size == nodes.size (== pipelines.size)
// Shape and strides are baked into the kernel as literals.
// Offset is applied at bind time in execute() (buffer byte offset), so generated
// kernels index from 0.
// kernels generated so that out buffer is idx 0, then the N inputs to the node (in order)
