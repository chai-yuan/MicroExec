#include "execution/exec_program.h"

#include "common/log.h"
#include "graph/graph.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>

static constexpr uint64_t kAlignment = 16;
// 可选优化：为静态中间张量启用生命周期复用（First-Fit）
// 关闭后仍可正确运行，只是运行时池占用更大。
static constexpr bool kEnableRuntimeReuseOptimization = true;

static uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

static uint64_t DataTypeSize(DataType dtype) {
    switch (dtype) {
    case DataType::FLOAT32:
        return 4;
    case DataType::FLOAT16:
        return 2;
    case DataType::INT64:
        return 8;
    case DataType::INT32:
        return 4;
    case DataType::INT8:
        return 1;
    case DataType::UINT8:
        return 1;
    case DataType::BOOL:
        return 1;
    default:
        return 0;
    }
}

static uint64_t ComputeTensorBytes(DataType dtype, const std::vector<int64_t> &shape) {
    if (shape.empty() || dtype == DataType::UNKNOWN) {
        return 0;
    }
    uint64_t elem_size    = DataTypeSize(dtype);
    uint64_t num_elements = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0; // 含动态维度，编译期无法确定大小
        }
        num_elements *= static_cast<uint64_t>(dim);
    }
    return num_elements * elem_size;
}

static const char *ExecValueKindStr(ExecValueKind kind) {
    switch (kind) {
    case ExecValueKind::INPUT:
        return "INPUT";
    case ExecValueKind::CONSTANT:
        return "CONST";
    case ExecValueKind::INTERMEDIATE:
        return "INTER";
    default:
        return "?";
    }
}

static const char *ExecOpKindStr(ExecOpKind kind) {
    switch (kind) {
    case ExecOpKind::KERNEL:
        return "KERNEL";
    case ExecOpKind::DELEGATE:
        return "DELEGATE";
    case ExecOpKind::MOVE:
        return "MOVE";
    case ExecOpKind::NOP:
        return "NOP";
    default:
        return "?";
    }
}

// ============================================================================
// BuildFromGraph: 拓扑排序 + 值编号 + 指令生成
// ============================================================================

int ExecProgram::BuildFromGraph(const Graph &graph) {
    const auto &nodes         = graph.GetNodes();
    const auto &edges         = graph.GetEdges();
    const auto &graph_inputs  = graph.GetGraphInputs();
    const auto &graph_outputs = graph.GetGraphOutputs();

    std::unordered_map<const Edge *, uint32_t> edge_to_value;

    // --- 阶段 1：为图输入创建 ExecValue ---
    for (const Edge *edge : graph_inputs) {
        ExecValue val;
        val.id          = value_id_counter_++;
        val.kind        = ExecValueKind::INPUT;
        val.dtype       = edge->dtype;
        val.shape       = edge->shape;
        val.size_bytes  = ComputeTensorBytes(edge->dtype, edge->shape);
        val.source_edge = edge;
        values_.push_back(val);
        edge_to_value[edge] = val.id;
        input_value_ids_.push_back(val.id);
    }

    // --- 阶段 2：为常量 Edge 创建 ExecValue ---
    for (const Edge *edge : edges) {
        if (!edge->is_constant) {
            continue;
        }
        if (edge_to_value.count(edge)) {
            continue; // 已作为 input 处理（不应出现，但防御性检查）
        }
        ExecValue val;
        val.id            = value_id_counter_++;
        val.kind          = ExecValueKind::CONSTANT;
        val.dtype         = edge->dtype;
        val.shape         = edge->shape;
        val.size_bytes    = edge->weight_data.size();
        val.constant_data = edge->weight_data.empty() ? nullptr : edge->weight_data.data();
        val.constant_size = edge->weight_data.size();
        val.buffer_id     = 0;
        val.source_edge   = edge;
        values_.push_back(val);
        edge_to_value[edge] = val.id;
    }

    // --- 阶段 3：拓扑排序（Kahn's Algorithm）---
    std::unordered_map<uint32_t, uint32_t>              in_degree;
    std::unordered_map<uint32_t, std::vector<uint32_t>> successors;
    std::unordered_map<uint32_t, const Node *>          id_to_node;

    for (const Node *node : nodes) {
        in_degree[node->id] = 0;
        id_to_node[node->id] = node;
    }

    for (const Node *node : nodes) {
        for (const Edge *input_edge : node->input_edges) {
            if (input_edge->producer != nullptr) {
                successors[input_edge->producer->id].push_back(node->id);
                in_degree[node->id]++;
            }
        }
    }

    std::queue<uint32_t> ready;
    for (const auto &[node_id, degree] : in_degree) {
        if (degree == 0) {
            ready.push(node_id);
        }
    }

    std::vector<const Node *> sorted_nodes;
    sorted_nodes.reserve(nodes.size());

    while (!ready.empty()) {
        uint32_t    node_id = ready.front();
        const Node *node    = id_to_node[node_id];
        ready.pop();
        sorted_nodes.push_back(node);

        for (uint32_t succ_id : successors[node_id]) {
            in_degree[succ_id]--;
            if (in_degree[succ_id] == 0) {
                ready.push(succ_id);
            }
        }
    }

    if (sorted_nodes.size() != nodes.size()) {
        LOG_ERROR("拓扑排序失败：图中存在环（已排序 %zu / 总计 %zu 节点）", sorted_nodes.size(), nodes.size());
        return -1;
    }

    // --- 阶段 4：按拓扑序生成 ExecInstr 和中间 ExecValue ---
    for (const Node *node : sorted_nodes) {
        // 为该节点的输出 Edge 创建 ExecValue（如果尚未创建）
        for (const Edge *out_edge : node->output_edges) {
            if (edge_to_value.count(out_edge)) {
                continue;
            }
            ExecValue val;
            val.id          = value_id_counter_++;
            val.kind        = ExecValueKind::INTERMEDIATE;
            val.dtype       = out_edge->dtype;
            val.shape       = out_edge->shape;
            val.size_bytes  = ComputeTensorBytes(out_edge->dtype, out_edge->shape);
            val.source_edge = out_edge;
            values_.push_back(val);
            edge_to_value[out_edge] = val.id;
        }

        ExecInstr instr;
        instr.id          = instr_id_counter_++;
        instr.op_kind     = ExecOpKind::KERNEL;
        instr.op_name     = node->op_type;
        instr.source_node = node;

        for (const Edge *in_edge : node->input_edges) {
            auto it = edge_to_value.find(in_edge);
            if (it != edge_to_value.end()) {
                instr.input_values.push_back(it->second);
            }
        }
        for (const Edge *out_edge : node->output_edges) {
            instr.output_values.push_back(edge_to_value[out_edge]);
        }

        instructions_.push_back(std::move(instr));
    }

    // --- 阶段 5：记录图输出对应的 ExecValue ID ---
    for (const Edge *edge : graph_outputs) {
        auto it = edge_to_value.find(edge);
        if (it != edge_to_value.end()) {
            output_value_ids_.push_back(it->second);
        } else {
            LOG_ERROR("图输出 `%s` 在 Execution IR 中找不到对应的 ExecValue", edge->name.c_str());
            return -1;
        }
    }

    LOG_INFO("Execution IR 构建完成：%zu 个值，%zu 条指令，%zu 个输入，%zu 个输出", values_.size(),
             instructions_.size(), input_value_ids_.size(), output_value_ids_.size());
    return 0;
}

// ============================================================================
// AnalyzeLifetimes: 标注每个 ExecValue 的定义点和最后使用点
// ============================================================================

int ExecProgram::AnalyzeLifetimes() {
    for (ExecValue &val : values_) {
        val.first_def = UINT32_MAX;
        val.last_use  = 0;
    }

    // 输入和常量在程序开始时就已存在
    for (ExecValue &val : values_) {
        if (val.kind == ExecValueKind::INPUT || val.kind == ExecValueKind::CONSTANT) {
            val.first_def = 0;
        }
    }

    // 扫描所有指令，标注定义点和使用点
    for (const ExecInstr &instr : instructions_) {
        for (uint32_t val_id : instr.output_values) {
            if (val_id < values_.size() && instr.id < values_[val_id].first_def) {
                values_[val_id].first_def = instr.id;
            }
        }
        for (uint32_t val_id : instr.input_values) {
            if (val_id < values_.size() && instr.id > values_[val_id].last_use) {
                values_[val_id].last_use = instr.id;
            }
        }
    }

    // 图输出必须存活到最后一条指令之后
    uint32_t program_end = instructions_.empty() ? 0 : static_cast<uint32_t>(instructions_.size());
    for (uint32_t val_id : output_value_ids_) {
        if (val_id < values_.size()) {
            values_[val_id].last_use = program_end;
        }
    }

    // 输入值存活期覆盖整个程序
    for (ExecValue &val : values_) {
        if (val.kind == ExecValueKind::INPUT) {
            val.last_use = program_end;
        }
    }

    LOG_INFO("生命周期分析完成");
    return 0;
}

// ============================================================================
// PlanMemory: 为常量和中间值分配 buffer_id + offset
// ============================================================================

int ExecProgram::PlanMemory() {
    memory_plan_.deferred_runtime_value_ids.clear();

    // --- 常量池：顺序排列在 buffer 0 ---
    uint64_t constant_offset = 0;
    for (ExecValue &val : values_) {
        if (val.kind != ExecValueKind::CONSTANT) {
            continue;
        }
        val.buffer_id   = 0;
        val.offset      = constant_offset;
        val.mem_planned = true;
        constant_offset = AlignUp(constant_offset + val.size_bytes, kAlignment);
    }
    memory_plan_.constant_pool_size = constant_offset;

    // --- 输入值由外部管理，标记为已规划 ---
    for (ExecValue &val : values_) {
        if (val.kind == ExecValueKind::INPUT) {
            val.buffer_id              = 1;
            val.offset                 = 0;
            val.deferred_runtime_alloc = (val.size_bytes == 0);
            if (val.deferred_runtime_alloc) {
                val.dynamic_alloc_symbol = "input:v" + std::to_string(val.id);
            } else {
                val.dynamic_alloc_symbol.clear();
            }
            val.mem_planned = true;
        }
    }

    // =========================================================================
    // 必须步骤：为中间值生成可执行的内存分配信息
    // - 静态大小中间值：必须有 buffer_id/offset（供 runtime 直接访问）
    // - 动态大小中间值：必须记录延迟分配符号（供 runtime 在执行时按 shape 分配）
    // =========================================================================
    uint64_t runtime_pool_size = 0;
    uint64_t runtime_bump_ptr  = 0;

    for (ExecValue &val : values_) {
        if (val.kind != ExecValueKind::INTERMEDIATE) {
            continue;
        }

        val.buffer_id = 1;
        if (val.size_bytes == 0) {
            val.offset                 = std::numeric_limits<uint64_t>::max();
            val.mem_planned            = true;
            val.deferred_runtime_alloc = true;
            val.dynamic_alloc_symbol   = "dyn:v" + std::to_string(val.id);
            memory_plan_.deferred_runtime_value_ids.push_back(val.id);
            continue;
        }

        // 先给静态值一个保守、可运行的默认布局（线性 bump-pointer）。
        val.offset                 = runtime_bump_ptr;
        val.mem_planned            = true;
        val.deferred_runtime_alloc = false;
        val.dynamic_alloc_symbol.clear();
        runtime_bump_ptr = AlignUp(runtime_bump_ptr + val.size_bytes, kAlignment);
        if (runtime_bump_ptr > runtime_pool_size) {
            runtime_pool_size = runtime_bump_ptr;
        }
    }

    // =========================================================================
    // 可选优化步骤：静态中间值内存复用（First-Fit）
    // - 输入输出语义不变
    // - 仅减少 runtime_pool_size，不影响正确性
    // =========================================================================
    if (kEnableRuntimeReuseOptimization) {
        runtime_pool_size = 0;

        // --- 中间值：贪心首次适配（Greedy First-Fit）在 buffer 1 ---
        struct LiveRegion {
            uint64_t offset;
            uint64_t size;
            uint32_t last_use;
        };
        std::vector<LiveRegion> live_regions;

        for (ExecValue &val : values_) {
            if (val.kind != ExecValueKind::INTERMEDIATE || val.size_bytes == 0) {
                continue;
            }

            // 收集与当前值生命周期重叠的已分配区域
            std::vector<std::pair<uint64_t, uint64_t>> occupied;
            for (const LiveRegion &region : live_regions) {
                if (region.last_use >= val.first_def) {
                    occupied.emplace_back(region.offset, region.offset + region.size);
                }
            }
            std::sort(occupied.begin(), occupied.end());

            // 首次适配：在已占用区域的间隙中找到第一个足够大的位置
            uint64_t candidate = 0;
            for (const auto &[start, end] : occupied) {
                if (candidate + val.size_bytes <= start) {
                    break;
                }
                if (end > candidate) {
                    candidate = AlignUp(end, kAlignment);
                }
            }

            val.buffer_id               = 1;
            val.offset                  = candidate;
            val.mem_planned             = true;
            val.deferred_runtime_alloc  = false;
            val.dynamic_alloc_symbol.clear();

            live_regions.push_back({candidate, val.size_bytes, val.last_use});

            uint64_t region_end = candidate + val.size_bytes;
            if (region_end > runtime_pool_size) {
                runtime_pool_size = region_end;
            }
        }
    }

    runtime_pool_size = AlignUp(runtime_pool_size, kAlignment);
    memory_plan_.runtime_pool_size = runtime_pool_size;

    memory_plan_.pools.clear();
    memory_plan_.pools.push_back({0, memory_plan_.constant_pool_size});
    memory_plan_.pools.push_back({1, memory_plan_.runtime_pool_size});

    LOG_INFO("内存规划完成：常量池 %lu 字节，运行时池 %lu 字节，动态延迟分配 %zu 个值",
             (unsigned long)memory_plan_.constant_pool_size, (unsigned long)memory_plan_.runtime_pool_size,
             memory_plan_.deferred_runtime_value_ids.size());
    return 0;
}

// ============================================================================
// Validate
// ============================================================================

int ExecProgram::Validate() const {
    for (const ExecInstr &instr : instructions_) {
        for (uint32_t val_id : instr.input_values) {
            if (val_id >= values_.size()) {
                LOG_ERROR("ExecProgram Validate: 指令 %u 引用了不存在的输入值 %u", instr.id, val_id);
                return -1;
            }
        }
        for (uint32_t val_id : instr.output_values) {
            if (val_id >= values_.size()) {
                LOG_ERROR("ExecProgram Validate: 指令 %u 引用了不存在的输出值 %u", instr.id, val_id);
                return -1;
            }
        }
        if (instr.output_values.empty()) {
            LOG_WARN("ExecProgram Validate: 指令 %u (%s) 没有输出值", instr.id, instr.op_name.c_str());
        }
    }

    for (uint32_t val_id : input_value_ids_) {
        if (val_id >= values_.size()) {
            LOG_ERROR("ExecProgram Validate: 程序输入引用了不存在的值 %u", val_id);
            return -1;
        }
    }
    for (uint32_t val_id : output_value_ids_) {
        if (val_id >= values_.size()) {
            LOG_ERROR("ExecProgram Validate: 程序输出引用了不存在的值 %u", val_id);
            return -1;
        }
    }

    // 检查中间值的生命周期和内存规划一致性
    for (const ExecValue &val : values_) {
        if (val.kind == ExecValueKind::INTERMEDIATE && !val.mem_planned) {
            LOG_WARN("ExecProgram Validate: 中间值 %u 尚未完成内存规划", val.id);
        }
        if (val.kind == ExecValueKind::INTERMEDIATE && val.size_bytes == 0 && !val.deferred_runtime_alloc) {
            LOG_WARN("ExecProgram Validate: 动态中间值 %u 未标记延迟分配", val.id);
        }
        if (val.kind == ExecValueKind::INTERMEDIATE && val.size_bytes == 0 && val.dynamic_alloc_symbol.empty()) {
            LOG_WARN("ExecProgram Validate: 动态中间值 %u 缺少分配符号", val.id);
        }
        if (val.kind == ExecValueKind::INTERMEDIATE && val.first_def == UINT32_MAX) {
            LOG_WARN("ExecProgram Validate: 中间值 %u 未被任何指令定义", val.id);
        }
    }

    LOG_INFO("ExecProgram 校验通过");
    return 0;
}

// ============================================================================
// Dump
// ============================================================================

void ExecProgram::Dump() const {
    LOG_INFO("=== ExecProgram Dump: \"%s\" ===", name.c_str());
    LOG_INFO("  Values: %zu, Instructions: %zu", values_.size(), instructions_.size());
    LOG_INFO("  Inputs:  [count=%zu]", input_value_ids_.size());
    LOG_INFO("  Outputs: [count=%zu]", output_value_ids_.size());

    LOG_INFO("--- Values ---");
    for (const ExecValue &val : values_) {
        const char *edge_name = val.source_edge ? val.source_edge->name.c_str() : "(none)";
        if (val.deferred_runtime_alloc) {
            LOG_INFO("  v%-4u  %-5s  size=dynamic    life=[%u, %u]  buf=%u off=deferred  sym=\"%s\"  edge=\"%s\"",
                     val.id, ExecValueKindStr(val.kind), val.first_def, val.last_use, val.buffer_id,
                     val.dynamic_alloc_symbol.c_str(), edge_name);
        } else {
            LOG_INFO("  v%-4u  %-5s  size=%-10lu  life=[%u, %u]  buf=%u off=%-8lu  edge=\"%s\"", val.id,
                     ExecValueKindStr(val.kind), (unsigned long)val.size_bytes, val.first_def, val.last_use, val.buffer_id,
                     (unsigned long)val.offset, edge_name);
        }
    }

    LOG_INFO("--- Instructions ---");
    for (const ExecInstr &instr : instructions_) {
        const char *node_name = instr.source_node ? instr.source_node->name.c_str() : "(none)";

        std::string in_str;
        for (size_t i = 0; i < instr.input_values.size(); ++i) {
            if (i > 0) in_str += ", ";
            in_str += "v" + std::to_string(instr.input_values[i]);
        }

        std::string out_str;
        for (size_t i = 0; i < instr.output_values.size(); ++i) {
            if (i > 0) out_str += ", ";
            out_str += "v" + std::to_string(instr.output_values[i]);
        }

        LOG_INFO("  i%-4u  %-8s  %-20s  in=[%s]  out=[%s]  node=\"%s\"", instr.id, ExecOpKindStr(instr.op_kind),
                 instr.op_name.c_str(), in_str.c_str(), out_str.c_str(), node_name);
    }

    LOG_INFO("--- Memory Plan ---");
    LOG_INFO("  Constant pool: %lu bytes", (unsigned long)memory_plan_.constant_pool_size);
    LOG_INFO("  Runtime pool:  %lu bytes", (unsigned long)memory_plan_.runtime_pool_size);
    LOG_INFO("  Deferred runtime values: %zu", memory_plan_.deferred_runtime_value_ids.size());
    LOG_INFO("=== End Dump ===");
}
