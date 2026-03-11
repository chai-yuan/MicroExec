#ifndef COMPILER_EXECUTION_EXEC_PROGRAM_H
#define COMPILER_EXECUTION_EXEC_PROGRAM_H

#include <cstdint>
#include <climits>
#include <string>
#include <vector>

#include "common/exec_types.h"
#include "common/graph_types.h"

class Graph;
class Node;
class Edge;

// 执行值：对应 Graph IR 中的一条 Edge，在 Execution IR 中统一编号
struct ExecValue {
    uint32_t      id   = 0;
    ExecValueKind kind = ExecValueKind::INTERMEDIATE;

    DataType             dtype = DataType::UNKNOWN;
    std::vector<int64_t> shape;

    // 生命周期（指令索引范围，由 AnalyzeLifetimes 填充）
    uint32_t first_def = UINT32_MAX;
    uint32_t last_use  = 0;

    // 内存规划结果（由 PlanMemory 填充）
    uint32_t buffer_id   = 0; // 0=常量池，1=运行时内存池
    uint64_t offset      = 0;
    uint64_t size_bytes  = 0;
    bool     mem_planned = false;

    // 常量数据引用（不拥有所有权，指向 Graph IR Edge 的 weight_data）
    const uint8_t *constant_data = nullptr;
    uint64_t       constant_size = 0;

    // 溯源：指向 Graph IR 的原始 Edge
    const Edge *source_edge = nullptr;
};

// 执行指令：对应 Graph IR 中的一个 Node，表示一步计算
struct ExecInstr {
    uint32_t   id      = 0;
    ExecOpKind op_kind = ExecOpKind::KERNEL;

    std::string op_name;

    std::vector<uint32_t> input_values;  // 输入 ExecValue ID 列表
    std::vector<uint32_t> output_values; // 输出 ExecValue ID 列表

    uint32_t delegate_id = 0; // 仅 DELEGATE 类型有效

    // 溯源：指向 Graph IR 的原始 Node
    const Node *source_node = nullptr;
};

// 单个内存池的规划结果
struct MemoryPoolPlan {
    uint32_t pool_id    = 0;
    uint64_t total_size = 0;
};

// 全局内存规划结果
struct MemoryPlan {
    std::vector<MemoryPoolPlan> pools;
    uint64_t constant_pool_size = 0;
    uint64_t runtime_pool_size  = 0;
};

// 从 Graph IR 降低得到的线性化执行计划
class ExecProgram {
  public:
    std::string name = "forward";

    // 主入口：从 Graph IR 构建（含拓扑排序 + 值编号 + 指令生成）
    int BuildFromGraph(const Graph &graph);

    // Pass：生命周期分析（标注每个 ExecValue 的 first_def / last_use）
    int AnalyzeLifetimes();

    // Pass：内存规划（为常量和中间值分配 buffer_id + offset）
    int PlanMemory();

    // 校验一致性
    int Validate() const;

    // 调试输出
    void Dump() const;

    const std::vector<ExecValue> &GetValues() const { return values_; }
    const std::vector<ExecInstr> &GetInstructions() const { return instructions_; }
    const std::vector<uint32_t>  &GetInputValueIds() const { return input_value_ids_; }
    const std::vector<uint32_t>  &GetOutputValueIds() const { return output_value_ids_; }
    const MemoryPlan             &GetMemoryPlan() const { return memory_plan_; }

  private:
    std::vector<ExecValue> values_;
    std::vector<ExecInstr> instructions_;

    std::vector<uint32_t> input_value_ids_;
    std::vector<uint32_t> output_value_ids_;

    MemoryPlan memory_plan_;

    uint32_t value_id_counter_ = 0;
    uint32_t instr_id_counter_ = 0;
};

#endif // COMPILER_EXECUTION_EXEC_PROGRAM_H
