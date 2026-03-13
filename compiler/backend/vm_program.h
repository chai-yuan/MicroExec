#ifndef COMPILER_BACKEND_VM_PROGRAM_H
#define COMPILER_BACKEND_VM_PROGRAM_H

#include <cstdint>
#include <string>
#include <vector>

#include "common/vm_types.h"

class ExecProgram;

// 仅用于编译期：将 Execution IR 降低为 VM Program 并序列化。
class ProgramBuilder {
  public:
    ProgramBuilder();

    // 将 Execution IR 降低为 VM Program 构建数据。
    int BuildFromExecProgram(const ExecProgram &exec);

    // 往各类线格式池中添加数据并返回 32 位索引。
    uint32_t AddString(const std::string &str);
    uint32_t AddIntArray(const std::vector<uint32_t> &arr);
    uint32_t AddTensorMeta(const TensorMeta &meta);
    uint32_t AddEValue(const EValue &evalue);
    uint32_t AddOperator(const std::string &name, const std::string &overload);
    uint32_t AddDelegate(const std::string &backend_id, uint32_t blob_offset, uint32_t blob_size);

    // 构建指令
    void AppendInstruction(const Instruction &inst);

    // 创建执行计划
    void BuildExecutionPlan(const std::string &name, const std::vector<uint32_t> &inputs,
                            const std::vector<uint32_t> &outputs, uint32_t memory_pool_size);

    // 将构建好的程序导出为 VM 二进制文件。
    int Serialize(const std::string &output_file);

  private:
    std::vector<uint8_t>           weight_pool_;
    uint32_t                       weight_tensor_count_ = 0;
    std::vector<std::string>       string_pool_;
    std::vector<uint32_t>          int_pool_;
    std::vector<TensorMeta>        tensor_pool_;
    std::vector<EValue>            evalue_pool_;
    std::vector<OperatorDef>       operator_pool_;
    std::vector<BackendDelegate>   delegate_pool_;
    std::vector<Instruction>       instruction_pool_;
    std::vector<ExecutionPlanData> plan_pool_;
};

#endif // COMPILER_BACKEND_VM_PROGRAM_H
