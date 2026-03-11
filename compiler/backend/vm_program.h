#include <string>
#include <vector>

#include "common/vm_types.h"

// 执行计划存储结构 (纯数据)
struct ExecutionPlanData {
    uint32_t name_idx; // 方法名 (如 "forward")

    uint32_t inputs_offset; // 输入参数在全局 Int Pool 中的索引列表起始位置
    uint32_t inputs_count;

    uint32_t outputs_offset; // 输出参数在全局 Int Pool 中的索引列表起始位置
    uint32_t outputs_count;

    uint32_t instructions_offset; // 指令流在全局 Instruction Pool 中的起始索引
    uint32_t instructions_count;

    uint32_t memory_pool_size; // 该执行计划所需的中间变量内存池大小 (预分配用)
};

class ExecutionPlan;
class ExecEngine;

// 序列化程序的统一容器
class Program {
  public:
    // 从文件加载 (通常在嵌入式中直接 mmap)
    int LoadFromFile(const std::string &file_name);

    // 获取指定的方法 (默认 "forward")
    ExecutionPlan *GetPlan(const std::string &plan_name);

    // 各种全局池的访问接口 (供引擎在运行时查询)
    const std::string &GetString(uint32_t idx) const;
    const uint32_t    *GetIntArray(uint32_t offset) const;
    const EValue      &GetEValue(uint32_t idx) const;
    const TensorMeta  &GetTensorMeta(uint32_t idx) const;

  private:
    // 平坦的存储池 (序列化时直接按字节写入，反序列化时直接读取)
    std::vector<std::string> string_pool;
    std::vector<uint32_t>    int_pool; // 用于存放 Shape 数组、参数列表等

    std::vector<EValue>          evalues;      // 寄存器池
    std::vector<TensorMeta>      tensors;      // 张量元数据池
    std::vector<OperatorDef>     operators;    // 算子表
    std::vector<BackendDelegate> delegates;    // 硬件代理表
    std::vector<Instruction>     instructions; // 全局指令池

    std::vector<ExecutionPlanData> plans; // 执行计划表

    // 权重数据的指针/内存映射引用 (这里不使用 string 避免拷贝)
    const uint8_t *weight_buffer_ptr  = nullptr;
    size_t         weight_buffer_size = 0;
};

// 辅助构建指令流文件的 Builder
class ProgramBuilder {
  public:
    ProgramBuilder();

    // 往池中添加数据并返回 32位 索引 (ID)
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

    // 将构建好的程序导出为二进制文件
    int Serialize(const std::string &output_file);

  private:
    Program program_data_; // 复用 Program 中的数据结构作为暂存
};
