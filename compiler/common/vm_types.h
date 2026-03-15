#ifndef COMPILER_COMMON_VM_TYPES_H
#define COMPILER_COMMON_VM_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
#include <type_traits>
extern "C" {
#endif

#if defined(__cplusplus)
#define VM_STATIC_ASSERT(condition, message) static_assert(condition, message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define VM_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#else
#define VM_STATIC_ASSERT(condition, message)
#endif

// 该头文件描述 VM 二进制线格式，只包含 C/C++ 共享的 POD 类型与常量定义。
static const uint32_t kVMDataAlignment    = 4u;
static const uint32_t kVMFileMagic        = 0x504D564Du; // "MVMP"
static const uint16_t kVMFileVersionMajor = 1u;
static const uint16_t kVMFileVersionMinor = 0u;

typedef enum VMEValueType {
    EVALUE_TYPE_NONE        = 0,
    EVALUE_TYPE_TENSOR      = 1,
    EVALUE_TYPE_INT         = 2,
    EVALUE_TYPE_DOUBLE      = 3,
    EVALUE_TYPE_BOOL        = 4,
    EVALUE_TYPE_STRING      = 5,
    EVALUE_TYPE_TENSOR_LIST = 6
} EValueType;

typedef enum VMOpcode {
    OPCODE_KERNEL_CALL     = 0, // 调用标准算子
    OPCODE_DELEGATE_CALL   = 1, // 调用 NPU/DSP 等硬件后端
    OPCODE_MOVE_CALL       = 2, // 数据移动
    OPCODE_JUMP_FALSE_CALL = 3, // 条件分支跳转
    OPCODE_FREE_CALL       = 4, // 释放内存复用
    OPCODE_NOP_CALL        = 5  // 空操作（调试/Profile 占位）
} Opcode;

typedef enum VMInstructionFlags {
    INSTR_FLAG_NONE            = 0,
    INSTR_FLAG_HAS_IMM_PAYLOAD = 1u << 0, // arg3 指向立即数池偏移
    INSTR_FLAG_INPLACE         = 1u << 1, // 可原地执行
    INSTR_FLAG_RESERVED2       = 1u << 2,
    INSTR_FLAG_RESERVED3       = 1u << 3,
    INSTR_FLAG_RESERVED4       = 1u << 4,
    INSTR_FLAG_RESERVED5       = 1u << 5,
    INSTR_FLAG_RESERVED6       = 1u << 6,
    INSTR_FLAG_RESERVED7       = 1u << 7
} InstructionFlags;

typedef enum VMTensorScalarType {
    VM_TENSOR_SCALAR_UNKNOWN = 0,
    VM_TENSOR_SCALAR_FLOAT32 = 1,
    VM_TENSOR_SCALAR_FLOAT16 = 2,
    VM_TENSOR_SCALAR_INT64   = 3,
    VM_TENSOR_SCALAR_INT32   = 4,
    VM_TENSOR_SCALAR_INT8    = 5,
    VM_TENSOR_SCALAR_UINT8   = 6,
    VM_TENSOR_SCALAR_BOOL    = 7,
    VM_TENSOR_SCALAR_STRING  = 8
} VMTensorScalarType;

typedef enum VMTensorShapeDynamism {
    VM_TENSOR_SHAPE_STATIC          = 0,
    VM_TENSOR_SHAPE_DYNAMIC_BOUND   = 1,
    VM_TENSOR_SHAPE_DYNAMIC_UNBOUND = 2
} VMTensorShapeDynamism;

// 张量元数据定义 (28 字节)
typedef struct TensorMeta {
    uint32_t scalar_type;    // VMTensorScalarType
    uint32_t shape_dynamism; // VMTensorShapeDynamism
    uint32_t ndim;
    uint32_t shape_offset;     // 维度数组在全局 Int Pool 中的索引
    uint32_t dim_order_offset; // 内存排布数组在全局 Int Pool 中的索引
    uint32_t buffer_id;        // 0 代表常量权重，1~N 代表不同的内存池
    uint32_t data_offset;      // 在上述 Buffer 中的字节偏移量，动态值使用uint32_t最大值作为“运行时分配”哨兵。
} TensorMeta;

// 寄存器/变量池实体 (8 字节)
typedef struct EValue {
    uint32_t type;    // EValueType
    uint32_t payload; // 如果是 Int/Bool，直接存值；如果是 Tensor/String，存对应的索引(ID)
} EValue;

// 算子定义 (8 字节)
typedef struct OperatorDef {
    uint32_t name_idx;     // 算子名在全局 String Pool 中的索引
    uint32_t overload_idx; // 重载名在全局 String Pool 中的索引
} OperatorDef;

// 硬件代理后端 (12 字节)
typedef struct BackendDelegate {
    uint32_t id_idx;      // 硬件标识符(如 "NPU")的 String Pool 索引
    uint32_t blob_offset; // 二进制机器码在外部大文件段中的偏移
    uint32_t blob_size;   // 二进制机器码的大小
} BackendDelegate;

// 指令结构 (16 字节)
typedef struct Instruction {
    uint8_t  opcode;       // Opcode
    uint8_t  flags;        // InstructionFlags
    uint8_t  input_count;  // args_list 中输入参数个数
    uint8_t  output_count; // args_list 中输出参数个数
    uint32_t arg1;
    uint32_t arg2;
    uint32_t arg3;
} Instruction;

// 指令设计约定：
// [KERNEL_CALL]     arg1: operator_idx, arg2: args_list_offset, arg3: args_count
// [DELEGATE_CALL]   arg1: delegate_idx, arg2: args_list_offset, arg3: args_count
// [MOVE_CALL]       arg1: src_evalue_idx, arg2: dst_evalue_idx, arg3: 保留
// [JUMP_FALSE_CALL] arg1: cond_evalue_idx, arg2: target_instruction_idx, arg3: 保留
// [FREE_CALL]       arg1: evalue_idx, arg2/arg3: 保留
// [NOP_CALL]        arg1/arg2/arg3: 可选调试元数据

// 单个 execution plan 的线格式描述。
typedef struct ExecutionPlanData {
    uint32_t name_idx; // 方法名 (如 "forward")
    uint32_t inputs_offset;
    uint32_t inputs_count;
    uint32_t outputs_offset;
    uint32_t outputs_count;
    uint32_t instructions_offset;
    uint32_t instructions_count;
    uint32_t memory_pool_size; // 所需中间变量内存池大小
} ExecutionPlanData;

// VM Program 二进制文件固定头（64 字节，4 字节对齐）
typedef struct VMFileHeader {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t header_size;
    uint32_t file_size;
    uint32_t section_count;
    uint32_t section_table_ofs;
    uint32_t entry_plan_idx;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
    uint32_t reserved6;
    uint32_t reserved7;
} VMFileHeader;

typedef enum VMSectionKind {
    VM_SECTION_STRINGS      = 0,
    VM_SECTION_INTS         = 1,
    VM_SECTION_TENSORS      = 2,
    VM_SECTION_EVALUES      = 3,
    VM_SECTION_OPERATORS    = 4,
    VM_SECTION_DELEGATES    = 5,
    VM_SECTION_INSTRUCTIONS = 6,
    VM_SECTION_EXEC_PLANS   = 7,
    VM_SECTION_WEIGHTS      = 8
} VMSectionKind;

// Section 索引项（16 字节）
typedef struct VMSectionDesc {
    uint32_t kind;       // VMSectionKind
    uint32_t offset;     // 段在文件内的起始偏移（4 字节对齐）
    uint32_t size_bytes; // 段字节数
    uint32_t count;      // 该段内元素数量（字节流段可为 0）
} VMSectionDesc;

VM_STATIC_ASSERT(sizeof(TensorMeta) % 4u == 0, "TensorMeta must be 4-byte aligned");
VM_STATIC_ASSERT(sizeof(EValue) % 4u == 0, "EValue must be 4-byte aligned");
VM_STATIC_ASSERT(sizeof(OperatorDef) % 4u == 0, "OperatorDef must be 4-byte aligned");
VM_STATIC_ASSERT(sizeof(BackendDelegate) % 4u == 0, "BackendDelegate must be 4-byte aligned");
VM_STATIC_ASSERT(sizeof(Instruction) == 16u, "Instruction must remain 16 bytes");
VM_STATIC_ASSERT(sizeof(ExecutionPlanData) % 4u == 0, "ExecutionPlanData alignment mismatch");
VM_STATIC_ASSERT(sizeof(VMFileHeader) == 64u, "VMFileHeader must be 64 bytes");
VM_STATIC_ASSERT(sizeof(VMSectionDesc) == 16u, "VMSectionDesc must be 16 bytes");
VM_STATIC_ASSERT(sizeof(VMFileHeader) % 4u == 0, "VMFileHeader alignment mismatch");
VM_STATIC_ASSERT(sizeof(VMSectionDesc) % 4u == 0, "VMSectionDesc alignment mismatch");

#ifdef __cplusplus
}
static_assert(alignof(Instruction) >= alignof(uint32_t), "Instruction should be naturally aligned");
static_assert(std::is_trivially_copyable<Instruction>::value, "Instruction must be trivially copyable");
#endif

#undef VM_STATIC_ASSERT

#endif // COMPILER_COMMON_VM_TYPES_H
