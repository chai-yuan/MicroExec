#ifndef COMPILER_COMMON_VM_TYPES_H
#define COMPILER_COMMON_VM_TYPES_H

#include <cstdint>

#include "common/graph_types.h"

enum class EValueType : uint32_t { NONE = 0, TENSOR, INT, DOUBLE, BOOL, STRING, TENSOR_LIST };

enum class Opcode : uint32_t {
    KERNEL_CALL = 0, // 调用标准算子
    DELEGATE_CALL,   // 调用NPU/DSP等硬件后端
    MOVE_CALL,       // 数据移动
    JUMP_FALSE_CALL, // 条件分支跳转
    FREE_CALL        // 释放内存复用
};

// 张量元数据定义 (24 字节)
struct TensorMeta {
    DataType      scalar_type;
    ShapeDynamism dynamism;
    uint32_t      ndim;
    uint32_t      shape_offset;     // 维度数组在全局 Int Pool 中的索引
    uint32_t      dim_order_offset; // 内存排布数组在全局 Int Pool 中的索引
    uint32_t      buffer_id;        // 0代表常量权重，1~N代表不同的内存池(用于中间变量)
    uint32_t      data_offset;      // 在上述 Buffer 中的字节偏移量
};

// 寄存器/变量池实体 (8 字节)
struct EValue {
    EValueType type;
    uint32_t   payload; // 如果是 Int/Bool，直接存值；如果是 Tensor/String，存对应的索引(ID)
};

// 算子定义 (8 字节)
struct OperatorDef {
    uint32_t name_idx;     // 算子名在全局 String Pool 中的索引
    uint32_t overload_idx; // 重载名在全局 String Pool 中的索引
};

// 硬件代理后端 (12 字节)
struct BackendDelegate {
    uint32_t id_idx;      // 硬件标识符(如"NPU")的 String Pool 索引
    uint32_t blob_offset; // 二进制机器码在外部大文件段中的偏移
    uint32_t blob_size;   // 二进制机器码的大小
};

// 指令结构 (16 字节)
struct Instruction {
    Opcode   opcode;
    uint32_t arg1;
    uint32_t arg2;
    uint32_t arg3;

    // 指令设计约定：
    // [KERNEL_CALL]    arg1: operator_idx, arg2: args_list_offset (入参出参列表索引), arg3: args_count
    // [DELEGATE_CALL]  arg1: delegate_idx, arg2: args_list_offset, arg3: args_count
    // [MOVE_CALL]      arg1: src_evalue_idx, arg2: dst_evalue_idx, arg3: 保留
    // [JUMP_FALSE_CALL]arg1: cond_evalue_idx, arg2: target_instruction_idx, arg3: 保留
    // [FREE_CALL]      arg1: evalue_idx, arg2: 保留, arg3: 保留
};

#endif // COMPILER_COMMON_VM_TYPES_H
