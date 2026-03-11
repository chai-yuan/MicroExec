#ifndef COMPILER_COMMON_EXEC_TYPES_H
#define COMPILER_COMMON_EXEC_TYPES_H

#include <cstdint>

enum class ExecValueKind : uint32_t {
    INPUT        = 0, // 图的外部输入，由调用者提供
    CONSTANT     = 1, // 常量/权重，来自模型文件
    INTERMEDIATE = 2  // 计算产生的中间值
};

enum class ExecOpKind : uint32_t {
    KERNEL   = 0, // 标准算子调用
    DELEGATE = 1, // 硬件后端代理调用
    MOVE     = 2, // 数据搬移
    NOP      = 3  // 空操作（用于 debug/profile 占位）
};

#endif // COMPILER_COMMON_EXEC_TYPES_H
