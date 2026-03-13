/**
 * @file me_operator.h
 * @brief MicroExec 运行时算子扩展 API。
 *
 * 提供内核函数签名和注册接口，允许用户用加速实现覆盖内置软算子。
 */
#ifndef MICROEXEC_ME_OPERATOR_H
#define MICROEXEC_ME_OPERATOR_H

#include "me_status.h"
#include "me_types.h"

/** 算子执行上下文（内核从中读取输入，写入输出） */
typedef struct MeOpContext {
    MeTensor    *inputs;
    uint32_t     input_count;
    MeTensor    *outputs;
    uint32_t     output_count;
    MeAllocator *allocator;
} MeOpContext;

/** 内核函数指针类型 */
typedef MeStatus (*MeKernelFunc)(MeOpContext *ctx);


/** 注册或覆盖算子内核（同名内核将被替换） */
MeStatus me_operator_register(MeRuntime rt, const char *op_name,
                              MeKernelFunc kernel);

/** 注销算子内核（回退到内置软实现） */
MeStatus me_operator_unregister(MeRuntime rt, const char *op_name);

#endif /* MICROEXEC_ME_OPERATOR_H */
