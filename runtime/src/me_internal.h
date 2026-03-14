/**
 * @file me_internal.h
 * @brief 跨运行时源文件共享的内部定义。
 *
 * 此头文件定义了不透明句柄（MeRuntime_T、MeProgram_T、MeTensor_T）背后的具体结构，
 * 并声明了内部子系统的入口点（加载器、执行器、注册表、内置算子）。
 *
 * 此文件永远不会作为公共SDK的一部分安装。
 */
#ifndef ME_INTERNAL_H
#define ME_INTERNAL_H

#include "me_memory.h"
#include "microexec.h"
#include "vm_types.h"

#include <stdbool.h>
#include <stdint.h>

/* ---- 算子注册表 ------------------------------------------------ */

typedef struct MeOpEntry {
    uint32_t     hash;
    const char  *name; /* 算子名称的副本 */
    MeKernelFunc kernel;
} MeOpEntry;

#define ME_REGISTRY_INIT_CAP 64

/**
 * 开放寻址哈希表，将算子名称映射到内核函数。
 * name == NULL 的条目表示空槽位。
 */
typedef struct MeOpRegistry {
    MeOpEntry   *entries;
    uint32_t     capacity;
    uint32_t     count;
    MeAllocator *allocator;
} MeOpRegistry;

// 初始化算子注册表 使用指定的分配器初始化算子注册表，分配初始容量的条目数组
MeStatus pMeOpRegistry_Init(MeOpRegistry *reg, MeAllocator *alloc);
// 销毁算子注册表 释放注册表中所有条目名称和条目数组占用的内存
void pMeOpRegistry_Destroy(MeOpRegistry *reg);
// 向注册表添加算子 将算子名称和对应的内核函数添加到注册表中，支持自动扩容
MeStatus pMeOpRegistry_Put(MeOpRegistry *reg, const char *name, MeKernelFunc kernel);
// 从注册表移除算子 根据算子名称从注册表中移除对应的条目
MeStatus pMeOpRegistry_Remove(MeOpRegistry *reg, const char *name);
// 在注册表中查找算子 根据算子名称查找并返回对应的内核函数指针
MeKernelFunc pMeOpRegistry_Lookup(const MeOpRegistry *reg, const char *name);

/* ---- 内置算子注册 ------------------------------------------- */

// 注册所有内置算子 将Conv、Relu、MaxPool等内置算子注册到运行时注册表中
MeStatus pMeRuntime_InitBuiltinOperators(MeRuntime rt);

/* ---- 程序加载器 ----------------------------------------------- */

// 解析程序二进制数据 解析VM文件格式，提取各个段（字符串池、整数池、张量池等）的信息
MeStatus pMeProgram_Parse(MeProgram prog, const void *data, uint32_t size);
// 解析算子内核函数 根据算子名称在注册表中查找并绑定对应的内核函数指针
MeStatus pMeProgram_ResolveKernels(MeProgram prog);

/* ---- 执行器 ------------------------------------------------ */

// 执行指定的执行计划 按照执行计划中的指令序列依次执行，处理输入输出张量的绑定
MeStatus pMeProgram_RunPlan(MeProgram prog, uint32_t plan_idx);

/* ==== 具体句柄结构 ======================================= */

struct MeRuntime_T {
    MeAllocator  allocator;
    MeOpRegistry op_registry;
};

struct MeProgram_T {
    MeRuntime runtime;

    /* 原始二进制数据（如果从文件/缓冲区加载则拥有所有权） */
    void    *raw_data;
    uint32_t raw_size;
    bool     owns_data;

    /* 解析后的段指针 — 直接指向raw_data中的位置 */
    const VMFileHeader  *header;
    const VMSectionDesc *sections;
    uint32_t             section_count;

    const char              *string_pool;
    const int32_t           *int_pool;
    const TensorMeta        *tensor_pool;
    const EValue            *evalue_pool;
    const OperatorDef       *operator_pool;
    const BackendDelegate   *delegate_pool;
    const Instruction       *instruction_pool;
    const ExecutionPlanData *plan_pool;
    const uint8_t           *weight_data;

    uint32_t string_size;
    uint32_t string_count;
    uint32_t int_count;
    uint32_t tensor_count;
    uint32_t evalue_count;
    uint32_t operator_count;
    uint32_t delegate_count;
    uint32_t instruction_count;
    uint32_t plan_count;
    uint32_t weight_size;

    /* 预解析的内核函数指针（按operator_pool索引） */
    MeKernelFunc *resolved_kernels;

    /* 执行内存池 — 大小由ExecutionPlanData.memory_pool_size决定 */
    MeArena exec_mem;

    /* 执行期间使用的可变张量句柄数组 */
    MeTensor *io_tensors;
    bool     *io_tensor_owned;
    uint32_t  io_tensor_count;

    /* 用户绑定的输入/输出张量 */
    MeTensor *bound_inputs;
    uint32_t  bound_input_count;
    MeTensor *bound_outputs;
    uint32_t  bound_output_count;
};

struct MeTensor_T {
    MeScalarType dtype;
    int32_t     *shape;
    uint32_t     ndim;
    void        *data;
    size_t       nbytes;
    bool         owns_data;
    MeAllocator *allocator;
};

#endif /* ME_INTERNAL_H */
