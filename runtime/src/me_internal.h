/**
 * @file me_internal.h
 * @brief 跨运行时源文件共享的内部定义。
 *
 * 此头文件定义了不透明句柄（MeProgram_T、MeTensor_T）背后的具体结构，
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

/* ---- 算子注册表（全局子系统） ---------------------------------------- */

typedef struct MeOpEntry {
    uint32_t     hash;
    const char  *name;
    MeKernelFunc kernel;
} MeOpEntry;

#define ME_REGISTRY_INIT_CAP 64

typedef struct MeOpRegistry {
    MeOpEntry *entries;
    uint32_t   capacity;
    uint32_t   count;
} MeOpRegistry;

MeStatus     pMeOpRegistry_Init(void);
void         pMeOpRegistry_Shutdown(void);
MeStatus     pMeOpRegistry_Put(const char *name, MeKernelFunc kernel);
MeStatus     pMeOpRegistry_Remove(const char *name);
MeKernelFunc pMeOpRegistry_Lookup(const char *name);

/* ---- 内置算子注册 ------------------------------------------- */

MeStatus pMeRuntime_InitBuiltinOperators(void);

/* ---- 程序加载器 ----------------------------------------------- */

MeStatus pMeProgram_Parse(MeProgram prog, const void *data, uint32_t size);
MeStatus pMeProgram_ResolveKernels(MeProgram prog);

/* ---- 执行器 ------------------------------------------------ */

MeStatus pMeProgram_RunPlan(MeProgram prog, uint32_t plan_idx);

/* ==== 具体句柄结构 ======================================= */

struct MeProgram_T {
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
};

#endif /* ME_INTERNAL_H */
