/**
 * @file me_internal.h
 * @brief Internal definitions shared across runtime source files.
 *
 * This header defines the concrete structures behind the opaque handles
 * (MeRuntime_T, MeProgram_T, MeTensor_T) and declares internal sub-system
 * entry points (loader, executor, registry, built-in operators).
 *
 * It is NEVER installed as part of the public SDK.
 */
#ifndef ME_INTERNAL_H
#define ME_INTERNAL_H

#include "microexec.h"
#include "me_memory.h"
#include "vm_types.h"

#include <stdbool.h>
#include <stdint.h>

/* ---- Operator Registry ------------------------------------------------ */

typedef struct MeOpEntry {
    uint32_t     hash;
    const char  *name;     /* owned copy of the operator name */
    MeKernelFunc kernel;
} MeOpEntry;

#define ME_REGISTRY_INIT_CAP 64

/**
 * Open-addressing hash table mapping operator names to kernel functions.
 * Entries with name == NULL are empty slots.
 */
typedef struct MeOpRegistry {
    MeOpEntry   *entries;
    uint32_t     capacity;
    uint32_t     count;
    MeAllocator *allocator;
} MeOpRegistry;

MeStatus     me_registry_init(MeOpRegistry *reg, MeAllocator *alloc);
void         me_registry_destroy(MeOpRegistry *reg);
MeStatus     me_registry_put(MeOpRegistry *reg, const char *name,
                             MeKernelFunc kernel);
MeStatus     me_registry_remove(MeOpRegistry *reg, const char *name);
MeKernelFunc me_registry_lookup(const MeOpRegistry *reg, const char *name);

/* ---- Built-in Operator Registration ----------------------------------- */

MeStatus me_register_builtin_operators(MeRuntime rt);

/* ---- Program Loader --------------------------------------------------- */

MeStatus me_loader_parse(MeProgram prog, const void *data, uint32_t size);
MeStatus me_loader_resolve_kernels(MeProgram prog);

/* ---- Executor --------------------------------------------------------- */

MeStatus me_executor_run_plan(MeProgram prog, uint32_t plan_idx);

/* ==== Concrete Handle Structures ======================================= */

struct MeRuntime_T {
    MeAllocator  allocator;
    bool         owns_allocator;
    MeOpRegistry op_registry;
};

struct MeProgram_T {
    MeRuntime runtime;

    /* Raw binary data (owned if loaded from file / buffer) */
    void    *raw_data;
    uint32_t raw_size;
    bool     owns_data;

    /* Parsed section pointers — point directly into raw_data */
    const VMFileHeader      *header;
    const VMSectionDesc     *sections;
    uint32_t                 section_count;

    const char              *string_pool;
    const int32_t           *int_pool;
    const TensorMeta        *tensor_pool;
    const EValue            *evalue_pool;
    const OperatorDef       *operator_pool;
    const BackendDelegate   *delegate_pool;
    const Instruction       *instruction_pool;
    const ExecutionPlanData *plan_pool;
    const uint8_t           *weight_data;

    uint32_t string_count;
    uint32_t int_count;
    uint32_t tensor_count;
    uint32_t evalue_count;
    uint32_t operator_count;
    uint32_t delegate_count;
    uint32_t instruction_count;
    uint32_t plan_count;
    uint32_t weight_size;

    /* Pre-resolved kernel pointers (indexed by operator_pool index) */
    MeKernelFunc *resolved_kernels;

    /* Execution memory pool — sized by ExecutionPlanData.memory_pool_size */
    MeArena exec_mem;

    /* Mutable tensor handle array used during execution */
    MeTensor *io_tensors;
    uint32_t  io_tensor_count;

    /* User-bound input / output tensors */
    MeTensor *bound_inputs;
    uint32_t  bound_input_count;
    MeTensor *bound_outputs;
    uint32_t  bound_output_count;
};

struct MeTensor_T {
    MeScalarType  dtype;
    int32_t      *shape;
    uint32_t      ndim;
    void         *data;
    size_t        nbytes;
    bool          owns_data;
    MeAllocator  *allocator;
};

#endif /* ME_INTERNAL_H */
