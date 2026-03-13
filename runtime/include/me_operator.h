/**
 * @file me_operator.h
 * @brief Operator extension API for MicroExec runtime.
 *
 * Provides the kernel function signature and registration interface that
 * allows users to override built-in soft operators with accelerated
 * implementations.
 */
#ifndef MICROEXEC_ME_OPERATOR_H
#define MICROEXEC_ME_OPERATOR_H

#include "me_status.h"
#include "me_types.h"

/* ---- Operator Kernel Interface ---------------------------------------- */

/**
 * Execution context passed to every operator kernel.
 *
 * Kernels read data from `inputs[0..input_count)` and write results
 * into `outputs[0..output_count)`.  Output tensor storage is pre-allocated
 * by the runtime according to the memory plan; kernels should NOT allocate
 * or free output tensors themselves.
 *
 * The `allocator` field may be used for temporary scratch buffers.
 */
typedef struct MeOpContext {
    MeTensor    *inputs;
    uint32_t     input_count;
    MeTensor    *outputs;
    uint32_t     output_count;
    MeAllocator *allocator;
} MeOpContext;

/** Kernel function pointer type. */
typedef MeStatus (*MeKernelFunc)(MeOpContext *ctx);

/* ---- Registration ----------------------------------------------------- */

/**
 * Register (or override) an operator kernel.
 *
 * @param rt      Runtime handle.
 * @param op_name Operator name (e.g. "onnx::Conv").  The name must match
 *                the string stored in the compiled program file.
 * @param kernel  Kernel function pointer.
 * @return ME_STATUS_OK on success.
 *
 * If a kernel with the same name already exists (including built-in soft
 * operators), it is replaced.
 */
MeStatus me_operator_register(MeRuntime rt, const char *op_name,
                              MeKernelFunc kernel);

/**
 * Unregister an operator kernel, falling back to the built-in soft
 * implementation (if one exists).
 */
MeStatus me_operator_unregister(MeRuntime rt, const char *op_name);

#endif /* MICROEXEC_ME_OPERATOR_H */
