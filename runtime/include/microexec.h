/**
 * @file microexec.h
 * @brief MicroExec Runtime — public API umbrella header.
 *
 * Include this single header to access the full runtime API.
 * The runtime uses opaque handles (MeRuntime, MeProgram, MeTensor)
 * and exposes all functionality through plain C functions.
 *
 * Typical usage:
 *
 *     MeRuntime rt;
 *     me_runtime_create(NULL, &rt);
 *
 *     MeProgram prog;
 *     me_program_load_file(rt, "model.mvmp", &prog);
 *
 *     MeTensor input;
 *     int32_t shape[] = {1, 1, 28, 28};
 *     me_tensor_create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);
 *     // ... fill input data ...
 *
 *     me_program_set_input(prog, 0, input);
 *     me_program_execute(prog);
 *
 *     MeTensor output;
 *     me_program_get_output(prog, 0, &output);
 *     // ... read output data ...
 *
 *     me_tensor_destroy(input);
 *     me_program_destroy(prog);
 *     me_runtime_destroy(rt);
 */
#ifndef MICROEXEC_MICROEXEC_H
#define MICROEXEC_MICROEXEC_H

#include "me_operator.h"
#include "me_status.h"
#include "me_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==== Runtime Lifecycle ================================================ */

/**
 * Create a runtime instance.
 *
 * @param config  Optional configuration (pass NULL for defaults).
 * @param out     Receives the new runtime handle on success.
 * @return ME_STATUS_OK on success.
 *
 * Built-in soft operators are automatically registered during creation.
 */
MeStatus me_runtime_create(const MeRuntimeConfig *config, MeRuntime *out);

/** Destroy a runtime and release all associated resources. */
void me_runtime_destroy(MeRuntime rt);

/* ==== Program Loading ================================================== */

/**
 * Load a compiled program from an in-memory buffer.
 *
 * The runtime makes an internal copy of @p data; the caller may free the
 * buffer after this call returns.
 */
MeStatus me_program_load(MeRuntime rt, const void *data, uint32_t size,
                         MeProgram *out);

/** Load a compiled program from a file path. */
MeStatus me_program_load_file(MeRuntime rt, const char *path, MeProgram *out);

/** Destroy a loaded program and release its resources. */
void me_program_destroy(MeProgram prog);

/** Query the number of input tensors expected by the program. */
MeStatus me_program_input_count(MeProgram prog, uint32_t *count);

/** Query the number of output tensors produced by the program. */
MeStatus me_program_output_count(MeProgram prog, uint32_t *count);

/* ==== Tensor Management ================================================ */

/**
 * Create a new tensor.
 *
 * Storage is allocated for the full element count implied by @p shape.
 * Contents are zero-initialised.
 */
MeStatus me_tensor_create(MeRuntime rt, MeScalarType dtype,
                          const int32_t *shape, uint32_t ndim,
                          MeTensor *out);

/** Destroy a user-created tensor. */
void me_tensor_destroy(MeTensor tensor);

/**
 * Copy data into a tensor.
 *
 * @param size  Byte count to copy; must equal me_tensor_nbytes(tensor).
 */
MeStatus me_tensor_set_data(MeTensor tensor, const void *src, size_t size);

/** Return a mutable pointer to the tensor's data buffer. */
void *me_tensor_data(MeTensor tensor);

/**
 * Query the shape of a tensor.
 *
 * @param shape_out  Caller-provided buffer (at least @p *ndim_out int32_t's).
 * @param ndim_out   On entry: capacity of shape_out; on exit: actual ndim.
 */
MeStatus me_tensor_shape(MeTensor tensor, int32_t *shape_out,
                         uint32_t *ndim_out);

/** Return the scalar type of a tensor. */
MeScalarType me_tensor_dtype(MeTensor tensor);

/** Return the total byte size of a tensor's data buffer. */
size_t me_tensor_nbytes(MeTensor tensor);

/* ==== Execution ======================================================== */

/** Bind an input tensor to the program before execution. */
MeStatus me_program_set_input(MeProgram prog, uint32_t index,
                              MeTensor tensor);

/** Execute the program (run all instructions in the default plan). */
MeStatus me_program_execute(MeProgram prog);

/**
 * Retrieve a borrowed reference to an output tensor after execution.
 *
 * The returned tensor is owned by the program and remains valid until the
 * next call to me_program_execute() or me_program_destroy().
 * Do NOT call me_tensor_destroy() on it.
 */
MeStatus me_program_get_output(MeProgram prog, uint32_t index,
                               MeTensor *out);

/* ==== Utility ========================================================== */

/** Return a version string for the runtime library (e.g. "0.1.0"). */
const char *me_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* MICROEXEC_MICROEXEC_H */
