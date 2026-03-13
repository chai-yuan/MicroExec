/**
 * @file me_types.h
 * @brief MicroExec runtime public type definitions.
 *
 * Defines opaque handles, scalar types, allocator interface and runtime
 * configuration.  This header is part of the public API surface.
 */
#ifndef MICROEXEC_ME_TYPES_H
#define MICROEXEC_ME_TYPES_H

#include <stddef.h>
#include <stdint.h>

/* ---- Opaque Handles --------------------------------------------------- */

typedef struct MeRuntime_T *MeRuntime;
typedef struct MeProgram_T *MeProgram;
typedef struct MeTensor_T  *MeTensor;

/* ---- Scalar Type ------------------------------------------------------ */

/** Mirrors VMTensorScalarType values from vm_types.h for binary compatibility. */
typedef enum MeScalarType {
    ME_SCALAR_UNKNOWN = 0,
    ME_SCALAR_FLOAT32 = 1,
    ME_SCALAR_FLOAT16 = 2,
    ME_SCALAR_INT64   = 3,
    ME_SCALAR_INT32   = 4,
    ME_SCALAR_INT8    = 5,
    ME_SCALAR_UINT8   = 6,
    ME_SCALAR_BOOL    = 7,
} MeScalarType;

/** Returns the byte width of a scalar type, or 0 for unknown types. */
static inline size_t me_scalar_type_size(MeScalarType dtype) {
    switch (dtype) {
    case ME_SCALAR_FLOAT32: return 4;
    case ME_SCALAR_FLOAT16: return 2;
    case ME_SCALAR_INT64:   return 8;
    case ME_SCALAR_INT32:   return 4;
    case ME_SCALAR_INT8:    return 1;
    case ME_SCALAR_UINT8:   return 1;
    case ME_SCALAR_BOOL:    return 1;
    default:                return 0;
    }
}

/* ---- User-replaceable Allocator --------------------------------------- */

/**
 * Allocator interface.
 *
 * Users may supply their own allocator at runtime creation.
 * The runtime calls `alloc` to obtain memory and `free` to release it.
 * `ctx` is forwarded untouched to every call so that stateful allocators
 * can be implemented.
 */
typedef struct MeAllocator {
    void *(*alloc)(void *ctx, size_t size, size_t alignment);
    void  (*free)(void *ctx, void *ptr);
    void  *ctx;
} MeAllocator;

/* ---- Runtime Configuration -------------------------------------------- */

typedef struct MeRuntimeConfig {
    /** Custom allocator.  NULL selects the built-in malloc/free allocator. */
    MeAllocator *allocator;
} MeRuntimeConfig;

#endif /* MICROEXEC_ME_TYPES_H */
