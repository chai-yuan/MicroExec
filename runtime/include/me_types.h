/**
 * @file me_types.h
 * @brief MicroExec 运行时公共类型定义。
 *
 * 定义不透明句柄、标量类型、分配器接口和运行时配置。
 * 此头文件属于公共 API 的一部分。
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

/** 与 vm_types.h 中的 VMTensorScalarType 值保持一致以确保二进制兼容 */
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

/** 获取标量类型的字节宽度（未知类型返回 0） */
static inline size_t MeScalarType_Size(MeScalarType dtype) {
    switch (dtype) {
    case ME_SCALAR_FLOAT32:
        return 4;
    case ME_SCALAR_FLOAT16:
        return 2;
    case ME_SCALAR_INT64:
        return 8;
    case ME_SCALAR_INT32:
        return 4;
    case ME_SCALAR_INT8:
        return 1;
    case ME_SCALAR_UINT8:
        return 1;
    case ME_SCALAR_BOOL:
        return 1;
    default:
        return 0;
    }
}

/* ---- User-replaceable Allocator --------------------------------------- */

/** 分配器接口（用户可在创建运行时传入自定义分配器） */
typedef struct MeAllocator {
    void *(*alloc)(void *ctx, size_t size, size_t alignment);
    void (*free)(void *ctx, void *ptr);
    void *ctx;
} MeAllocator;

/* ---- Runtime Configuration -------------------------------------------- */

typedef struct MeRuntimeConfig {
    /** 自定义分配器（NULL 表示使用内置 malloc/free） */
    MeAllocator *allocator;
} MeRuntimeConfig;

#endif /* MICROEXEC_ME_TYPES_H */
