/**
 * @file me_memory.h
 * @brief 独立内存管理模块。
 *
 * 提供全局内存子系统的初始化/关闭接口，以及便捷的分配/释放包装器。
 * 同时包含两种专为VM执行模式设计的专用分配器：
 *
 *   MeArena     – 线性递增分配器，仅支持批量释放。
 *                 用于执行内存池，其总大小在程序加载时已知。
 *
 *   MeBlockPool – 固定块大小的空闲列表分配器，分配/释放时间复杂度为O(1)。
 *                 用于统一大小的结构（如EValue槽位）。
 */
#ifndef ME_MEMORY_H
#define ME_MEMORY_H

#include "me_status.h"
#include "me_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- 全局内存子系统 ------------------------------------------------ */

/** 初始化全局内存子系统（传入 NULL 使用默认 malloc/free） */
MeStatus MeMemory_Init(const MeAllocator *custom_allocator);

/** 关闭全局内存子系统 */
void MeMemory_Shutdown(void);

/** 获取当前活跃的全局分配器（未初始化时返回 NULL） */
MeAllocator *MeMemory_GetAllocator(void);

/* ---- 默认分配器 ------------------------------------------------ */

void pMeMemory_DefaultInit(MeAllocator *alloc);

/* ---- 便捷包装器（使用全局分配器） --------------------------------------------- */

static inline void *me_alloc(size_t size) {
    MeAllocator *a = MeMemory_GetAllocator();
    return a->alloc(a->ctx, size, sizeof(void *));
}

static inline void *me_alloc_aligned(size_t size, size_t alignment) {
    MeAllocator *a = MeMemory_GetAllocator();
    return a->alloc(a->ctx, size, alignment);
}

static inline void me_free(void *ptr) {
    if (ptr) {
        MeAllocator *a = MeMemory_GetAllocator();
        a->free(a->ctx, ptr);
    }
}

/* ---- 内存池（递增分配器） ------------------------------------------- */

typedef struct MeArena {
    uint8_t *base;
    size_t   capacity;
    size_t   offset;
} MeArena;

MeStatus MeArena_Init(MeArena *arena, size_t capacity);
void     MeArena_Destroy(MeArena *arena);
void    *MeArena_Alloc(MeArena *arena, size_t size, size_t alignment);
void     MeArena_Reset(MeArena *arena);

static inline size_t MeArena_Used(const MeArena *arena) { return arena->offset; }

/* ---- 块池（固定大小空闲列表） -------------------------------- */

typedef struct MeBlockPool {
    uint8_t  *base;
    size_t    block_size;
    uint32_t  block_count;
    uint32_t *free_stack;
    uint32_t  free_top;
} MeBlockPool;

MeStatus MeBlockPool_Init(MeBlockPool *pool, size_t block_size, uint32_t block_count);
void     MeBlockPool_Destroy(MeBlockPool *pool);
void    *MeBlockPool_Alloc(MeBlockPool *pool);
void     MeBlockPool_Free(MeBlockPool *pool, void *ptr);

#endif /* ME_MEMORY_H */
