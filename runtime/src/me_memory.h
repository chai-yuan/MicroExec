/**
 * @file me_memory.h
 * @brief 内部内存管理原语。
 *
 * 提供默认的malloc/free分配器、便捷包装器，以及两种专为VM执行模式设计的专用分配器：
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

/* ---- 默认分配器 ------------------------------------------------ */

// 初始化默认分配器 将分配器初始化为使用标准malloc/free进行内存分配和释放
void me_default_alloc_init(MeAllocator *alloc);

/* ---- 便捷包装器 --------------------------------------------- */

// 分配内存 使用指定分配器分配指定大小的内存，按void*对齐
static inline void *me_alloc(MeAllocator *a, size_t size) { return a->alloc(a->ctx, size, sizeof(void *)); }

// 分配对齐内存 使用指定分配器分配指定大小的内存，按指定对齐方式对齐
static inline void *me_alloc_aligned(MeAllocator *a, size_t size, size_t alignment) {
    return a->alloc(a->ctx, size, alignment);
}

// 释放内存 使用指定分配器释放先前分配的内存块
static inline void me_free(MeAllocator *a, void *ptr) {
    if (ptr)
        a->free(a->ctx, ptr);
}

/* ---- 内存池（递增分配器） ------------------------------------------- */

typedef struct MeArena {
    uint8_t     *base;
    size_t       capacity;
    size_t       offset;
    MeAllocator *parent;
} MeArena;

// 创建内存池 使用父分配器创建一个具有指定容量的内存池，获取单一连续内存块
MeStatus me_arena_init(MeArena *arena, MeAllocator *parent, size_t capacity);

// 销毁内存池 将内存池的底层内存块释放回父分配器
void me_arena_destroy(MeArena *arena);

// 从内存池分配内存 使用递增分配方式从内存池中分配指定大小和对齐方式的内存，内存池耗尽时返回NULL
void *me_arena_alloc(MeArena *arena, size_t size, size_t alignment);

// 重置内存池 重置内存池的偏移量，使所有先前的分配失效
void me_arena_reset(MeArena *arena);

// 获取已使用字节数 返回内存池当前已使用的字节数
static inline size_t me_arena_used(const MeArena *arena) { return arena->offset; }

/* ---- 块池（固定大小空闲列表） -------------------------------- */

typedef struct MeBlockPool {
    uint8_t     *base;
    size_t       block_size;
    uint32_t     block_count;
    uint32_t    *free_stack;
    uint32_t     free_top;
    MeAllocator *parent;
} MeBlockPool;

// 初始化块池 使用父分配器创建具有指定块大小和数量的块池
MeStatus me_block_pool_init(MeBlockPool *pool, MeAllocator *parent, size_t block_size, uint32_t block_count);

// 销毁块池 释放块池的空闲栈和基础内存块
void me_block_pool_destroy(MeBlockPool *pool);

// 从块池分配内存 从块池中分配一个固定大小的内存块
void *me_block_pool_alloc(MeBlockPool *pool);

// 释放内存回块池 将先前分配的内存块释放回块池的空闲列表
void me_block_pool_free(MeBlockPool *pool, void *ptr);

#endif /* ME_MEMORY_H */
