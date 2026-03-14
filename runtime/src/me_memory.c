#include "me_memory.h"

#include <stdlib.h>
#include <string.h>

/* ---- 默认分配器（malloc / free）-------------------------------- */

// 默认分配函数 使用标准malloc分配指定大小的内存，忽略上下文和对齐参数
static void *default_alloc(void *ctx, size_t size, size_t alignment) {
    (void)ctx;
    (void)alignment;
    return malloc(size);
}

// 默认释放函数 使用标准free释放内存，忽略上下文参数
static void default_free(void *ctx, void *ptr) {
    (void)ctx;
    free(ptr);
}

// 初始化默认分配器 将分配器结构体设置为使用标准malloc/free函数
void me_default_alloc_init(MeAllocator *alloc) {
    alloc->alloc = default_alloc;
    alloc->free  = default_free;
    alloc->ctx   = NULL;
}

/* ---- 内存池 ------------------------------------------------ */

// 初始化内存池 使用父分配器分配指定容量的连续内存块作为内存池的基础
MeStatus me_arena_init(MeArena *arena, MeAllocator *parent, size_t capacity) {
    if (!arena || !parent || capacity == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    arena->base = (uint8_t *)me_alloc_aligned(parent, capacity, sizeof(void *));
    if (!arena->base)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;

    arena->capacity = capacity;
    arena->offset   = 0;
    arena->parent   = parent;
    return ME_STATUS_OK;
}

// 销毁内存池 释放内存池占用的基础内存块并重置状态
void me_arena_destroy(MeArena *arena) {
    if (!arena || !arena->base)
        return;
    me_free(arena->parent, arena->base);
    arena->base     = NULL;
    arena->capacity = 0;
    arena->offset   = 0;
}

// 从内存池分配内存 按指定对齐方式从内存池当前位置递增分配指定大小的内存
void *me_arena_alloc(MeArena *arena, size_t size, size_t alignment) {
    if (!arena || !arena->base || size == 0)
        return NULL;

    size_t mask    = alignment - 1;
    size_t aligned = (arena->offset + mask) & ~mask;

    if (aligned + size > arena->capacity)
        return NULL;

    void *ptr     = arena->base + aligned;
    arena->offset = aligned + size;
    return ptr;
}

// 重置内存池 将内存池的偏移量重置为零，使所有先前的分配失效
void me_arena_reset(MeArena *arena) {
    if (arena)
        arena->offset = 0;
}

/* ---- 块池 ------------------------------------------------------- */

// 初始化块池 创建具有指定块大小和数量的块池，初始化空闲栈为所有块索引
MeStatus me_block_pool_init(MeBlockPool *pool, MeAllocator *parent, size_t block_size, uint32_t block_count) {
    if (!pool || !parent || block_size == 0 || block_count == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    size_t data_bytes  = block_size * block_count;
    size_t stack_bytes = sizeof(uint32_t) * block_count;

    pool->base = (uint8_t *)me_alloc(parent, data_bytes);
    if (!pool->base)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;

    pool->free_stack = (uint32_t *)me_alloc(parent, stack_bytes);
    if (!pool->free_stack) {
        me_free(parent, pool->base);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }

    pool->block_size  = block_size;
    pool->block_count = block_count;
    pool->parent      = parent;

    /* 将所有块索引压入空闲栈（后进先出）。 */
    for (uint32_t i = 0; i < block_count; ++i)
        pool->free_stack[i] = block_count - 1 - i;
    pool->free_top = block_count;

    return ME_STATUS_OK;
}

// 销毁块池 释放块池的空闲栈和基础内存块，并重置池状态
void me_block_pool_destroy(MeBlockPool *pool) {
    if (!pool)
        return;
    me_free(pool->parent, pool->free_stack);
    me_free(pool->parent, pool->base);
    memset(pool, 0, sizeof(*pool));
}

// 从块池分配内存 从空闲栈弹出一个块索引并返回对应的内存块地址
void *me_block_pool_alloc(MeBlockPool *pool) {
    if (!pool || pool->free_top == 0)
        return NULL;
    uint32_t idx = pool->free_stack[--pool->free_top];
    return pool->base + idx * pool->block_size;
}

// 释放内存回块池 计算指针对应的块索引并将其压入空闲栈
void me_block_pool_free(MeBlockPool *pool, void *ptr) {
    if (!pool || !ptr)
        return;
    size_t   byte_offset               = (uint8_t *)ptr - pool->base;
    uint32_t idx                       = (uint32_t)(byte_offset / pool->block_size);
    pool->free_stack[pool->free_top++] = idx;
}
