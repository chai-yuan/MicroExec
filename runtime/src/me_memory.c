#include "me_memory.h"

#include <stdlib.h>
#include <string.h>

/* ---- 全局内存子系统状态 -------------------------------------------- */

static MeAllocator g_allocator;
static bool        g_memory_initialized = false;

/* ---- 默认分配器（malloc / free）-------------------------------- */

static void *default_alloc(void *ctx, size_t size, size_t alignment) {
    (void)ctx;
    (void)alignment;
    return malloc(size);
}

static void default_free(void *ctx, void *ptr) {
    (void)ctx;
    free(ptr);
}

void pMeMemory_DefaultInit(MeAllocator *alloc) {
    alloc->alloc = default_alloc;
    alloc->free  = default_free;
    alloc->ctx   = NULL;
}

/* ---- 全局内存子系统接口 -------------------------------------------- */

MeStatus MeMemory_Init(const MeAllocator *custom_allocator) {
    if (g_memory_initialized)
        return ME_STATUS_ERROR_INTERNAL;

    if (custom_allocator) {
        g_allocator = *custom_allocator;
    } else {
        pMeMemory_DefaultInit(&g_allocator);
    }

    g_memory_initialized = true;
    return ME_STATUS_OK;
}

void MeMemory_Shutdown(void) {
    if (!g_memory_initialized)
        return;
    memset(&g_allocator, 0, sizeof(g_allocator));
    g_memory_initialized = false;
}

MeAllocator *MeMemory_GetAllocator(void) { return g_memory_initialized ? &g_allocator : NULL; }

/* ---- 内存池 ------------------------------------------------ */

MeStatus MeArena_Init(MeArena *arena, size_t capacity) {
    if (!arena || capacity == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    arena->base = (uint8_t *)me_alloc_aligned(capacity, sizeof(void *));
    if (!arena->base)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;

    arena->capacity = capacity;
    arena->offset   = 0;
    return ME_STATUS_OK;
}

void MeArena_Destroy(MeArena *arena) {
    if (!arena || !arena->base)
        return;
    me_free(arena->base);
    arena->base     = NULL;
    arena->capacity = 0;
    arena->offset   = 0;
}

void *MeArena_Alloc(MeArena *arena, size_t size, size_t alignment) {
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

void MeArena_Reset(MeArena *arena) {
    if (arena)
        arena->offset = 0;
}

/* ---- 块池 ------------------------------------------------------- */

MeStatus MeBlockPool_Init(MeBlockPool *pool, size_t block_size, uint32_t block_count) {
    if (!pool || block_size == 0 || block_count == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    size_t data_bytes  = block_size * block_count;
    size_t stack_bytes = sizeof(uint32_t) * block_count;

    pool->base = (uint8_t *)me_alloc(data_bytes);
    if (!pool->base)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;

    pool->free_stack = (uint32_t *)me_alloc(stack_bytes);
    if (!pool->free_stack) {
        me_free(pool->base);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }

    pool->block_size  = block_size;
    pool->block_count = block_count;

    for (uint32_t i = 0; i < block_count; ++i)
        pool->free_stack[i] = block_count - 1 - i;
    pool->free_top = block_count;

    return ME_STATUS_OK;
}

void MeBlockPool_Destroy(MeBlockPool *pool) {
    if (!pool)
        return;
    me_free(pool->free_stack);
    me_free(pool->base);
    memset(pool, 0, sizeof(*pool));
}

void *MeBlockPool_Alloc(MeBlockPool *pool) {
    if (!pool || pool->free_top == 0)
        return NULL;
    uint32_t idx = pool->free_stack[--pool->free_top];
    return pool->base + idx * pool->block_size;
}

void MeBlockPool_Free(MeBlockPool *pool, void *ptr) {
    if (!pool || !ptr)
        return;
    size_t   byte_offset               = (uint8_t *)ptr - pool->base;
    uint32_t idx                       = (uint32_t)(byte_offset / pool->block_size);
    pool->free_stack[pool->free_top++] = idx;
}
