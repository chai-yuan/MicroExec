/**
 * @file me_memory.h
 * @brief Internal memory management primitives.
 *
 * Provides a default malloc/free allocator, convenience wrappers, and two
 * special-purpose allocators tailored to VM execution patterns:
 *
 *   MeArena     – linear bump allocator, bulk-free only.
 *                 Used for the execution memory pool whose total size is
 *                 known at program-load time.
 *
 *   MeBlockPool – fixed-block-size free-list allocator with O(1) alloc/free.
 *                 Used for uniform-size structures (e.g. EValue slots).
 */
#ifndef ME_MEMORY_H
#define ME_MEMORY_H

#include "me_status.h"
#include "me_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- Default Allocator ------------------------------------------------ */

/** Initialise @p alloc to wrap the standard malloc / free. */
void me_default_alloc_init(MeAllocator *alloc);

/* ---- Convenience Wrappers --------------------------------------------- */

static inline void *me_alloc(MeAllocator *a, size_t size) {
    return a->alloc(a->ctx, size, sizeof(void *));
}

static inline void *me_alloc_aligned(MeAllocator *a, size_t size,
                                     size_t alignment) {
    return a->alloc(a->ctx, size, alignment);
}

static inline void me_free(MeAllocator *a, void *ptr) {
    if (ptr) a->free(a->ctx, ptr);
}

/* ---- Arena (Bump Allocator) ------------------------------------------- */

typedef struct MeArena {
    uint8_t     *base;
    size_t       capacity;
    size_t       offset;
    MeAllocator *parent;
} MeArena;

/**
 * Create an arena backed by @p parent.
 * The arena obtains a single contiguous block of @p capacity bytes.
 */
MeStatus me_arena_init(MeArena *arena, MeAllocator *parent, size_t capacity);

/** Release the underlying block back to the parent allocator. */
void me_arena_destroy(MeArena *arena);

/**
 * Bump-allocate @p size bytes with the given alignment.
 * Returns NULL if the arena is exhausted.
 */
void *me_arena_alloc(MeArena *arena, size_t size, size_t alignment);

/** Reset the arena (all prior allocations become invalid). */
void me_arena_reset(MeArena *arena);

/** Return the number of bytes currently in use. */
static inline size_t me_arena_used(const MeArena *arena) {
    return arena->offset;
}

/* ---- Block Pool (Fixed-size Free-list) -------------------------------- */

typedef struct MeBlockPool {
    uint8_t     *base;
    size_t       block_size;
    uint32_t     block_count;
    uint32_t    *free_stack;
    uint32_t     free_top;
    MeAllocator *parent;
} MeBlockPool;

MeStatus me_block_pool_init(MeBlockPool *pool, MeAllocator *parent,
                            size_t block_size, uint32_t block_count);
void     me_block_pool_destroy(MeBlockPool *pool);
void    *me_block_pool_alloc(MeBlockPool *pool);
void     me_block_pool_free(MeBlockPool *pool, void *ptr);

#endif /* ME_MEMORY_H */
