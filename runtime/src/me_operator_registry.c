#include "me_internal.h"

#include <string.h>

/* FNV-1a hash */
static uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; ++s)
        h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static char *dup_str(MeAllocator *a, const char *s) {
    size_t len = strlen(s) + 1;
    char  *d   = (char *)me_alloc(a, len);
    if (d) memcpy(d, s, len);
    return d;
}

/* ---- Registry Lifecycle ----------------------------------------------- */

MeStatus me_registry_init(MeOpRegistry *reg, MeAllocator *alloc) {
    if (!reg || !alloc) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    reg->capacity  = ME_REGISTRY_INIT_CAP;
    reg->count     = 0;
    reg->allocator = alloc;

    size_t bytes = reg->capacity * sizeof(MeOpEntry);
    reg->entries = (MeOpEntry *)me_alloc(alloc, bytes);
    if (!reg->entries) return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(reg->entries, 0, bytes);

    return ME_STATUS_OK;
}

void me_registry_destroy(MeOpRegistry *reg) {
    if (!reg || !reg->entries) return;
    for (uint32_t i = 0; i < reg->capacity; ++i) {
        if (reg->entries[i].name)
            me_free(reg->allocator, (void *)reg->entries[i].name);
    }
    me_free(reg->allocator, reg->entries);
    reg->entries  = NULL;
    reg->capacity = 0;
    reg->count    = 0;
}

/* ---- Internal Helpers ------------------------------------------------- */

static MeStatus registry_grow(MeOpRegistry *reg) {
    uint32_t new_cap  = reg->capacity * 2;
    size_t   new_size = new_cap * sizeof(MeOpEntry);

    MeOpEntry *new_entries = (MeOpEntry *)me_alloc(reg->allocator, new_size);
    if (!new_entries) return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(new_entries, 0, new_size);

    for (uint32_t i = 0; i < reg->capacity; ++i) {
        MeOpEntry *e = &reg->entries[i];
        if (!e->name) continue;
        uint32_t slot = e->hash & (new_cap - 1);
        while (new_entries[slot].name)
            slot = (slot + 1) & (new_cap - 1);
        new_entries[slot] = *e;
    }

    me_free(reg->allocator, reg->entries);
    reg->entries  = new_entries;
    reg->capacity = new_cap;
    return ME_STATUS_OK;
}

/* ---- Public Operations ----------------------------------------------- */

MeStatus me_registry_put(MeOpRegistry *reg, const char *name,
                         MeKernelFunc kernel) {
    if (!reg || !name || !kernel) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h &&
            strcmp(reg->entries[slot].name, name) == 0) {
            reg->entries[slot].kernel = kernel;
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (reg->capacity - 1);
    }

    if (reg->count * 4 >= reg->capacity * 3) {
        MeStatus s = registry_grow(reg);
        if (s != ME_STATUS_OK) return s;
        slot = h & (reg->capacity - 1);
        while (reg->entries[slot].name)
            slot = (slot + 1) & (reg->capacity - 1);
    }

    reg->entries[slot].hash   = h;
    reg->entries[slot].name   = dup_str(reg->allocator, name);
    reg->entries[slot].kernel = kernel;
    if (!reg->entries[slot].name) return ME_STATUS_ERROR_OUT_OF_MEMORY;
    reg->count++;

    return ME_STATUS_OK;
}

MeStatus me_registry_remove(MeOpRegistry *reg, const char *name) {
    if (!reg || !name) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h &&
            strcmp(reg->entries[slot].name, name) == 0) {
            me_free(reg->allocator, (void *)reg->entries[slot].name);
            memset(&reg->entries[slot], 0, sizeof(MeOpEntry));
            reg->count--;
            /* TODO: rehash following cluster entries */
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (reg->capacity - 1);
    }

    return ME_STATUS_ERROR_OPERATOR_NOT_FOUND;
}

MeKernelFunc me_registry_lookup(const MeOpRegistry *reg, const char *name) {
    if (!reg || !name) return NULL;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h &&
            strcmp(reg->entries[slot].name, name) == 0)
            return reg->entries[slot].kernel;
        slot = (slot + 1) & (reg->capacity - 1);
    }

    return NULL;
}

/* ---- Public Wrappers (forward to registry) ---------------------------- */

MeStatus me_operator_register(MeRuntime rt, const char *op_name,
                              MeKernelFunc kernel) {
    if (!rt) return ME_STATUS_ERROR_INVALID_ARGUMENT;
    return me_registry_put(&rt->op_registry, op_name, kernel);
}

MeStatus me_operator_unregister(MeRuntime rt, const char *op_name) {
    if (!rt) return ME_STATUS_ERROR_INVALID_ARGUMENT;
    return me_registry_remove(&rt->op_registry, op_name);
}
