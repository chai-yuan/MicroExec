#include "me_internal.h"

#include <string.h>

/* ---- 全局注册表实例 ------------------------------------------------ */

static MeOpRegistry g_registry;

/* FNV-1a哈希算法 */

static uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; ++s)
        h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static char *dup_str(const char *s) {
    size_t len = strlen(s) + 1;
    char  *d   = (char *)me_alloc(len);
    if (d)
        memcpy(d, s, len);
    return d;
}

/* ---- 注册表生命周期 ----------------------------------------------- */

MeStatus pMeOpRegistry_Init(void) {
    g_registry.capacity = ME_REGISTRY_INIT_CAP;
    g_registry.count    = 0;

    size_t bytes       = g_registry.capacity * sizeof(MeOpEntry);
    g_registry.entries = (MeOpEntry *)me_alloc(bytes);
    if (!g_registry.entries)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(g_registry.entries, 0, bytes);

    return ME_STATUS_OK;
}

void pMeOpRegistry_Shutdown(void) {
    if (!g_registry.entries)
        return;
    for (uint32_t i = 0; i < g_registry.capacity; ++i) {
        if (g_registry.entries[i].name)
            me_free((void *)g_registry.entries[i].name);
    }
    me_free(g_registry.entries);
    g_registry.entries  = NULL;
    g_registry.capacity = 0;
    g_registry.count    = 0;
}

/* ---- 内部辅助函数 ------------------------------------------------- */

static MeStatus registry_grow(void) {
    uint32_t new_cap  = g_registry.capacity * 2;
    size_t   new_size = new_cap * sizeof(MeOpEntry);

    MeOpEntry *new_entries = (MeOpEntry *)me_alloc(new_size);
    if (!new_entries)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(new_entries, 0, new_size);

    for (uint32_t i = 0; i < g_registry.capacity; ++i) {
        MeOpEntry *e = &g_registry.entries[i];
        if (!e->name)
            continue;
        uint32_t slot = e->hash & (new_cap - 1);
        while (new_entries[slot].name)
            slot = (slot + 1) & (new_cap - 1);
        new_entries[slot] = *e;
    }

    me_free(g_registry.entries);
    g_registry.entries  = new_entries;
    g_registry.capacity = new_cap;
    return ME_STATUS_OK;
}

/* ---- 注册表操作 ----------------------------------------------- */

MeStatus pMeOpRegistry_Put(const char *name, MeKernelFunc kernel) {
    if (!name || !kernel)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (g_registry.capacity - 1);

    while (g_registry.entries[slot].name) {
        if (g_registry.entries[slot].hash == h && strcmp(g_registry.entries[slot].name, name) == 0) {
            g_registry.entries[slot].kernel = kernel;
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (g_registry.capacity - 1);
    }

    if (g_registry.count * 4 >= g_registry.capacity * 3) {
        MeStatus s = registry_grow();
        if (s != ME_STATUS_OK)
            return s;
        slot = h & (g_registry.capacity - 1);
        while (g_registry.entries[slot].name)
            slot = (slot + 1) & (g_registry.capacity - 1);
    }

    g_registry.entries[slot].hash   = h;
    g_registry.entries[slot].name   = dup_str(name);
    g_registry.entries[slot].kernel = kernel;
    if (!g_registry.entries[slot].name)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    g_registry.count++;

    return ME_STATUS_OK;
}

MeStatus pMeOpRegistry_Remove(const char *name) {
    if (!name)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (g_registry.capacity - 1);

    while (g_registry.entries[slot].name) {
        if (g_registry.entries[slot].hash == h && strcmp(g_registry.entries[slot].name, name) == 0) {
            me_free((void *)g_registry.entries[slot].name);
            memset(&g_registry.entries[slot], 0, sizeof(MeOpEntry));
            g_registry.count--;
            /* TODO: 重新哈希后续簇条目 */
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (g_registry.capacity - 1);
    }

    return ME_STATUS_ERROR_OPERATOR_NOT_FOUND;
}

MeKernelFunc pMeOpRegistry_Lookup(const char *name) {
    if (!name || !g_registry.entries)
        return NULL;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (g_registry.capacity - 1);

    while (g_registry.entries[slot].name) {
        if (g_registry.entries[slot].hash == h && strcmp(g_registry.entries[slot].name, name) == 0)
            return g_registry.entries[slot].kernel;
        slot = (slot + 1) & (g_registry.capacity - 1);
    }

    return NULL;
}

/* ---- 公共包装函数 ------------------------------------------------- */

MeStatus MeRuntime_Register(const char *op_name, MeKernelFunc kernel) { return pMeOpRegistry_Put(op_name, kernel); }

MeStatus MeRuntime_Unregister(const char *op_name) { return pMeOpRegistry_Remove(op_name); }
