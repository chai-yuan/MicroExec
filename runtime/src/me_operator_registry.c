#include "me_internal.h"

#include <string.h>

/* FNV-1a哈希算法 */

/**
 * 计算字符串哈希值 使用FNV-1a算法计算字符串的32位哈希值
 */
static uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; ++s)
        h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

/**
 * 复制字符串 使用指定分配器分配内存并复制字符串内容
 */
static char *dup_str(MeAllocator *a, const char *s) {
    size_t len = strlen(s) + 1;
    char  *d   = (char *)me_alloc(a, len);
    if (d)
        memcpy(d, s, len);
    return d;
}

/* ---- 注册表生命周期 ----------------------------------------------- */

/**
 * 初始化算子注册表 使用指定分配器初始化算子注册表，分配初始容量的条目数组
 */
MeStatus me_registry_init(MeOpRegistry *reg, MeAllocator *alloc) {
    if (!reg || !alloc)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    reg->capacity  = ME_REGISTRY_INIT_CAP;
    reg->count     = 0;
    reg->allocator = alloc;

    size_t bytes = reg->capacity * sizeof(MeOpEntry);
    reg->entries = (MeOpEntry *)me_alloc(alloc, bytes);
    if (!reg->entries)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(reg->entries, 0, bytes);

    return ME_STATUS_OK;
}

/**
 * 销毁算子注册表 释放注册表中所有条目名称和条目数组占用的内存
 */
void me_registry_destroy(MeOpRegistry *reg) {
    if (!reg || !reg->entries)
        return;
    for (uint32_t i = 0; i < reg->capacity; ++i) {
        if (reg->entries[i].name)
            me_free(reg->allocator, (void *)reg->entries[i].name);
    }
    me_free(reg->allocator, reg->entries);
    reg->entries  = NULL;
    reg->capacity = 0;
    reg->count    = 0;
}

/* ---- 内部辅助函数 ------------------------------------------------- */

/**
 * 扩展注册表容量 将注册表容量翻倍，重新哈希所有现有条目到新数组
 */
static MeStatus registry_grow(MeOpRegistry *reg) {
    uint32_t new_cap  = reg->capacity * 2;
    size_t   new_size = new_cap * sizeof(MeOpEntry);

    MeOpEntry *new_entries = (MeOpEntry *)me_alloc(reg->allocator, new_size);
    if (!new_entries)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(new_entries, 0, new_size);

    for (uint32_t i = 0; i < reg->capacity; ++i) {
        MeOpEntry *e = &reg->entries[i];
        if (!e->name)
            continue;
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

/* ---- 公共操作 ----------------------------------------------- */

/**
 * 向注册表添加算子 将算子名称和对应的内核函数添加到注册表中，支持自动扩容
 */
MeStatus me_registry_put(MeOpRegistry *reg, const char *name, MeKernelFunc kernel) {
    if (!reg || !name || !kernel)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h && strcmp(reg->entries[slot].name, name) == 0) {
            reg->entries[slot].kernel = kernel;
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (reg->capacity - 1);
    }

    if (reg->count * 4 >= reg->capacity * 3) {
        MeStatus s = registry_grow(reg);
        if (s != ME_STATUS_OK)
            return s;
        slot = h & (reg->capacity - 1);
        while (reg->entries[slot].name)
            slot = (slot + 1) & (reg->capacity - 1);
    }

    reg->entries[slot].hash   = h;
    reg->entries[slot].name   = dup_str(reg->allocator, name);
    reg->entries[slot].kernel = kernel;
    if (!reg->entries[slot].name)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    reg->count++;

    return ME_STATUS_OK;
}

/**
 * 从注册表移除算子 根据算子名称从注册表中移除对应的条目
 */
MeStatus me_registry_remove(MeOpRegistry *reg, const char *name) {
    if (!reg || !name)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h && strcmp(reg->entries[slot].name, name) == 0) {
            me_free(reg->allocator, (void *)reg->entries[slot].name);
            memset(&reg->entries[slot], 0, sizeof(MeOpEntry));
            reg->count--;
            /* TODO: 重新哈希后续簇条目 */
            return ME_STATUS_OK;
        }
        slot = (slot + 1) & (reg->capacity - 1);
    }

    return ME_STATUS_ERROR_OPERATOR_NOT_FOUND;
}

/**
 * 在注册表中查找算子 根据算子名称查找并返回对应的内核函数指针
 */
MeKernelFunc me_registry_lookup(const MeOpRegistry *reg, const char *name) {
    if (!reg || !name)
        return NULL;

    uint32_t h    = hash_str(name);
    uint32_t slot = h & (reg->capacity - 1);

    while (reg->entries[slot].name) {
        if (reg->entries[slot].hash == h && strcmp(reg->entries[slot].name, name) == 0)
            return reg->entries[slot].kernel;
        slot = (slot + 1) & (reg->capacity - 1);
    }

    return NULL;
}

/* ---- 公共包装函数（转发到注册表） ---------------------------- */

/**
 * 注册算子 将算子注册到运行时的算子注册表中
 */
MeStatus me_operator_register(MeRuntime rt, const char *op_name, MeKernelFunc kernel) {
    if (!rt)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    return me_registry_put(&rt->op_registry, op_name, kernel);
}

/**
 * 注销算子 从运行时的算子注册表中移除指定算子
 */
MeStatus me_operator_unregister(MeRuntime rt, const char *op_name) {
    if (!rt)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    return me_registry_remove(&rt->op_registry, op_name);
}
