#include "me_internal.h"

#include <string.h>

#define ME_VERSION_STRING "0.1.0"

/* ---- Public: Version & Status ----------------------------------------- */

const char *me_version_string(void) { return ME_VERSION_STRING; }

const char *me_status_str(MeStatus status) {
    switch (status) {
    case ME_STATUS_OK:
        return "OK";
    case ME_STATUS_ERROR_INVALID_ARGUMENT:
        return "Invalid argument";
    case ME_STATUS_ERROR_OUT_OF_MEMORY:
        return "Out of memory";
    case ME_STATUS_ERROR_INVALID_PROGRAM:
        return "Invalid program";
    case ME_STATUS_ERROR_OPERATOR_NOT_FOUND:
        return "Operator not found";
    case ME_STATUS_ERROR_SHAPE_MISMATCH:
        return "Shape mismatch";
    case ME_STATUS_ERROR_EXECUTION_FAILED:
        return "Execution failed";
    case ME_STATUS_ERROR_IO:
        return "I/O error";
    case ME_STATUS_ERROR_UNSUPPORTED:
        return "Unsupported";
    case ME_STATUS_ERROR_INTERNAL:
        return "Internal error";
    default:
        return "Unknown error";
    }
}

/* ---- Public: Runtime Lifecycle ---------------------------------------- */

MeStatus me_runtime_create(const MeRuntimeConfig *config, MeRuntime *out) {
    if (!out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeAllocator alloc;
    bool        owns = false;

    if (config && config->allocator) {
        alloc = *config->allocator;
    } else {
        me_default_alloc_init(&alloc);
        owns = true;
    }

    MeRuntime rt = (MeRuntime)me_alloc(&alloc, sizeof(struct MeRuntime_T));
    if (!rt)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(rt, 0, sizeof(*rt));

    rt->allocator      = alloc;
    rt->owns_allocator = owns;

    MeStatus s = me_registry_init(&rt->op_registry, &rt->allocator);
    if (s != ME_STATUS_OK) {
        me_free(&alloc, rt);
        return s;
    }

    s = me_register_builtin_operators(rt);
    if (s != ME_STATUS_OK) {
        me_registry_destroy(&rt->op_registry);
        me_free(&alloc, rt);
        return s;
    }

    *out = rt;
    return ME_STATUS_OK;
}

void me_runtime_destroy(MeRuntime rt) {
    if (!rt)
        return;
    me_registry_destroy(&rt->op_registry);
    MeAllocator a = rt->allocator;
    me_free(&a, rt);
}
