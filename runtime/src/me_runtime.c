#include "me_internal.h"

#include <string.h>

#define ME_VERSION_STRING "0.2.0"

static bool g_runtime_initialized = false;

const char *Microexec_Version(void) { return ME_VERSION_STRING; }

const char *MeStatus_String(MeStatus status) {
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

MeStatus MeRuntime_Init(const MeRuntimeConfig *config) {
    if (g_runtime_initialized)
        return ME_STATUS_ERROR_INTERNAL;

    const MeAllocator *custom = (config && config->allocator) ? config->allocator : NULL;
    MeStatus           s      = MeMemory_Init(custom);
    if (s != ME_STATUS_OK)
        return s;

    s = pMeOpRegistry_Init();
    if (s != ME_STATUS_OK) {
        MeMemory_Shutdown();
        return s;
    }

    s = pMeRuntime_InitBuiltinOperators();
    if (s != ME_STATUS_OK) {
        pMeOpRegistry_Shutdown();
        MeMemory_Shutdown();
        return s;
    }

    g_runtime_initialized = true;
    return ME_STATUS_OK;
}

void MeRuntime_Shutdown(void) {
    if (!g_runtime_initialized)
        return;
    pMeOpRegistry_Shutdown();
    MeMemory_Shutdown();
    g_runtime_initialized = false;
}
