/**
 * @file me_builtin_ops.c
 * @brief Registers all built-in soft operators into the runtime registry.
 *
 * Operator names must match those emitted by the MicroExec compiler
 * (OperatorDef.name_idx strings in the .mvmp file).
 */
#include "me_internal.h"
#include "soft_operators.h"

typedef struct {
    const char  *name;
    MeKernelFunc kernel;
} BuiltinEntry;

static const BuiltinEntry kBuiltins[] = {
    {"Conv", me_op_soft_conv}, {"Relu", me_op_soft_relu},       {"MaxPool", me_op_soft_maxpool},
    {"Gemm", me_op_soft_gemm}, {"Reshape", me_op_soft_reshape}, {"Softmax", me_op_soft_softmax},
};

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

MeStatus me_register_builtin_operators(MeRuntime rt) {
    for (size_t i = 0; i < ARRAY_LEN(kBuiltins); ++i) {
        MeStatus s = me_operator_register(rt, kBuiltins[i].name, kBuiltins[i].kernel);
        if (s != ME_STATUS_OK)
            return s;
    }
    return ME_STATUS_OK;
}
