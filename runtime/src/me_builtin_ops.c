/**
 * @file me_builtin_ops.c
 * @brief 将所有内置软算子注册到运行时注册表中。
 *
 * 算子名称必须与MicroExec编译器生成的名称匹配
 * （.mvmp文件中的OperatorDef.name_idx字符串）。
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

MeStatus pMeRuntime_InitBuiltinOperators(void) {
    for (size_t i = 0; i < ARRAY_LEN(kBuiltins); ++i) {
        MeStatus s = MeRuntime_Register(kBuiltins[i].name, kBuiltins[i].kernel);
        if (s != ME_STATUS_OK)
            return s;
    }
    return ME_STATUS_OK;
}
