#include "soft_operators.h"

#include <stddef.h>

MeStatus me_op_soft_relu(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];
    if (!in || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (me_tensor_dtype(in) != ME_SCALAR_FLOAT32 || me_tensor_dtype(out) != ME_SCALAR_FLOAT32)
        return ME_STATUS_ERROR_UNSUPPORTED;
    if (me_tensor_nbytes(in) != me_tensor_nbytes(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    const float *src = (const float *)me_tensor_data(in);
    float       *dst = (float *)me_tensor_data(out);
    size_t       n   = me_tensor_nbytes(in) / sizeof(float);
    for (size_t i = 0; i < n; ++i)
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;

    return ME_STATUS_OK;
}
