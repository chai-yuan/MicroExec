#include "soft_operators.h"

#include <math.h>
#include <stddef.h>

MeStatus me_op_soft_softmax(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];
    if (!in || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (MeTensor_GetDtype(in) != ME_SCALAR_FLOAT32 || MeTensor_GetDtype(out) != ME_SCALAR_FLOAT32)
        return ME_STATUS_ERROR_UNSUPPORTED;
    if (MeTensor_GetNbytes(in) != MeTensor_GetNbytes(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    uint32_t ndim = 0;
    if (MeTensor_GetShape(in, NULL, &ndim) != ME_STATUS_OK || ndim == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    int32_t shape[8];
    if (ndim > 8)
        return ME_STATUS_ERROR_UNSUPPORTED;
    uint32_t cap = ndim;
    if (MeTensor_GetShape(in, shape, &cap) != ME_STATUS_OK)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    size_t inner = (size_t)shape[ndim - 1];
    if (inner == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    size_t outer = 1;
    for (uint32_t i = 0; i + 1 < ndim; ++i)
        outer *= (size_t)shape[i];

    const float *src = (const float *)MeTensor_GetData(in);
    float       *dst = (float *)MeTensor_GetData(out);
    for (size_t o = 0; o < outer; ++o) {
        const float *row     = src + o * inner;
        float       *out_row = dst + o * inner;
        float        max_v   = row[0];
        for (size_t i = 1; i < inner; ++i)
            if (row[i] > max_v)
                max_v = row[i];
        float sum = 0.0f;
        for (size_t i = 0; i < inner; ++i) {
            out_row[i] = expf(row[i] - max_v);
            sum += out_row[i];
        }
        if (sum == 0.0f)
            return ME_STATUS_ERROR_EXECUTION_FAILED;
        for (size_t i = 0; i < inner; ++i)
            out_row[i] /= sum;
    }

    return ME_STATUS_OK;
}
