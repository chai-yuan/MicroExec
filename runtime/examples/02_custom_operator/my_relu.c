#include "my_relu.h"
#include "microexec.h"

#include <stdio.h>

/**
 * Custom ReLU implementation.
 *
 * Identical semantics to the built-in soft ReLU but prints a message
 * to show that the override is active.  In production, this would
 * contain platform-specific SIMD intrinsics or other optimisations.
 */
MeStatus my_fast_relu(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    printf("[custom] my_fast_relu called\n");

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];

    size_t       n   = MeTensor_GetNbytes(in) / sizeof(float);
    const float *src = (const float *)MeTensor_GetData(in);
    float       *dst = (float *)MeTensor_GetData(out);

    for (size_t i = 0; i < n; ++i)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;

    return ME_STATUS_OK;
}
