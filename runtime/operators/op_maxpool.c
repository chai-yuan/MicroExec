#include "soft_operators.h"

MeStatus me_op_soft_maxpool(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: sliding-window max pooling
     *
     * inputs[0]:  input tensor (N, C, H, W)
     * outputs[0]: output tensor (N, C, OH, OW)
     *
     * Kernel size, strides, padding read from operator attributes
     * (to be passed through MeOpContext in a future revision).
     */

    return ME_STATUS_ERROR_UNSUPPORTED;
}
