#include "soft_operators.h"

MeStatus me_op_soft_softmax(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: numerically-stable softmax
     *
     * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
     *
     * inputs[0]:  input tensor
     * outputs[0]: output tensor (same shape)
     */

    return ME_STATUS_ERROR_UNSUPPORTED;
}
