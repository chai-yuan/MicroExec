#include "soft_operators.h"

MeStatus me_op_soft_relu(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: element-wise max(0, x)
     *
     * inputs[0]:  input tensor
     * outputs[0]: output tensor (same shape)
     */

    return ME_STATUS_ERROR_UNSUPPORTED;
}
