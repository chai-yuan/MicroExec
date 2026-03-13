#include "soft_operators.h"

MeStatus me_op_soft_conv(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: naive im2col + GEMM convolution
     *
     * inputs[0]: input tensor  (N, C, H, W)
     * inputs[1]: weight tensor (OC, IC, KH, KW)
     * inputs[2]: bias (optional)
     * outputs[0]: output tensor (N, OC, OH, OW)
     */

    return ME_STATUS_ERROR_UNSUPPORTED;
}
