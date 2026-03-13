#include "soft_operators.h"

MeStatus me_op_soft_reshape(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: reshape — zero-copy when contiguous
     *
     * inputs[0]: data tensor
     * inputs[1]: shape tensor (int64 1-D)
     * outputs[0]: reshaped view
     */

    return ME_STATUS_ERROR_UNSUPPORTED;
}
