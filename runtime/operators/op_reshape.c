#include "soft_operators.h"

#include <string.h>

MeStatus me_op_soft_reshape(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];
    if (!in || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    if (me_tensor_nbytes(in) != me_tensor_nbytes(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    if (me_tensor_dtype(in) != me_tensor_dtype(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    memcpy(me_tensor_data(out), me_tensor_data(in), me_tensor_nbytes(in));

    return ME_STATUS_OK;
}
