#include "soft_operators.h"

#include <string.h>

MeStatus me_op_soft_reshape(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];
    if (!in || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    if (MeTensor_GetNbytes(in) != MeTensor_GetNbytes(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    if (MeTensor_GetDtype(in) != MeTensor_GetDtype(out))
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    memcpy(MeTensor_GetData(out), MeTensor_GetData(in), MeTensor_GetNbytes(in));

    return ME_STATUS_OK;
}
