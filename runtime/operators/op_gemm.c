#include "soft_operators.h"

#include <stddef.h>

MeStatus me_op_soft_gemm(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor A = ctx->inputs[0];
    MeTensor B = ctx->inputs[1];
    MeTensor C = (ctx->input_count > 2) ? ctx->inputs[2] : NULL;
    MeTensor Y = ctx->outputs[0];
    if (!A || !B || !Y) return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (me_tensor_dtype(A) != ME_SCALAR_FLOAT32 ||
        me_tensor_dtype(B) != ME_SCALAR_FLOAT32 ||
        me_tensor_dtype(Y) != ME_SCALAR_FLOAT32)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int32_t a_shape[2], b_shape[2], y_shape[2];
    uint32_t andim = 2, bndim = 2, yndim = 2;
    if (me_tensor_shape(A, a_shape, &andim) != ME_STATUS_OK ||
        me_tensor_shape(B, b_shape, &bndim) != ME_STATUS_OK ||
        me_tensor_shape(Y, y_shape, &yndim) != ME_STATUS_OK)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (andim != 2 || bndim != 2 || yndim != 2)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int M = a_shape[0], K = a_shape[1];
    int b0 = b_shape[0], b1 = b_shape[1];
    int N = 0;
    int trans_b = 0;
    if (K == b0) {
        N = b1;
        trans_b = 0;
    } else if (K == b1) {
        N = b0;
        trans_b = 1;
    } else {
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    }
    if (y_shape[0] != M || y_shape[1] != N)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    const float *ap = (const float *)me_tensor_data(A);
    const float *bp = (const float *)me_tensor_data(B);
    const float *cp = C ? (const float *)me_tensor_data(C) : NULL;
    float *yp = (float *)me_tensor_data(Y);

    int32_t c_shape[2] = {0, 0};
    uint32_t cndim = 2;
    if (C && me_tensor_shape(C, c_shape, &cndim) != ME_STATUS_OK)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float bv = trans_b ? bp[(size_t)n * K + k]
                                   : bp[(size_t)k * N + n];
                acc += ap[(size_t)m * K + k] * bv;
            }

            if (cp) {
                if (cndim == 1 && c_shape[0] == N) {
                    acc += cp[n];
                } else if (cndim == 2 && c_shape[0] == M && c_shape[1] == N) {
                    acc += cp[(size_t)m * N + n];
                } else if (me_tensor_nbytes(C) == sizeof(float)) {
                    acc += cp[0];
                } else {
                    return ME_STATUS_ERROR_SHAPE_MISMATCH;
                }
            }
            yp[(size_t)m * N + n] = acc;
        }
    }

    return ME_STATUS_OK;
}
