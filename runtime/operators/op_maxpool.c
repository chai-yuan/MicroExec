#include "soft_operators.h"

#include <float.h>

static int infer_pool_params(int in_sz, int out_sz, int *k, int *s, int *p) {
    if (in_sz >= 2 && out_sz == (in_sz / 2)) {
        *k = 2;
        *s = 2;
        *p = 0;
        return 1;
    }
    if (in_sz >= 3 && out_sz == ((in_sz + 1) / 2)) {
        *k = 3;
        *s = 2;
        *p = 1;
        return 1;
    }
    for (int stride = 1; stride <= 8; ++stride) {
        for (int kernel = 2; kernel <= in_sz; ++kernel) {
            for (int pad = 0; pad <= kernel; ++pad) {
                int num = in_sz + 2 * pad - kernel;
                if (num < 0) continue;
                if ((num % stride) == 0 && (num / stride + 1) == out_sz) {
                    *k = kernel;
                    *s = stride;
                    *p = pad;
                    return 1;
                }
            }
        }
    }
    return 0;
}

MeStatus me_op_soft_maxpool(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];
    if (!in || !out) return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (me_tensor_dtype(in) != ME_SCALAR_FLOAT32 ||
        me_tensor_dtype(out) != ME_SCALAR_FLOAT32)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int32_t in_shape[4], out_shape[4];
    uint32_t in_ndim = 4, out_ndim = 4;
    if (me_tensor_shape(in, in_shape, &in_ndim) != ME_STATUS_OK ||
        me_tensor_shape(out, out_shape, &out_ndim) != ME_STATUS_OK)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (in_ndim != 4 || out_ndim != 4)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    int ON = out_shape[0], OC = out_shape[1], OH = out_shape[2], OW = out_shape[3];
    if (N != ON || C != OC) return ME_STATUS_ERROR_SHAPE_MISMATCH;

    int kh = 0, sh = 0, ph = 0;
    int kw = 0, sw = 0, pw = 0;
    if (!infer_pool_params(H, OH, &kh, &sh, &ph) ||
        !infer_pool_params(W, OW, &kw, &sw, &pw))
        return ME_STATUS_ERROR_UNSUPPORTED;

    const float *src = (const float *)me_tensor_data(in);
    float *dst = (float *)me_tensor_data(out);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float max_v = -FLT_MAX;
                    for (int k1 = 0; k1 < kh; ++k1) {
                        int ih = oh * sh + k1 - ph;
                        if (ih < 0 || ih >= H) continue;
                        for (int k2 = 0; k2 < kw; ++k2) {
                            int iw = ow * sw + k2 - pw;
                            if (iw < 0 || iw >= W) continue;
                            size_t in_idx = (size_t)(((n * C + c) * H + ih) * W + iw);
                            if (src[in_idx] > max_v) max_v = src[in_idx];
                        }
                    }
                    size_t out_idx = (size_t)(((n * C + c) * OH + oh) * OW + ow);
                    dst[out_idx] = max_v;
                }
            }
        }
    }

    return ME_STATUS_OK;
}
