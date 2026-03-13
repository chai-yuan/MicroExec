#include "soft_operators.h"

#include <stddef.h>

static int infer_conv_params(int in_sz, int k_sz, int out_sz, int *s, int *p) {
    for (int stride = 1; stride <= 8; ++stride) {
        for (int pad = 0; pad <= k_sz; ++pad) {
            int num = in_sz + 2 * pad - k_sz;
            if (num < 0)
                continue;
            if ((num % stride) == 0 && (num / stride + 1) == out_sz) {
                *s = stride;
                *p = pad;
                return 1;
            }
        }
    }
    return 0;
}

MeStatus me_op_soft_conv(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor x = ctx->inputs[0];
    MeTensor w = ctx->inputs[1];
    MeTensor b = (ctx->input_count > 2) ? ctx->inputs[2] : NULL;
    MeTensor y = ctx->outputs[0];
    if (!x || !w || !y)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (me_tensor_dtype(x) != ME_SCALAR_FLOAT32 || me_tensor_dtype(w) != ME_SCALAR_FLOAT32 ||
        me_tensor_dtype(y) != ME_SCALAR_FLOAT32)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int32_t  xs[4], ws[4], ys[4];
    uint32_t xnd = 4, wnd = 4, ynd = 4;
    if (me_tensor_shape(x, xs, &xnd) != ME_STATUS_OK || me_tensor_shape(w, ws, &wnd) != ME_STATUS_OK ||
        me_tensor_shape(y, ys, &ynd) != ME_STATUS_OK)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (xnd != 4 || wnd != 4 || ynd != 4)
        return ME_STATUS_ERROR_UNSUPPORTED;

    int N = xs[0], IC = xs[1], H = xs[2], W = xs[3];
    int OC = ws[0], WIC = ws[1], KH = ws[2], KW = ws[3];
    int ON = ys[0], OOC = ys[1], OH = ys[2], OW = ys[3];
    if (N != ON || OC != OOC || IC != WIC)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;

    int sh = 0, ph = 0, sw = 0, pw = 0;
    if (!infer_conv_params(H, KH, OH, &sh, &ph) || !infer_conv_params(W, KW, OW, &sw, &pw))
        return ME_STATUS_ERROR_UNSUPPORTED;

    const float *xptr = (const float *)me_tensor_data(x);
    const float *wptr = (const float *)me_tensor_data(w);
    const float *bptr = b ? (const float *)me_tensor_data(b) : NULL;
    float       *yptr = (float *)me_tensor_data(y);

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float acc = bptr ? bptr[oc] : 0.0f;
                    for (int ic = 0; ic < IC; ++ic) {
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * sh + kh - ph;
                            if (ih < 0 || ih >= H)
                                continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * sw + kw - pw;
                                if (iw < 0 || iw >= W)
                                    continue;
                                size_t x_idx = (size_t)(((n * IC + ic) * H + ih) * W + iw);
                                size_t w_idx = (size_t)((((oc * IC + ic) * KH) + kh) * KW + kw);
                                acc += xptr[x_idx] * wptr[w_idx];
                            }
                        }
                    }
                    size_t y_idx = (size_t)(((n * OC + oc) * OH + oh) * OW + ow);
                    yptr[y_idx]  = acc;
                }
            }
        }
    }

    return ME_STATUS_OK;
}
