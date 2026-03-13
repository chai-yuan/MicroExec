#include "x86_ops.h"

#include <stdio.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

/* ---- x86 ReLU (SSE2) ------------------------------------------------- */

MeStatus x86_relu(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 1 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor in  = ctx->inputs[0];
    MeTensor out = ctx->outputs[0];

    size_t       n   = me_tensor_nbytes(in) / sizeof(float);
    const float *src = (const float *)me_tensor_data(in);
    float       *dst = (float *)me_tensor_data(out);

    printf("[x86] SSE2 ReLU — %zu elements\n", n);

#ifdef __SSE2__
    __m128 zero = _mm_setzero_ps();
    size_t i    = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dst + i, _mm_max_ps(v, zero));
    }
    for (; i < n; ++i)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
#else
    for (size_t i = 0; i < n; ++i)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
#endif

    return ME_STATUS_OK;
}

/* ---- x86 GEMM (stub — AVX2 placeholder) ------------------------------ */

MeStatus x86_gemm(MeOpContext *ctx) {
    if (!ctx || ctx->input_count < 2 || ctx->output_count < 1)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    printf("[x86] AVX2 GEMM (stub)\n");

    /* TODO: implement tiled GEMM using AVX2 _mm256 intrinsics */

    return ME_STATUS_ERROR_UNSUPPORTED;
}

/* ---- Batch Registration ----------------------------------------------- */

MeStatus x86_register_operators(MeRuntime rt) {
    MeStatus s;

    s = me_operator_register(rt, "Relu", x86_relu);
    if (s != ME_STATUS_OK)
        return s;

    s = me_operator_register(rt, "Gemm", x86_gemm);
    if (s != ME_STATUS_OK)
        return s;

    return ME_STATUS_OK;
}
