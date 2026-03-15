#include "me_internal.h"

#include <string.h>

static size_t compute_nbytes(MeScalarType dtype, const int32_t *shape, uint32_t ndim) {
    size_t elem_size = MeScalarType_Size(dtype);
    if (elem_size == 0)
        return 0;

    size_t count = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0)
            return 0;
        count *= (size_t)shape[i];
    }
    return count * elem_size;
}

/* ---- 公共接口：张量生命周期 ----------------------------------------- */

MeStatus MeTensor_Create(MeScalarType dtype, const int32_t *shape, uint32_t ndim, MeTensor *out) {
    if (!shape || ndim == 0 || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    size_t nbytes = compute_nbytes(dtype, shape, ndim);
    if (nbytes == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeTensor t = (MeTensor)me_alloc(sizeof(struct MeTensor_T));
    if (!t)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(t, 0, sizeof(*t));

    t->shape = (int32_t *)me_alloc(ndim * sizeof(int32_t));
    if (!t->shape) {
        me_free(t);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    memcpy(t->shape, shape, ndim * sizeof(int32_t));

    t->data = me_alloc_aligned(nbytes, 16);
    if (!t->data) {
        me_free(t->shape);
        me_free(t);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    memset(t->data, 0, nbytes);

    t->dtype     = dtype;
    t->ndim      = ndim;
    t->nbytes    = nbytes;
    t->owns_data = true;

    *out = t;
    return ME_STATUS_OK;
}

void MeTensor_Destroy(MeTensor tensor) {
    if (!tensor)
        return;
    if (tensor->owns_data)
        me_free(tensor->data);
    me_free(tensor->shape);
    me_free(tensor);
}

/* ---- 公共接口：数据访问 ---------------------------------------------- */

MeStatus MeTensor_SetData(MeTensor tensor, const void *src, size_t size) {
    if (!tensor || !src)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (size != tensor->nbytes)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    memcpy(tensor->data, src, size);
    return ME_STATUS_OK;
}

void *MeTensor_GetData(MeTensor tensor) { return tensor ? tensor->data : NULL; }

MeStatus MeTensor_GetShape(MeTensor tensor, int32_t *shape_out, uint32_t *ndim_out) {
    if (!tensor || !ndim_out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t copy_n = *ndim_out < tensor->ndim ? *ndim_out : tensor->ndim;
    if (shape_out && copy_n > 0)
        memcpy(shape_out, tensor->shape, copy_n * sizeof(int32_t));
    *ndim_out = tensor->ndim;

    return ME_STATUS_OK;
}

MeScalarType MeTensor_GetDtype(MeTensor tensor) { return tensor ? tensor->dtype : ME_SCALAR_UNKNOWN; }

size_t MeTensor_GetNbytes(MeTensor tensor) { return tensor ? tensor->nbytes : 0; }
