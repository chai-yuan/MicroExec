#include "me_internal.h"

#include <string.h>

/**
 * 计算张量字节数 根据数据类型和形状计算张量所需的总字节数
 */
static size_t compute_nbytes(MeScalarType dtype, const int32_t *shape, uint32_t ndim) {
    size_t elem_size = me_scalar_type_size(dtype);
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

/**
 * 创建张量 根据数据类型和形状创建张量，分配形状数组和数据内存并初始化为零
 */
MeStatus me_tensor_create(MeRuntime rt, MeScalarType dtype, const int32_t *shape, uint32_t ndim, MeTensor *out) {
    if (!rt || !shape || ndim == 0 || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    size_t nbytes = compute_nbytes(dtype, shape, ndim);
    if (nbytes == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeAllocator *a = &rt->allocator;

    MeTensor t = (MeTensor)me_alloc(a, sizeof(struct MeTensor_T));
    if (!t)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(t, 0, sizeof(*t));

    t->shape = (int32_t *)me_alloc(a, ndim * sizeof(int32_t));
    if (!t->shape) {
        me_free(a, t);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    memcpy(t->shape, shape, ndim * sizeof(int32_t));

    t->data = me_alloc_aligned(a, nbytes, 16);
    if (!t->data) {
        me_free(a, t->shape);
        me_free(a, t);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    memset(t->data, 0, nbytes);

    t->dtype     = dtype;
    t->ndim      = ndim;
    t->nbytes    = nbytes;
    t->owns_data = true;
    t->allocator = a;

    *out = t;
    return ME_STATUS_OK;
}

/**
 * 销毁张量 释放张量的数据内存、形状数组和张量结构体本身
 */
void me_tensor_destroy(MeTensor tensor) {
    if (!tensor)
        return;
    MeAllocator *a = tensor->allocator;
    if (tensor->owns_data)
        me_free(a, tensor->data);
    me_free(a, tensor->shape);
    me_free(a, tensor);
}

/* ---- 公共接口：数据访问 ---------------------------------------------- */

/**
 * 设置张量数据 将源数据复制到张量的数据缓冲区中，要求大小必须匹配
 */
MeStatus me_tensor_set_data(MeTensor tensor, const void *src, size_t size) {
    if (!tensor || !src)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (size != tensor->nbytes)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    memcpy(tensor->data, src, size);
    return ME_STATUS_OK;
}

/**
 * 获取张量数据指针 返回张量数据缓冲区的指针
 */
void *me_tensor_data(MeTensor tensor) { return tensor ? tensor->data : NULL; }

/**
 * 获取张量形状 将张量的形状维度复制到输出数组，并返回实际的维度数
 */
MeStatus me_tensor_shape(MeTensor tensor, int32_t *shape_out, uint32_t *ndim_out) {
    if (!tensor || !ndim_out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t copy_n = *ndim_out < tensor->ndim ? *ndim_out : tensor->ndim;
    if (shape_out && copy_n > 0)
        memcpy(shape_out, tensor->shape, copy_n * sizeof(int32_t));
    *ndim_out = tensor->ndim;

    return ME_STATUS_OK;
}

/**
 * 获取张量数据类型 返回张量的标量数据类型
 */
MeScalarType me_tensor_dtype(MeTensor tensor) { return tensor ? tensor->dtype : ME_SCALAR_UNKNOWN; }

/**
 * 获取张量字节数 返回张量数据缓冲区的总字节数
 */
size_t me_tensor_nbytes(MeTensor tensor) { return tensor ? tensor->nbytes : 0; }
