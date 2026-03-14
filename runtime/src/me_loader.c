#include "me_internal.h"
#include "soft_operators.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 检查边界是否有效 验证给定的偏移量和大小是否在文件大小范围内
static bool bounds_ok(uint32_t ofs, uint32_t size, uint32_t file_size) {
    return ofs <= file_size && size <= (file_size - ofs);
}

// 检查乘法溢出 检查两个size_t相乘是否会导致溢出，并将结果写入输出参数
static bool mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return false;
    }
    if (a > SIZE_MAX / b)
        return true;
    *out = a * b;
    return false;
}

// 获取字符串池中的字符串 从字符串池中获取指定索引的字符串及其长度
static const char *string_pool_at(const char *pool, uint32_t count, uint32_t size_bytes, uint32_t idx,
                                  uint32_t *out_len) {
    if (!pool || idx >= count || !out_len)
        return NULL;
    uint32_t off = 0;
    for (uint32_t i = 0; i <= idx; ++i) {
        if (off > size_bytes || size_bytes - off < sizeof(uint32_t))
            return NULL;
        uint32_t len = 0;
        memcpy(&len, pool + off, sizeof(uint32_t));
        off += sizeof(uint32_t);
        if (off > size_bytes || len > size_bytes - off)
            return NULL;
        if (i == idx) {
            *out_len = len;
            return pool + off;
        }
        off += len;
    }
    return NULL;
}

// 获取回退内置内核函数 根据算子名称返回对应的软件实现内核函数指针
static MeKernelFunc fallback_builtin_kernel(const char *name) {
    if (!name)
        return NULL;
    if (strcmp(name, "Conv") == 0 || strcmp(name, "onnx::Conv") == 0)
        return me_op_soft_conv;
    if (strcmp(name, "Relu") == 0 || strcmp(name, "onnx::Relu") == 0)
        return me_op_soft_relu;
    if (strcmp(name, "MaxPool") == 0 || strcmp(name, "onnx::MaxPool") == 0)
        return me_op_soft_maxpool;
    if (strcmp(name, "Gemm") == 0 || strcmp(name, "onnx::Gemm") == 0)
        return me_op_soft_gemm;
    if (strcmp(name, "Reshape") == 0 || strcmp(name, "onnx::Reshape") == 0)
        return me_op_soft_reshape;
    if (strcmp(name, "Softmax") == 0 || strcmp(name, "onnx::Softmax") == 0)
        return me_op_soft_softmax;
    return NULL;
}

/* ---- 内部：二进制解析器 ------------------------------------------ */

// 解析程序二进制数据 解析VM文件格式，验证文件头并提取各个段（字符串池、整数池、张量池等）的信息
MeStatus pMeProgram_Parse(MeProgram prog, const void *data, uint32_t size) {
    if (size < sizeof(VMFileHeader))
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    const VMFileHeader *hdr = (const VMFileHeader *)data;
    if (hdr->magic != kVMFileMagic)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    if (hdr->version_major != kVMFileVersionMajor)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (hdr->header_size < sizeof(VMFileHeader))
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (hdr->file_size != size)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (hdr->section_count == 0)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    size_t sec_tbl_bytes = 0;
    if (mul_overflow_size((size_t)hdr->section_count, sizeof(VMSectionDesc), &sec_tbl_bytes))
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (hdr->section_table_ofs > size || sec_tbl_bytes > (size_t)(size - hdr->section_table_ofs))
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    prog->header        = hdr;
    prog->section_count = hdr->section_count;
    prog->sections      = (const VMSectionDesc *)((const uint8_t *)data + hdr->section_table_ofs);

    for (uint32_t i = 0; i < prog->section_count; ++i) {
        const VMSectionDesc *sec = &prog->sections[i];
        if (!bounds_ok(sec->offset, sec->size_bytes, size))
            return ME_STATUS_ERROR_INVALID_PROGRAM;

        const uint8_t *base = (const uint8_t *)data + sec->offset;
        switch ((VMSectionKind)sec->kind) {
        case VM_SECTION_STRINGS:
            prog->string_pool  = (const char *)base;
            prog->string_size  = sec->size_bytes;
            prog->string_count = sec->count;
            break;
        case VM_SECTION_INTS:
            if ((sec->size_bytes % sizeof(int32_t)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->int_pool  = (const int32_t *)base;
            prog->int_count = sec->count;
            break;
        case VM_SECTION_TENSORS:
            if ((sec->size_bytes % sizeof(TensorMeta)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->tensor_pool  = (const TensorMeta *)base;
            prog->tensor_count = sec->count;
            break;
        case VM_SECTION_EVALUES:
            if ((sec->size_bytes % sizeof(EValue)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->evalue_pool  = (const EValue *)base;
            prog->evalue_count = sec->count;
            break;
        case VM_SECTION_OPERATORS:
            if ((sec->size_bytes % sizeof(OperatorDef)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->operator_pool  = (const OperatorDef *)base;
            prog->operator_count = sec->count;
            break;
        case VM_SECTION_DELEGATES:
            if ((sec->size_bytes % sizeof(BackendDelegate)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->delegate_pool  = (const BackendDelegate *)base;
            prog->delegate_count = sec->count;
            break;
        case VM_SECTION_INSTRUCTIONS:
            if ((sec->size_bytes % sizeof(Instruction)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->instruction_pool  = (const Instruction *)base;
            prog->instruction_count = sec->count;
            break;
        case VM_SECTION_EXEC_PLANS:
            if ((sec->size_bytes % sizeof(ExecutionPlanData)) != 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->plan_pool  = (const ExecutionPlanData *)base;
            prog->plan_count = sec->count;
            break;
        case VM_SECTION_WEIGHTS:
            prog->weight_data = (const uint8_t *)base;
            prog->weight_size = sec->size_bytes;
            break;
        default:
            break;
        }
    }

    if (!prog->plan_pool || prog->plan_count == 0)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (hdr->entry_plan_idx >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (!prog->int_pool || !prog->evalue_pool || !prog->tensor_pool || !prog->instruction_pool)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    const ExecutionPlanData *entry = &prog->plan_pool[hdr->entry_plan_idx];
    if (entry->inputs_offset > prog->int_count || entry->inputs_count > prog->int_count - entry->inputs_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (entry->outputs_offset > prog->int_count || entry->outputs_count > prog->int_count - entry->outputs_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (entry->instructions_offset > prog->instruction_count ||
        entry->instructions_count > prog->instruction_count - entry->instructions_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    MeAllocator *a = &prog->runtime->allocator;
    if (entry->inputs_count > 0) {
        prog->bound_inputs = (MeTensor *)me_alloc(a, entry->inputs_count * sizeof(MeTensor));
        if (!prog->bound_inputs)
            return ME_STATUS_ERROR_OUT_OF_MEMORY;
        memset(prog->bound_inputs, 0, entry->inputs_count * sizeof(MeTensor));
    }
    if (entry->outputs_count > 0) {
        prog->bound_outputs = (MeTensor *)me_alloc(a, entry->outputs_count * sizeof(MeTensor));
        if (!prog->bound_outputs)
            return ME_STATUS_ERROR_OUT_OF_MEMORY;
        memset(prog->bound_outputs, 0, entry->outputs_count * sizeof(MeTensor));
    }
    prog->bound_input_count  = entry->inputs_count;
    prog->bound_output_count = entry->outputs_count;

    return ME_STATUS_OK;
}

// 解析算子内核函数 根据算子名称在注册表中查找并绑定对应的内核函数指针，支持回退到软件实现
MeStatus pMeProgram_ResolveKernels(MeProgram prog) {
    if (prog->operator_count == 0)
        return ME_STATUS_OK;

    MeAllocator *a         = &prog->runtime->allocator;
    prog->resolved_kernels = (MeKernelFunc *)me_alloc(a, prog->operator_count * sizeof(MeKernelFunc));
    if (!prog->resolved_kernels)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;

    uint32_t strings_size = 0;
    for (uint32_t i = 0; i < prog->section_count; ++i) {
        if (prog->sections[i].kind == VM_SECTION_STRINGS) {
            strings_size = prog->sections[i].size_bytes;
            break;
        }
    }
    if (!prog->string_pool && prog->operator_count > 0)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    for (uint32_t i = 0; i < prog->operator_count; ++i) {
        const OperatorDef *op       = &prog->operator_pool[i];
        uint32_t           name_len = 0;
        const char        *name_ptr =
            string_pool_at(prog->string_pool, prog->string_count, strings_size, op->name_idx, &name_len);
        if (!name_ptr)
            return ME_STATUS_ERROR_INVALID_PROGRAM;

        char *name = (char *)me_alloc(a, (size_t)name_len + 1);
        if (!name)
            return ME_STATUS_ERROR_OUT_OF_MEMORY;
        memcpy(name, name_ptr, name_len);
        name[name_len] = '\0';

        MeKernelFunc k = pMeOpRegistry_Lookup(&prog->runtime->op_registry, name);
        if (!k) {
            const char *onnx_prefix = "onnx::";
            if (strncmp(name, onnx_prefix, 6) == 0) {
                k = pMeOpRegistry_Lookup(&prog->runtime->op_registry, name + 6);
            } else {
                size_t aliased_len = name_len + 6;
                char  *alias       = (char *)me_alloc(a, aliased_len + 1);
                if (!alias) {
                    me_free(a, name);
                    return ME_STATUS_ERROR_OUT_OF_MEMORY;
                }
                memcpy(alias, onnx_prefix, 6);
                memcpy(alias + 6, name, name_len + 1);
                k = pMeOpRegistry_Lookup(&prog->runtime->op_registry, alias);
                me_free(a, alias);
            }
        }
        if (!k)
            k = fallback_builtin_kernel(name);
        me_free(a, name);
        if (!k)
            return ME_STATUS_ERROR_OPERATOR_NOT_FOUND;
        prog->resolved_kernels[i] = k;
    }

    return ME_STATUS_OK;
}

/* ---- 公共接口：程序加载 ------------------------------------------ */

// 从内存加载程序 从内存缓冲区加载程序数据，解析并解析内核函数
MeStatus MeProgram_CreateFromBuffer(MeRuntime rt, const void *data, uint32_t size, MeProgram *out) {
    if (!rt || !data || !size || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeAllocator *a = &rt->allocator;

    MeProgram prog = (MeProgram)me_alloc(a, sizeof(struct MeProgram_T));
    if (!prog)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(prog, 0, sizeof(*prog));
    prog->runtime = rt;

    prog->raw_data = me_alloc(a, size);
    if (!prog->raw_data) {
        me_free(a, prog);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    memcpy(prog->raw_data, data, size);
    prog->raw_size  = size;
    prog->owns_data = true;

    MeStatus s = pMeProgram_Parse(prog, prog->raw_data, prog->raw_size);
    if (s != ME_STATUS_OK) {
        MeProgram_Destroy(prog);
        return s;
    }

    s = pMeProgram_ResolveKernels(prog);
    if (s != ME_STATUS_OK) {
        MeProgram_Destroy(prog);
        return s;
    }

    *out = prog;
    return ME_STATUS_OK;
}

// 销毁程序 释放程序占用的所有资源，包括执行内存、张量、内核函数数组等
void MeProgram_Destroy(MeProgram prog) {
    if (!prog)
        return;
    MeAllocator *a = &prog->runtime->allocator;

    me_arena_destroy(&prog->exec_mem);

    if (prog->io_tensors && prog->io_tensor_owned) {
        for (uint32_t i = 0; i < prog->io_tensor_count; ++i) {
            if (prog->io_tensor_owned[i] && prog->io_tensors[i])
                MeTensor_Destroy(prog->io_tensors[i]);
        }
    }
    if (prog->resolved_kernels)
        me_free(a, prog->resolved_kernels);
    if (prog->io_tensor_owned)
        me_free(a, prog->io_tensor_owned);
    if (prog->io_tensors)
        me_free(a, prog->io_tensors);
    if (prog->bound_inputs)
        me_free(a, prog->bound_inputs);
    if (prog->bound_outputs)
        me_free(a, prog->bound_outputs);

    if (prog->owns_data && prog->raw_data)
        me_free(a, prog->raw_data);

    me_free(a, prog);
}

// 获取程序输入数量 返回程序入口执行计划的输入张量数量
MeStatus MeProgram_InputCount(MeProgram prog, uint32_t *count) {
    if (!prog || !count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = prog->header ? prog->header->entry_plan_idx : 0;
    if (!prog->plan_pool || entry >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    *count = prog->plan_pool[entry].inputs_count;
    return ME_STATUS_OK;
}

// 获取程序输出数量 返回程序入口执行计划的输出张量数量
MeStatus MeProgram_OutputCount(MeProgram prog, uint32_t *count) {
    if (!prog || !count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = prog->header ? prog->header->entry_plan_idx : 0;
    if (!prog->plan_pool || entry >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    *count = prog->plan_pool[entry].outputs_count;
    return ME_STATUS_OK;
}
