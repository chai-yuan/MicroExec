#include "me_internal.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static size_t tensor_nbytes_from_meta(const TensorMeta *meta, const int32_t *int_pool, uint32_t int_count) {
    if (!meta || !int_pool)
        return 0;
    size_t elem_size = MeScalarType_Size((MeScalarType)meta->scalar_type);
    if (elem_size == 0)
        return 0;
    if (meta->ndim == 0)
        return 0;
    if (meta->shape_offset > int_count || meta->ndim > int_count - meta->shape_offset)
        return 0;

    size_t count = 1;
    for (uint32_t i = 0; i < meta->ndim; ++i) {
        int32_t dim = int_pool[meta->shape_offset + i];
        if (dim <= 0)
            dim = 1;
        if (count > SIZE_MAX / (size_t)dim)
            return 0;
        count *= (size_t)dim;
    }
    if (count > SIZE_MAX / elem_size)
        return 0;
    return count * elem_size;
}

static MeStatus get_tensor_dims(MeTensor t, int32_t *dims, uint32_t *ndim) {
    if (!t || !dims || !ndim || *ndim == 0)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    return MeTensor_GetShape(t, dims, ndim);
}

static MeStatus ensure_tensor_storage(MeProgram prog, const TensorMeta *meta, MeTensor t) {
    if (!prog || !meta || !t)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (meta->buffer_id == 0)
        return ME_STATUS_OK;

    if (meta->data_offset != UINT32_MAX && prog->exec_mem.base) {
        if (meta->data_offset > prog->exec_mem.capacity || t->nbytes > prog->exec_mem.capacity - meta->data_offset)
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        if (t->owns_data && t->data) {
            me_free(t->data);
        }
        t->data      = prog->exec_mem.base + meta->data_offset;
        t->owns_data = false;
        return ME_STATUS_OK;
    }

    if (t->owns_data && t->data) {
        me_free(t->data);
        t->data = NULL;
    }
    t->data = me_alloc_aligned(t->nbytes, 16);
    if (!t->data)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    t->owns_data = true;
    return ME_STATUS_OK;
}

static MeStatus apply_runtime_shape_to_tensor(MeProgram prog, MeTensor t, const TensorMeta *meta,
                                              const int32_t *ref_dims, uint32_t ref_ndim) {
    if (!prog || !t || !meta)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (meta->ndim == 0 || meta->ndim > 8)
        return ME_STATUS_ERROR_UNSUPPORTED;
    if (meta->shape_offset > prog->int_count || meta->ndim > prog->int_count - meta->shape_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (t->ndim != meta->ndim)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    size_t elem_size = MeScalarType_Size(t->dtype);
    if (elem_size == 0)
        return ME_STATUS_ERROR_UNSUPPORTED;

    size_t count = 1;
    for (uint32_t i = 0; i < meta->ndim; ++i) {
        int32_t dim = prog->int_pool[meta->shape_offset + i];
        if (dim <= 0) {
            dim = (i < ref_ndim && ref_dims) ? ref_dims[i] : 1;
        }
        if (dim <= 0)
            return ME_STATUS_ERROR_SHAPE_MISMATCH;
        t->shape[i] = dim;
        if (count > SIZE_MAX / (size_t)dim)
            return ME_STATUS_ERROR_SHAPE_MISMATCH;
        count *= (size_t)dim;
    }
    if (count > SIZE_MAX / elem_size)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    t->nbytes = count * elem_size;
    return ME_STATUS_OK;
}

static MeStatus create_view_tensor(MeProgram prog, const TensorMeta *meta, void *data, MeTensor *out) {
    if (!prog || !meta || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (meta->ndim == 0)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (meta->shape_offset > prog->int_count || meta->ndim > prog->int_count - meta->shape_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    size_t nbytes = tensor_nbytes_from_meta(meta, prog->int_pool, prog->int_count);
    if (nbytes == 0)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    MeTensor t = (MeTensor)me_alloc(sizeof(struct MeTensor_T));
    if (!t)
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(t, 0, sizeof(*t));

    t->shape = (int32_t *)me_alloc(meta->ndim * sizeof(int32_t));
    if (!t->shape) {
        me_free(t);
        return ME_STATUS_ERROR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < meta->ndim; ++i) {
        int32_t dim = prog->int_pool[meta->shape_offset + i];
        t->shape[i] = (dim <= 0) ? 1 : dim;
    }

    t->dtype     = (MeScalarType)meta->scalar_type;
    t->ndim      = meta->ndim;
    t->data      = data;
    t->nbytes    = nbytes;
    t->owns_data = false;

    *out = t;
    return ME_STATUS_OK;
}

static bool evalue_is_plan_input(const ExecutionPlanData *plan, const int32_t *int_pool, uint32_t int_count,
                                 uint32_t evalue_idx) {
    if (!plan || !int_pool)
        return false;
    if (plan->inputs_offset > int_count || plan->inputs_count > int_count - plan->inputs_offset)
        return false;
    for (uint32_t i = 0; i < plan->inputs_count; ++i) {
        if ((uint32_t)int_pool[plan->inputs_offset + i] == evalue_idx)
            return true;
    }
    return false;
}

static MeStatus ensure_tensor_views(MeProgram prog, const ExecutionPlanData *plan) {
    if (!prog || !plan)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    if (!prog->io_tensors) {
        prog->io_tensors = (MeTensor *)me_alloc(prog->evalue_count * sizeof(MeTensor));
        if (!prog->io_tensors)
            return ME_STATUS_ERROR_OUT_OF_MEMORY;
        memset(prog->io_tensors, 0, prog->evalue_count * sizeof(MeTensor));
    }
    if (!prog->io_tensor_owned) {
        prog->io_tensor_owned = (bool *)me_alloc(prog->evalue_count * sizeof(bool));
        if (!prog->io_tensor_owned)
            return ME_STATUS_ERROR_OUT_OF_MEMORY;
        memset(prog->io_tensor_owned, 0, prog->evalue_count * sizeof(bool));
    }
    prog->io_tensor_count = prog->evalue_count;

    for (uint32_t i = 0; i < prog->evalue_count; ++i) {
        const EValue *ev = &prog->evalue_pool[i];
        if (ev->type != EVALUE_TYPE_TENSOR)
            continue;
        if (ev->payload >= prog->tensor_count)
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        if (evalue_is_plan_input(plan, prog->int_pool, prog->int_count, i))
            continue;
        if (prog->io_tensors[i])
            continue;

        const TensorMeta *meta     = &prog->tensor_pool[ev->payload];
        void             *data_ptr = NULL;
        if (meta->buffer_id == 0) {
            size_t nbytes = tensor_nbytes_from_meta(meta, prog->int_pool, prog->int_count);
            if (nbytes == 0)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if (meta->data_offset == UINT32_MAX)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if (meta->data_offset > prog->weight_size || nbytes > (size_t)(prog->weight_size - meta->data_offset))
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            data_ptr = (void *)(prog->weight_data + meta->data_offset);
        }

        MeStatus s = create_view_tensor(prog, meta, data_ptr, &prog->io_tensors[i]);
        if (s != ME_STATUS_OK)
            return s;
        prog->io_tensor_owned[i] = true;
    }

    return ME_STATUS_OK;
}

static MeStatus prepare_runtime_buffers(MeProgram prog, const ExecutionPlanData *plan) {
    if (!prog || !plan)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (plan->memory_pool_size > 0) {
        if (!prog->exec_mem.base) {
            MeStatus s = MeArena_Init(&prog->exec_mem, (size_t)plan->memory_pool_size);
            if (s != ME_STATUS_OK)
                return s;
        } else if (prog->exec_mem.capacity < (size_t)plan->memory_pool_size) {
            MeArena_Destroy(&prog->exec_mem);
            MeStatus s = MeArena_Init(&prog->exec_mem, (size_t)plan->memory_pool_size);
            if (s != ME_STATUS_OK)
                return s;
        } else {
            MeArena_Reset(&prog->exec_mem);
        }
    }

    for (uint32_t i = 0; i < prog->evalue_count; ++i) {
        if (!prog->io_tensor_owned[i] || !prog->io_tensors[i])
            continue;
        const EValue *ev = &prog->evalue_pool[i];
        if (ev->type != EVALUE_TYPE_TENSOR || ev->payload >= prog->tensor_count)
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        const TensorMeta *meta = &prog->tensor_pool[ev->payload];
        if (meta->buffer_id == 0)
            continue;
        if (meta->data_offset == UINT32_MAX) {
            if (prog->io_tensors[i]->owns_data && prog->io_tensors[i]->data) {
                me_free(prog->io_tensors[i]->data);
            }
            prog->io_tensors[i]->data = me_alloc_aligned(prog->io_tensors[i]->nbytes, 16);
            if (!prog->io_tensors[i]->data)
                return ME_STATUS_ERROR_OUT_OF_MEMORY;
            prog->io_tensors[i]->owns_data = true;
            continue;
        }
        MeStatus s = ensure_tensor_storage(prog, meta, prog->io_tensors[i]);
        if (s != ME_STATUS_OK)
            return s;
    }
    return ME_STATUS_OK;
}

MeStatus pMeProgram_RunPlan(MeProgram prog, uint32_t plan_idx) {
    if (!prog || plan_idx >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    const ExecutionPlanData *plan = &prog->plan_pool[plan_idx];
    if (plan->instructions_offset > prog->instruction_count ||
        plan->instructions_count > prog->instruction_count - plan->instructions_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (plan->inputs_offset > prog->int_count || plan->inputs_count > prog->int_count - plan->inputs_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (plan->outputs_offset > prog->int_count || plan->outputs_count > prog->int_count - plan->outputs_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    MeStatus s = ensure_tensor_views(prog, plan);
    if (s != ME_STATUS_OK)
        return s;

    for (uint32_t i = 0; i < plan->inputs_count; ++i) {
        uint32_t eidx = (uint32_t)prog->int_pool[plan->inputs_offset + i];
        if (eidx >= prog->evalue_count)
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        if (!prog->bound_inputs || !prog->bound_inputs[i])
            return ME_STATUS_ERROR_INVALID_ARGUMENT;
        prog->io_tensors[eidx] = prog->bound_inputs[i];
    }

    int32_t  ref_dims[8];
    uint32_t ref_ndim = 0;
    if (plan->inputs_count > 0 && prog->bound_inputs && prog->bound_inputs[0]) {
        ref_ndim = 8;
        if (get_tensor_dims(prog->bound_inputs[0], ref_dims, &ref_ndim) != ME_STATUS_OK)
            return ME_STATUS_ERROR_SHAPE_MISMATCH;
    }
    for (uint32_t i = 0; i < prog->evalue_count; ++i) {
        if (!prog->io_tensor_owned[i] || !prog->io_tensors[i])
            continue;
        const EValue *ev = &prog->evalue_pool[i];
        if (ev->type != EVALUE_TYPE_TENSOR || ev->payload >= prog->tensor_count)
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        const TensorMeta *meta = &prog->tensor_pool[ev->payload];
        s                      = apply_runtime_shape_to_tensor(prog, prog->io_tensors[i], meta, ref_dims, ref_ndim);
        if (s != ME_STATUS_OK)
            return s;
    }

    s = prepare_runtime_buffers(prog, plan);
    if (s != ME_STATUS_OK)
        return s;

    MeAllocator *a = MeMemory_GetAllocator();
    for (uint32_t pc = 0; pc < plan->instructions_count; ++pc) {
        const Instruction *instr = &prog->instruction_pool[plan->instructions_offset + pc];
        switch ((Opcode)instr->opcode) {
        case OPCODE_KERNEL_CALL: {
            if (instr->arg1 >= prog->operator_count)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if (instr->arg2 > prog->int_count || instr->arg3 > prog->int_count - instr->arg2)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if ((uint32_t)instr->input_count + (uint32_t)instr->output_count > instr->arg3)
                return ME_STATUS_ERROR_INVALID_PROGRAM;

            MeKernelFunc kernel = prog->resolved_kernels[instr->arg1];
            if (!kernel)
                return ME_STATUS_ERROR_OPERATOR_NOT_FOUND;

            MeTensor *ins  = NULL;
            MeTensor *outs = NULL;
            if (instr->input_count > 0) {
                ins = (MeTensor *)me_alloc(instr->input_count * sizeof(MeTensor));
                if (!ins)
                    return ME_STATUS_ERROR_OUT_OF_MEMORY;
            }
            if (instr->output_count > 0) {
                outs = (MeTensor *)me_alloc(instr->output_count * sizeof(MeTensor));
                if (!outs) {
                    me_free(ins);
                    return ME_STATUS_ERROR_OUT_OF_MEMORY;
                }
            }

            for (uint32_t i = 0; i < instr->input_count; ++i) {
                uint32_t eidx = (uint32_t)prog->int_pool[instr->arg2 + i];
                if (eidx >= prog->evalue_count || !prog->io_tensors[eidx]) {
                    me_free(outs);
                    me_free(ins);
                    return ME_STATUS_ERROR_EXECUTION_FAILED;
                }
                ins[i] = prog->io_tensors[eidx];
            }
            for (uint32_t i = 0; i < instr->output_count; ++i) {
                uint32_t eidx = (uint32_t)prog->int_pool[instr->arg2 + instr->input_count + i];
                if (eidx >= prog->evalue_count || !prog->io_tensors[eidx]) {
                    me_free(outs);
                    me_free(ins);
                    return ME_STATUS_ERROR_EXECUTION_FAILED;
                }
                outs[i] = prog->io_tensors[eidx];
            }

            MeOpContext ctx;
            ctx.inputs       = ins;
            ctx.input_count  = instr->input_count;
            ctx.outputs      = outs;
            ctx.output_count = instr->output_count;
            ctx.allocator    = a;

            MeStatus ks = kernel(&ctx);
            me_free(outs);
            me_free(ins);
            if (ks != ME_STATUS_OK)
                return ks;
            break;
        }
        case OPCODE_DELEGATE_CALL:
            return ME_STATUS_ERROR_UNSUPPORTED;
        case OPCODE_MOVE_CALL:
            if (instr->arg1 >= prog->evalue_count || instr->arg2 >= prog->evalue_count)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->io_tensors[instr->arg2] = prog->io_tensors[instr->arg1];
            break;
        case OPCODE_JUMP_FALSE_CALL:
            if (instr->arg1 >= prog->evalue_count)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if (instr->arg2 >= plan->instructions_count)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            if (prog->evalue_pool[instr->arg1].type == EVALUE_TYPE_BOOL &&
                prog->evalue_pool[instr->arg1].payload == 0) {
                pc = instr->arg2 - 1;
            }
            break;
        case OPCODE_FREE_CALL:
        case OPCODE_NOP_CALL:
            break;
        default:
            return ME_STATUS_ERROR_INVALID_PROGRAM;
        }
    }

    if (prog->bound_outputs) {
        for (uint32_t i = 0; i < plan->outputs_count; ++i) {
            uint32_t eidx = (uint32_t)prog->int_pool[plan->outputs_offset + i];
            if (eidx >= prog->evalue_count)
                return ME_STATUS_ERROR_INVALID_PROGRAM;
            prog->bound_outputs[i] = prog->io_tensors[eidx];
        }
    }
    return ME_STATUS_OK;
}

MeStatus MeProgram_SetInput(MeProgram prog, uint32_t index, MeTensor tensor) {
    if (!prog || !tensor)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = prog->header ? prog->header->entry_plan_idx : 0;
    if (!prog->plan_pool || entry >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    const ExecutionPlanData *plan = &prog->plan_pool[entry];
    if (index >= plan->inputs_count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (!prog->bound_inputs)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    uint32_t eidx = (uint32_t)prog->int_pool[plan->inputs_offset + index];
    if (eidx >= prog->evalue_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    const EValue *ev = &prog->evalue_pool[eidx];
    if (ev->type != EVALUE_TYPE_TENSOR || ev->payload >= prog->tensor_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    const TensorMeta *meta = &prog->tensor_pool[ev->payload];

    if (MeTensor_GetDtype(tensor) != (MeScalarType)meta->scalar_type)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    uint32_t in_ndim = 8;
    int32_t  in_dims[8];
    if (get_tensor_dims(tensor, in_dims, &in_ndim) != ME_STATUS_OK)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    if (in_ndim != meta->ndim)
        return ME_STATUS_ERROR_SHAPE_MISMATCH;
    if (meta->shape_offset > prog->int_count || meta->ndim > prog->int_count - meta->shape_offset)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    for (uint32_t i = 0; i < meta->ndim; ++i) {
        int32_t md = prog->int_pool[meta->shape_offset + i];
        if (md > 0 && md != in_dims[i])
            return ME_STATUS_ERROR_SHAPE_MISMATCH;
    }

    prog->bound_inputs[index] = tensor;
    return ME_STATUS_OK;
}

MeStatus MeProgram_Execute(MeProgram prog) {
    if (!prog)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = 0;
    if (prog->header)
        entry = prog->header->entry_plan_idx;

    return pMeProgram_RunPlan(prog, entry);
}

MeStatus MeProgram_GetOutput(MeProgram prog, uint32_t index, MeTensor *out) {
    if (!prog || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    uint32_t entry = prog->header ? prog->header->entry_plan_idx : 0;
    if (!prog->plan_pool || entry >= prog->plan_count)
        return ME_STATUS_ERROR_INVALID_PROGRAM;
    if (index >= prog->plan_pool[entry].outputs_count)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;
    if (!prog->bound_outputs || !prog->bound_outputs[index])
        return ME_STATUS_ERROR_EXECUTION_FAILED;
    *out = prog->bound_outputs[index];
    return ME_STATUS_OK;
}
