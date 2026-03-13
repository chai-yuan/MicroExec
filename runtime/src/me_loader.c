#include "me_internal.h"

#include <stdio.h>
#include <string.h>

/* ---- Internal: Binary Parser ------------------------------------------ */

MeStatus me_loader_parse(MeProgram prog, const void *data, uint32_t size) {
    if (size < sizeof(VMFileHeader))
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    const VMFileHeader *hdr = (const VMFileHeader *)data;
    if (hdr->magic != kVMFileMagic)
        return ME_STATUS_ERROR_INVALID_PROGRAM;

    /* TODO: validate version, section table bounds, per-section offsets */
    prog->header        = hdr;
    prog->section_count = hdr->section_count;
    prog->sections      = (const VMSectionDesc *)((const uint8_t *)data +
                                                   hdr->section_table_ofs);

    /* TODO: walk section table, assign pool pointers and counts */

    return ME_STATUS_OK;
}

MeStatus me_loader_resolve_kernels(MeProgram prog) {
    if (prog->operator_count == 0)
        return ME_STATUS_OK;

    MeAllocator *a = &prog->runtime->allocator;
    prog->resolved_kernels = (MeKernelFunc *)me_alloc(
        a, prog->operator_count * sizeof(MeKernelFunc));
    if (!prog->resolved_kernels) return ME_STATUS_ERROR_OUT_OF_MEMORY;

    /* TODO: for each OperatorDef, look up name in string pool, then look up
       kernel in registry, store pointer in resolved_kernels[] */

    return ME_STATUS_OK;
}

/* ---- Public: Program Loading ------------------------------------------ */

MeStatus me_program_load(MeRuntime rt, const void *data, uint32_t size,
                         MeProgram *out) {
    if (!rt || !data || !size || !out)
        return ME_STATUS_ERROR_INVALID_ARGUMENT;

    MeAllocator *a = &rt->allocator;

    MeProgram prog = (MeProgram)me_alloc(a, sizeof(struct MeProgram_T));
    if (!prog) return ME_STATUS_ERROR_OUT_OF_MEMORY;
    memset(prog, 0, sizeof(*prog));
    prog->runtime = rt;

    prog->raw_data = me_alloc(a, size);
    if (!prog->raw_data) { me_free(a, prog); return ME_STATUS_ERROR_OUT_OF_MEMORY; }
    memcpy(prog->raw_data, data, size);
    prog->raw_size  = size;
    prog->owns_data = true;

    MeStatus s = me_loader_parse(prog, prog->raw_data, prog->raw_size);
    if (s != ME_STATUS_OK) { me_program_destroy(prog); return s; }

    s = me_loader_resolve_kernels(prog);
    if (s != ME_STATUS_OK) { me_program_destroy(prog); return s; }

    *out = prog;
    return ME_STATUS_OK;
}

MeStatus me_program_load_file(MeRuntime rt, const char *path,
                              MeProgram *out) {
    if (!rt || !path || !out) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    FILE *f = fopen(path, "rb");
    if (!f) return ME_STATUS_ERROR_IO;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return ME_STATUS_ERROR_INVALID_PROGRAM; }

    MeAllocator *a = &rt->allocator;
    void *buf = me_alloc(a, (size_t)len);
    if (!buf) { fclose(f); return ME_STATUS_ERROR_OUT_OF_MEMORY; }

    size_t read_sz = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (read_sz != (size_t)len) { me_free(a, buf); return ME_STATUS_ERROR_IO; }

    MeStatus s = me_program_load(rt, buf, (uint32_t)len, out);
    me_free(a, buf);
    return s;
}

void me_program_destroy(MeProgram prog) {
    if (!prog) return;
    MeAllocator *a = &prog->runtime->allocator;

    me_arena_destroy(&prog->exec_mem);

    if (prog->resolved_kernels) me_free(a, prog->resolved_kernels);
    if (prog->io_tensors)       me_free(a, prog->io_tensors);
    if (prog->bound_inputs)     me_free(a, prog->bound_inputs);
    if (prog->bound_outputs)    me_free(a, prog->bound_outputs);

    if (prog->owns_data && prog->raw_data) me_free(a, prog->raw_data);

    me_free(a, prog);
}

MeStatus me_program_input_count(MeProgram prog, uint32_t *count) {
    if (!prog || !count) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: read from plan_pool[entry_plan_idx].inputs_count */
    *count = prog->bound_input_count;
    return ME_STATUS_OK;
}

MeStatus me_program_output_count(MeProgram prog, uint32_t *count) {
    if (!prog || !count) return ME_STATUS_ERROR_INVALID_ARGUMENT;

    /* TODO: read from plan_pool[entry_plan_idx].outputs_count */
    *count = prog->bound_output_count;
    return ME_STATUS_OK;
}
