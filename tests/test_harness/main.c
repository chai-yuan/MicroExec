/**
 * MicroExec Test Harness
 *
 * Usage:
 *   ./test_harness.elf <model.mvmp> <work_dir> <num_inputs> \
 *       <dtype0> <ndim0> <d0> <d1> ... \
 *       [<dtype1> <ndim1> <d0> <d1> ...] ...
 *
 * Reads input tensors from <work_dir>/input_0.bin, input_1.bin, ...
 * Writes output tensors to <work_dir>/output_0.bin, output_1.bin, ...
 * Prints output metadata to stdout, one line per output:
 *   <dtype_int> <ndim> <d0> <d1> ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "microexec.h"

#define MAX_DIMS 8
#define MAX_INPUTS 16
#define MAX_OUTPUTS 16

static int read_file(const char *path, uint8_t **out_data, uint32_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[harness] failed to open: %s\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) {
        fclose(fp);
        return -1;
    }
    uint8_t *data = (uint8_t *)malloc(size);
    if (!data) {
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out_data = data;
    *out_size = (uint32_t)size;
    return 0;
}

static int write_file(const char *path, const void *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[harness] failed to create: %s\n", path);
        return -1;
    }
    if (fwrite(data, 1, size, fp) != size) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <model.mvmp> <work_dir> <num_inputs> "
                "[<dtype> <ndim> <dims...>] ...\n",
                argv[0]);
        return 1;
    }

    const char *mvmp_path = argv[1];
    const char *work_dir  = argv[2];
    int         num_inputs = atoi(argv[3]);

    if (num_inputs <= 0 || num_inputs > MAX_INPUTS) {
        fprintf(stderr, "[harness] invalid num_inputs: %d\n", num_inputs);
        return 1;
    }

    /* Parse per-input specs from remaining args */
    MeScalarType input_dtypes[MAX_INPUTS];
    int32_t      input_shapes[MAX_INPUTS][MAX_DIMS];
    uint32_t     input_ndims[MAX_INPUTS];

    int arg_idx = 4;
    for (int i = 0; i < num_inputs; i++) {
        if (arg_idx >= argc) {
            fprintf(stderr, "[harness] not enough args for input %d\n", i);
            return 1;
        }
        input_dtypes[i] = (MeScalarType)atoi(argv[arg_idx++]);

        if (arg_idx >= argc) {
            fprintf(stderr, "[harness] missing ndim for input %d\n", i);
            return 1;
        }
        input_ndims[i] = (uint32_t)atoi(argv[arg_idx++]);

        if (input_ndims[i] > MAX_DIMS) {
            fprintf(stderr, "[harness] ndim too large for input %d\n", i);
            return 1;
        }
        for (uint32_t d = 0; d < input_ndims[i]; d++) {
            if (arg_idx >= argc) {
                fprintf(stderr, "[harness] missing dim %u for input %d\n", d, i);
                return 1;
            }
            input_shapes[i][d] = atoi(argv[arg_idx++]);
        }
    }

    MeStatus  s;
    MeProgram prog = NULL;
    MeTensor  inputs[MAX_INPUTS];
    memset(inputs, 0, sizeof(inputs));

    /* Init runtime */
    s = MeRuntime_Init(NULL);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "[harness] MeRuntime_Init: %s\n", MeStatus_String(s));
        return 1;
    }

    /* Load model */
    uint8_t *model_data = NULL;
    uint32_t model_size = 0;
    if (read_file(mvmp_path, &model_data, &model_size) != 0) {
        fprintf(stderr, "[harness] failed to read model: %s\n", mvmp_path);
        goto fail;
    }

    s = MeProgram_CreateFromBuffer(model_data, model_size, &prog);
    free(model_data);
    model_data = NULL;
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "[harness] MeProgram_CreateFromBuffer: %s\n",
                MeStatus_String(s));
        goto fail;
    }

    /* Create and bind input tensors */
    for (int i = 0; i < num_inputs; i++) {
        s = MeTensor_Create(input_dtypes[i], input_shapes[i], input_ndims[i],
                            &inputs[i]);
        if (s != ME_STATUS_OK) {
            fprintf(stderr, "[harness] MeTensor_Create(input %d): %s\n", i,
                    MeStatus_String(s));
            goto fail;
        }

        char path_buf[512];
        snprintf(path_buf, sizeof(path_buf), "%s/input_%d.bin", work_dir, i);

        uint8_t *input_data = NULL;
        uint32_t input_size = 0;
        if (read_file(path_buf, &input_data, &input_size) != 0) {
            fprintf(stderr, "[harness] failed to read: %s\n", path_buf);
            goto fail;
        }

        size_t expected = MeTensor_GetNbytes(inputs[i]);
        if (input_size != expected) {
            fprintf(stderr,
                    "[harness] input %d size mismatch: file=%u expected=%zu\n",
                    i, input_size, expected);
            free(input_data);
            goto fail;
        }

        s = MeTensor_SetData(inputs[i], input_data, input_size);
        free(input_data);
        if (s != ME_STATUS_OK) {
            fprintf(stderr, "[harness] MeTensor_SetData(input %d): %s\n", i,
                    MeStatus_String(s));
            goto fail;
        }

        s = MeProgram_SetInput(prog, (uint32_t)i, inputs[i]);
        if (s != ME_STATUS_OK) {
            fprintf(stderr, "[harness] MeProgram_SetInput(%d): %s\n", i,
                    MeStatus_String(s));
            goto fail;
        }
    }

    /* Execute inference */
    s = MeProgram_Execute(prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "[harness] MeProgram_Execute: %s\n", MeStatus_String(s));
        goto fail;
    }

    /* Retrieve and save outputs */
    uint32_t num_outputs = 0;
    s = MeProgram_OutputCount(prog, &num_outputs);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "[harness] MeProgram_OutputCount: %s\n",
                MeStatus_String(s));
        goto fail;
    }

    for (uint32_t i = 0; i < num_outputs && i < MAX_OUTPUTS; i++) {
        MeTensor output = NULL;
        s = MeProgram_GetOutput(prog, i, &output);
        if (s != ME_STATUS_OK) {
            fprintf(stderr, "[harness] MeProgram_GetOutput(%u): %s\n", i,
                    MeStatus_String(s));
            goto fail;
        }

        const void *out_ptr  = MeTensor_GetData(output);
        size_t      out_size = MeTensor_GetNbytes(output);

        char path_buf[512];
        snprintf(path_buf, sizeof(path_buf), "%s/output_%u.bin", work_dir, i);
        if (write_file(path_buf, out_ptr, out_size) != 0)
            goto fail;

        int32_t  shape[MAX_DIMS];
        uint32_t ndim = MAX_DIMS;
        MeTensor_GetShape(output, shape, &ndim);
        MeScalarType dtype = MeTensor_GetDtype(output);

        printf("%d %u", (int)dtype, ndim);
        for (uint32_t d = 0; d < ndim; d++)
            printf(" %d", shape[d]);
        printf("\n");
    }

    /* Cleanup */
    for (int i = 0; i < num_inputs; i++)
        MeTensor_Destroy(inputs[i]);
    MeProgram_Destroy(prog);
    MeRuntime_Shutdown();
    return 0;

fail:
    for (int i = 0; i < num_inputs; i++)
        if (inputs[i])
            MeTensor_Destroy(inputs[i]);
    if (prog)
        MeProgram_Destroy(prog);
    MeRuntime_Shutdown();
    return 1;
}
