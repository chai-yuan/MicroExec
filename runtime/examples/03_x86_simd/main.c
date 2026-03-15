/**
 * Example 03 — x86 SIMD Acceleration
 *
 * Replaces select built-in operators with SSE2 / AVX2 implementations
 * on x86-64 Linux.  The x86_register_operators() helper bundles all
 * platform-specific overrides in a single call.
 */
#include <stdio.h>
#include <stdlib.h>

#include "microexec.h"
#include "x86_ops.h"

static int read_file_to_buffer(const char *path, uint8_t **out_data, uint32_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", path);
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mvmp>\n", argv[0]);
        return 1;
    }

    MeStatus s = MeRuntime_Init(NULL);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeRuntime_Init: %s\n", MeStatus_String(s));
        return 1;
    }

    /* Register all x86-accelerated operators */
    s = x86_register_operators();
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "x86_register: %s\n", MeStatus_String(s));
        MeRuntime_Shutdown();
        return 1;
    }
    printf("x86 SIMD operators registered.\n");

    /* Load model */
    uint8_t *model_data = NULL;
    uint32_t model_size = 0;
    if (read_file_to_buffer(argv[1], &model_data, &model_size) != 0) {
        MeRuntime_Shutdown();
        return 1;
    }

    MeProgram prog = NULL;
    s              = MeProgram_CreateFromBuffer(model_data, model_size, &prog);
    free(model_data);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeProgram_CreateFromBuffer: %s\n", MeStatus_String(s));
        MeRuntime_Shutdown();
        return 1;
    }

    /* Create dummy input and execute */
    int32_t  shape[] = {1, 1, 28, 28};
    MeTensor input   = NULL;
    MeTensor_Create(ME_SCALAR_FLOAT32, shape, 4, &input);

    MeProgram_SetInput(prog, 0, input);
    s = MeProgram_Execute(prog);
    printf("Execute result: %s\n", MeStatus_String(s));

    MeTensor_Destroy(input);
    MeProgram_Destroy(prog);
    MeRuntime_Shutdown();
    return 0;
}
