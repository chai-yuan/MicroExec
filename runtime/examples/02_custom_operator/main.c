/**
 * Example 02 — Custom Operator Override
 *
 * Shows how to replace a built-in soft operator with a user-supplied
 * kernel at runtime.  The custom ReLU in my_relu.c simply prints a
 * log line so you can verify it is being called.
 */
#include <stdio.h>
#include <stdlib.h>

#include "microexec.h"
#include "my_relu.h"

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

    /* Override the built-in ReLU with our custom version */
    s = MeRuntime_Register("Relu", my_fast_relu);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeRuntime_Register: %s\n", MeStatus_String(s));
        MeRuntime_Shutdown();
        return 1;
    }
    printf("Registered custom ReLU operator.\n");

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
