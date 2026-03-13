/**
 * Example 01 — Basic Inference
 *
 * Demonstrates the minimal workflow:
 *   1. Create a runtime
 *   2. Load a compiled .mvmp program
 *   3. Create and bind an input tensor
 *   4. Execute inference
 *   5. Read the output
 */
#include <stdio.h>
#include <stdlib.h>

#include "microexec.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mvmp>\n", argv[0]);
        return 1;
    }

    MeRuntime rt   = NULL;
    MeProgram prog = NULL;
    MeTensor  input  = NULL;
    MeTensor  output = NULL;
    MeStatus  s;

    /* 1. Create runtime with default configuration */
    s = me_runtime_create(NULL, &rt);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "runtime_create: %s\n", me_status_str(s));
        return 1;
    }
    printf("MicroExec runtime v%s\n", me_version_string());

    /* 2. Load the compiled model */
    s = me_program_load_file(rt, argv[1], &prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "program_load: %s\n", me_status_str(s));
        goto cleanup;
    }

    /* 3. Create an input tensor (e.g. LeNet: 1×1×28×28 float32) */
    int32_t shape[] = {1, 1, 28, 28};
    s = me_tensor_create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "tensor_create: %s\n", me_status_str(s));
        goto cleanup;
    }

    /* Fill with dummy data (in real use, load actual image data here) */
    float *data = (float *)me_tensor_data(input);
    for (size_t i = 0; i < me_tensor_nbytes(input) / sizeof(float); ++i)
        data[i] = 0.5f;

    /* 4. Bind input and execute */
    s = me_program_set_input(prog, 0, input);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "set_input: %s\n", me_status_str(s));
        goto cleanup;
    }

    s = me_program_execute(prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "execute: %s\n", me_status_str(s));
        goto cleanup;
    }

    /* 5. Retrieve and print output */
    s = me_program_get_output(prog, 0, &output);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "get_output: %s\n", me_status_str(s));
        goto cleanup;
    }

    printf("Inference complete. Output bytes: %zu\n",
           me_tensor_nbytes(output));

cleanup:
    me_tensor_destroy(input);
    me_program_destroy(prog);
    me_runtime_destroy(rt);
    return (s == ME_STATUS_OK) ? 0 : 1;
}
