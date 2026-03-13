/**
 * Example 02 — Custom Operator Override
 *
 * Shows how to replace a built-in soft operator with a user-supplied
 * kernel at runtime.  The custom ReLU in my_relu.c simply prints a
 * log line so you can verify it is being called.
 */
#include <stdio.h>

#include "microexec.h"
#include "my_relu.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mvmp>\n", argv[0]);
        return 1;
    }

    MeRuntime rt = NULL;
    MeStatus  s  = me_runtime_create(NULL, &rt);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "runtime_create: %s\n", me_status_str(s));
        return 1;
    }

    /* Override the built-in ReLU with our custom version */
    s = me_operator_register(rt, "onnx::Relu", my_fast_relu);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "operator_register: %s\n", me_status_str(s));
        me_runtime_destroy(rt);
        return 1;
    }
    printf("Registered custom ReLU operator.\n");

    /* Load model and run inference (same as example 01) */
    MeProgram prog = NULL;
    s = me_program_load_file(rt, argv[1], &prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "program_load: %s\n", me_status_str(s));
        me_runtime_destroy(rt);
        return 1;
    }

    int32_t shape[] = {1, 1, 28, 28};
    MeTensor input = NULL;
    me_tensor_create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);

    me_program_set_input(prog, 0, input);
    s = me_program_execute(prog);
    printf("Execute result: %s\n", me_status_str(s));

    me_tensor_destroy(input);
    me_program_destroy(prog);
    me_runtime_destroy(rt);
    return 0;
}
