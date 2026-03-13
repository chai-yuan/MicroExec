/**
 * Example 03 — x86 SIMD Acceleration
 *
 * Replaces select built-in operators with SSE2 / AVX2 implementations
 * on x86-64 Linux.  The x86_register_operators() helper bundles all
 * platform-specific overrides in a single call.
 */
#include <stdio.h>

#include "microexec.h"
#include "x86_ops.h"

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

    /* Register all x86-accelerated operators */
    s = x86_register_operators(rt);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "x86_register: %s\n", me_status_str(s));
        me_runtime_destroy(rt);
        return 1;
    }
    printf("x86 SIMD operators registered.\n");

    /* Load model */
    MeProgram prog = NULL;
    s = me_program_load_file(rt, argv[1], &prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "program_load: %s\n", me_status_str(s));
        me_runtime_destroy(rt);
        return 1;
    }

    /* Create dummy input and execute */
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
