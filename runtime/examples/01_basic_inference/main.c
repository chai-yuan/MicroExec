/**
 * 示例 01 — 基础推理
 *
 * 演示最小化的工作流程：
 *   1. 创建运行时环境
 *   2. 加载编译后的 .mvmp 程序
 *   3. 创建并绑定输入张量
 *   4. 执行推理
 *   5. 读取输出结果
 */
#include <stdio.h>
#include <stdlib.h>

#include "microexec.h"

static const float kImageInput[] = {
#include "image_2_f32.txt"
};

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

    /* 使用默认配置创建运行时环境 */
    s = me_runtime_create(NULL, &rt);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "runtime_create: %s\n", me_status_str(s));
        return 1;
    }
    printf("MicroExec runtime v%s\n", me_version_string());

    /* 加载编译后的模型 */
    s = me_program_load_file(rt, argv[1], &prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "program_load: %s\n", me_status_str(s));
        goto cleanup;
    }

    /* 创建输入张量（例如 LeNet: 1×1×28×28 float32） */
    int32_t shape[] = {1, 1, 28, 28};
    s = me_tensor_create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "tensor_create: %s\n", me_status_str(s));
        goto cleanup;
    }

    /* 使用编译时嵌入的测试输入图像 */
    float *data = (float *)me_tensor_data(input);
    size_t elem_count = me_tensor_nbytes(input) / sizeof(float);
    size_t test_count = sizeof(kImageInput) / sizeof(kImageInput[0]);
    if (test_count != elem_count) {
        fprintf(stderr, "Input element count mismatch: model expects %zu, test has %zu\n",
                elem_count, test_count);
        s = ME_STATUS_ERROR_SHAPE_MISMATCH;
        goto cleanup;
    }
    for (size_t i = 0; i < elem_count; ++i) data[i] = kImageInput[i];

    /* 绑定输入并执行推理 */
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

    /* 获取并打印输出结果 */
    s = me_program_get_output(prog, 0, &output);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "get_output: %s\n", me_status_str(s));
        goto cleanup;
    }

    printf("Inference complete. Output bytes: %zu\n", me_tensor_nbytes(output));
    if (me_tensor_dtype(output) == ME_SCALAR_FLOAT32) {
        const float *out_data = (const float *)me_tensor_data(output);
        size_t out_count = me_tensor_nbytes(output) / sizeof(float);
        if (out_count > 0) {
            size_t argmax = 0;
            for (size_t i = 1; i < out_count; ++i) {
                if (out_data[i] > out_data[argmax]) argmax = i;
            }
            printf("Predicted class: %zu\n", argmax);
        }
    }

cleanup:
    me_tensor_destroy(input);
    me_program_destroy(prog);
    me_runtime_destroy(rt);
    return (s == ME_STATUS_OK) ? 0 : 1;
}
