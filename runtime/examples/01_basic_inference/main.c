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

/** 辅助函数：从文件读取整个内容到缓冲区 */
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
        fprintf(stderr, "Invalid file size: %ld\n", size);
        fclose(fp);
        return -1;
    }
    uint8_t *data = (uint8_t *)malloc(size);
    if (!data) {
        fprintf(stderr, "Failed to allocate %ld bytes\n", size);
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, size, fp) != (size_t)size) {
        fprintf(stderr, "Failed to read file\n");
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

    MeRuntime rt     = NULL;
    MeProgram prog   = NULL;
    MeTensor  input  = NULL;
    MeTensor  output = NULL;
    MeStatus  s;
    uint8_t  *model_data = NULL;
    uint32_t  model_size = 0;

    /* 使用默认配置创建运行时环境 */
    s = MeRuntime_Create(NULL, &rt);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeRuntime_Create: %s\n", MeStatus_String(s));
        return 1;
    }
    printf("MicroExec runtime v%s\n", Microexec_Version());

    /* 读取模型文件到缓冲区 */
    if (read_file_to_buffer(argv[1], &model_data, &model_size) != 0) {
        s = ME_STATUS_ERROR_IO;
        goto cleanup;
    }

    /* 加载编译后的模型 */
    s = MeProgram_CreateFromBuffer(rt, model_data, model_size, &prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeProgram_CreateFromBuffer: %s\n", MeStatus_String(s));
        goto cleanup;
    }

    /* 创建输入张量（例如 LeNet: 1×1×28×28 float32） */
    int32_t shape[] = {1, 1, 28, 28};
    s = MeTensor_Create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeTensor_Create: %s\n", MeStatus_String(s));
        goto cleanup;
    }

    /* 使用编译时嵌入的测试输入图像 */
    float *data       = (float *)MeTensor_GetData(input);
    size_t elem_count = MeTensor_GetNbytes(input) / sizeof(float);
    size_t test_count = sizeof(kImageInput) / sizeof(kImageInput[0]);
    if (test_count != elem_count) {
        fprintf(stderr, "Input element count mismatch: model expects %zu, test has %zu\n", elem_count, test_count);
        s = ME_STATUS_ERROR_SHAPE_MISMATCH;
        goto cleanup;
    }
    for (size_t i = 0; i < elem_count; ++i)
        data[i] = kImageInput[i];

    /* 绑定输入并执行推理 */
    s = MeProgram_SetInput(prog, 0, input);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeProgram_SetInput: %s\n", MeStatus_String(s));
        goto cleanup;
    }

    s = MeProgram_Execute(prog);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeProgram_Execute: %s\n", MeStatus_String(s));
        goto cleanup;
    }

    /* 获取并打印输出结果 */
    s = MeProgram_GetOutput(prog, 0, &output);
    if (s != ME_STATUS_OK) {
        fprintf(stderr, "MeProgram_GetOutput: %s\n", MeStatus_String(s));
        goto cleanup;
    }

    printf("Inference complete. Output bytes: %zu\n", MeTensor_GetNbytes(output));
    if (MeTensor_GetDtype(output) == ME_SCALAR_FLOAT32) {
        const float *out_data  = (const float *)MeTensor_GetData(output);
        size_t       out_count = MeTensor_GetNbytes(output) / sizeof(float);
        if (out_count > 0) {
            size_t argmax = 0;
            for (size_t i = 1; i < out_count; ++i) {
                if (out_data[i] > out_data[argmax])
                    argmax = i;
            }
            printf("Predicted class: %zu\n", argmax);
        }
    }

cleanup:
    free(model_data);
    MeTensor_Destroy(input);
    MeProgram_Destroy(prog);
    MeRuntime_Destroy(rt);
    return (s == ME_STATUS_OK) ? 0 : 1;
}
