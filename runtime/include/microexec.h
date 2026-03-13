/**
 * @file microexec.h
 * @brief MicroExec 运行时公共 API 总头文件。
 *
 * 包含此单一头文件即可访问完整的运行时 API。
 *
 * 典型用法：
 *
 *     MeRuntime rt;
 *     me_runtime_create(NULL, &rt);
 *
 *     MeProgram prog;
 *     me_program_load_file(rt, "model.mvmp", &prog);
 *
 *     MeTensor input;
 *     int32_t shape[] = {1, 1, 28, 28};
 *     me_tensor_create(rt, ME_SCALAR_FLOAT32, shape, 4, &input);
 *     // ... 填充输入数据 ...
 *
 *     me_program_set_input(prog, 0, input);
 *     me_program_execute(prog);
 *
 *     MeTensor output;
 *     me_program_get_output(prog, 0, &output);
 *     // ... 读取输出数据 ...
 *
 *     me_tensor_destroy(input);
 *     me_program_destroy(prog);
 *     me_runtime_destroy(rt);
 */
#ifndef MICROEXEC_MICROEXEC_H
#define MICROEXEC_MICROEXEC_H

#include "me_operator.h"
#include "me_status.h"
#include "me_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==== Runtime Lifecycle ================================================ */

/** 创建运行时实例（传入 NULL 使用默认配置） */
MeStatus me_runtime_create(const MeRuntimeConfig *config, MeRuntime *out);
/** 销毁运行时并释放所有关联资源 */
void me_runtime_destroy(MeRuntime rt);

/* ==== Program Loading ================================================== */

/** 从内存缓冲区加载编译后的程序 */
MeStatus me_program_load(MeRuntime rt, const void *data, uint32_t size, MeProgram *out);
/** 从文件路径加载编译后的程序 */
MeStatus me_program_load_file(MeRuntime rt, const char *path, MeProgram *out);
/** 销毁已加载的程序 */
void me_program_destroy(MeProgram prog);
/** 查询程序期望的输入张量数量 */
MeStatus me_program_input_count(MeProgram prog, uint32_t *count);
/** 查询程序产生的输出张量数量 */
MeStatus me_program_output_count(MeProgram prog, uint32_t *count);

/* ==== Tensor Management ================================================ */

/** 创建张量（根据形状自动分配存储空间，内容初始化为零） */
MeStatus me_tensor_create(MeRuntime rt, MeScalarType dtype, const int32_t *shape, uint32_t ndim, MeTensor *out);
/** 销毁用户创建的张量 */
void me_tensor_destroy(MeTensor tensor);
/** 向张量拷贝数据（size 必须等于 me_tensor_nbytes） */
MeStatus me_tensor_set_data(MeTensor tensor, const void *src, size_t size);
/** 获取张量数据缓冲区的可写指针 */
void *me_tensor_data(MeTensor tensor);
/** 查询张量形状 */
MeStatus me_tensor_shape(MeTensor tensor, int32_t *shape_out, uint32_t *ndim_out);
/** 获取张量的标量类型 */
MeScalarType me_tensor_dtype(MeTensor tensor);
/** 获取张量数据缓冲区的总字节大小 */
size_t me_tensor_nbytes(MeTensor tensor);

/* ==== Execution ======================================================== */

/** 绑定输入张量到程序 */
MeStatus me_program_set_input(MeProgram prog, uint32_t index, MeTensor tensor);
/** 执行程序（运行默认计划中的所有指令） */
MeStatus me_program_execute(MeProgram prog);
/** 获取输出张量的借用引用（由程序拥有，无需销毁） */
MeStatus me_program_get_output(MeProgram prog, uint32_t index, MeTensor *out);

/* ==== Utility ========================================================== */

/** 获取运行时库版本字符串 */
const char *me_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* MICROEXEC_MICROEXEC_H */
