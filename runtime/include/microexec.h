/**
 * @file microexec.h
 * @brief MicroExec 运行时公共 API 总头文件。
 *
 * 包含此单一头文件即可访问完整的运行时 API。
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

/** 初始化全局运行时单例（传入 NULL 使用默认配置） */
MeStatus MeRuntime_Init(const MeRuntimeConfig *config);
/** 关闭运行时并释放所有关联资源 */
void MeRuntime_Shutdown(void);

/* ==== Program Management ================================================== */

/** 从内存缓冲区加载编译后的程序 */
MeStatus MeProgram_CreateFromBuffer(const void *data, uint32_t size, MeProgram *out);
/** 销毁已加载的程序 */
void MeProgram_Destroy(MeProgram prog);
/** 查询程序期望的输入张量数量 */
MeStatus MeProgram_InputCount(MeProgram prog, uint32_t *count);
/** 查询程序产生的输出张量数量 */
MeStatus MeProgram_OutputCount(MeProgram prog, uint32_t *count);
/** 绑定输入张量到程序 */
MeStatus MeProgram_SetInput(MeProgram prog, uint32_t index, MeTensor tensor);
/** 执行程序（运行默认计划中的所有指令） */
MeStatus MeProgram_Execute(MeProgram prog);
/** 获取输出张量的借用引用（由程序拥有，无需销毁） */
MeStatus MeProgram_GetOutput(MeProgram prog, uint32_t index, MeTensor *out);

/* ==== Tensor Management ================================================ */

/** 创建张量（根据形状自动分配存储空间，内容初始化为零） */
MeStatus MeTensor_Create(MeScalarType dtype, const int32_t *shape, uint32_t ndim, MeTensor *out);
/** 销毁用户创建的张量 */
void MeTensor_Destroy(MeTensor tensor);
/** 向张量拷贝数据（size 必须等于 MeTensor_GetNbytes） */
MeStatus MeTensor_SetData(MeTensor tensor, const void *src, size_t size);
/** 获取张量数据缓冲区的可写指针 */
void *MeTensor_GetData(MeTensor tensor);
/** 查询张量形状 */
MeStatus MeTensor_GetShape(MeTensor tensor, int32_t *shape_out, uint32_t *ndim_out);
/** 获取张量的标量类型 */
MeScalarType MeTensor_GetDtype(MeTensor tensor);
/** 获取张量数据缓冲区的总字节大小 */
size_t MeTensor_GetNbytes(MeTensor tensor);

/* ==== Utility ========================================================== */

/** 获取运行时库版本字符串 */
const char *Microexec_Version(void);

#ifdef __cplusplus
}
#endif

#endif /* MICROEXEC_MICROEXEC_H */
