/**
 * @file soft_operators.h
 * @brief Declarations of all built-in software operator kernels.
 *
 * These portable C implementations serve as the default fallback.
 * Users may override any of them via me_operator_register().
 */
#ifndef SOFT_OPERATORS_H
#define SOFT_OPERATORS_H

#include "microexec.h"

MeStatus me_op_soft_conv(MeOpContext *ctx);
MeStatus me_op_soft_relu(MeOpContext *ctx);
MeStatus me_op_soft_maxpool(MeOpContext *ctx);
MeStatus me_op_soft_gemm(MeOpContext *ctx);
MeStatus me_op_soft_reshape(MeOpContext *ctx);
MeStatus me_op_soft_softmax(MeOpContext *ctx);

#endif /* SOFT_OPERATORS_H */
