/**
 * x86 SIMD-accelerated operator kernels.
 *
 * Demonstrates how to package a set of platform-specific operator
 * overrides and register them in one call.
 */
#ifndef X86_OPS_H
#define X86_OPS_H

#include "microexec.h"

/** Register all x86-accelerated operator overrides. */
MeStatus x86_register_operators(void);

MeStatus x86_relu(MeOpContext *ctx);
MeStatus x86_gemm(MeOpContext *ctx);

#endif /* X86_OPS_H */
