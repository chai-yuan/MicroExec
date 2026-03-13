/**
 * Example custom ReLU operator.
 *
 * This demonstrates how a user writes an optimised kernel and
 * overrides the built-in soft implementation.
 */
#ifndef MY_RELU_H
#define MY_RELU_H

#include "me_operator.h"

MeStatus my_fast_relu(MeOpContext *ctx);

#endif /* MY_RELU_H */
