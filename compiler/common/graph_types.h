#ifndef ONNX_C_GRAPH_TYPES_H
#define ONNX_C_GRAPH_TYPES_H

#include <cstdint>

enum class DataType : uint32_t { UNKNOWN = 0, FLOAT32, FLOAT16, INT64, INT32, INT8, UINT8, BOOL, STRING };

enum class ShapeDynamism : uint32_t {
    STATIC          = 0, // 静态大小，AOT分配
    DYNAMIC_BOUND   = 1, // 有上限的动态大小
    DYNAMIC_UNBOUND = 2  // 完全动态大小
};

enum class AttributeKind : uint32_t { UNDEFINED = 0, INT, FLOAT, STRING, INTS, FLOATS, GRAPH };

#endif // ONNX_C_GRAPH_TYPES_H
