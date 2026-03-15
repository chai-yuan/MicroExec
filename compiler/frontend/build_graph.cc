#include <fstream>

#include "common/log.h"
#include "graph/graph.h"
#include "onnx.pb.h"

#include <cstring>
#include <unordered_map>

// ONNX类型转换为内部DataType
static DataType OnnxTypeToDataType(int32_t onnx_type) {
    switch (onnx_type) {
    case 1:
        return DataType::FLOAT32; // onnx::TensorProto::FLOAT
    case 2:
        return DataType::UINT8; // onnx::TensorProto::UINT8
    case 3:
        return DataType::INT8; // onnx::TensorProto::INT8
    case 6:
        return DataType::INT32; // onnx::TensorProto::INT32
    case 7:
        return DataType::INT64; // onnx::TensorProto::INT64
    case 8:
        return DataType::STRING; // onnx::TensorProto::STRING
    case 9:
        return DataType::BOOL; // onnx::TensorProto::BOOL
    case 10:
        return DataType::FLOAT16; // onnx::TensorProto::FLOAT16
    default:
        return DataType::UNKNOWN;
    }
}

// 将ONNX ValueInfo的shape和type信息合并到Edge中
// max_dynamic_size: 动态维度的最大值上界（正数），动态维度将存储为 -max_dynamic_size
static void MergeShapeAndType(const onnx::ValueInfoProto &value_info, Edge *edge, int64_t max_dynamic_size) {
    if (edge == nullptr) {
        return;
    }
    if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
        return;
    }

    const auto &tensor_type = value_info.type().tensor_type();

    if (!edge->has_dtype && tensor_type.elem_type() != 0) {
        edge->dtype     = OnnxTypeToDataType(tensor_type.elem_type());
        edge->has_dtype = (edge->dtype != DataType::UNKNOWN);
    }

    if (!edge->has_shape && tensor_type.has_shape()) {
        edge->shape.clear();
        for (int i = 0; i < tensor_type.shape().dim_size(); ++i) {
            const auto &dim = tensor_type.shape().dim(i);
            if (dim.has_dim_value()) {
                edge->shape.push_back(dim.dim_value());
            } else {
                // 动态维度：-x 表示最大值为 x 的有界动态维度
                edge->shape.push_back(-max_dynamic_size);
            }
        }
        edge->has_shape = true;
    }
}

// 加载ONNX Tensor的原始数据到内存
static bool LoadTensorRawData(const onnx::TensorProto &tensor, std::vector<uint8_t> *out_data) {
    if (out_data == nullptr) {
        return false;
    }

    if (!tensor.raw_data().empty()) {
        out_data->resize(tensor.raw_data().size());
        std::memcpy(out_data->data(), tensor.raw_data().data(), tensor.raw_data().size());
        return true;
    }

    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT: {
        const size_t byte_size = static_cast<size_t>(tensor.float_data_size()) * sizeof(float);
        out_data->resize(byte_size);
        if (byte_size > 0) {
            std::memcpy(out_data->data(), tensor.float_data().data(), byte_size);
        }
        return true;
    }
    case onnx::TensorProto::INT32: {
        const size_t byte_size = static_cast<size_t>(tensor.int32_data_size()) * sizeof(int32_t);
        out_data->resize(byte_size);
        if (byte_size > 0) {
            std::memcpy(out_data->data(), tensor.int32_data().data(), byte_size);
        }
        return true;
    }
    case onnx::TensorProto::INT64: {
        const size_t byte_size = static_cast<size_t>(tensor.int64_data_size()) * sizeof(int64_t);
        out_data->resize(byte_size);
        if (byte_size > 0) {
            std::memcpy(out_data->data(), tensor.int64_data().data(), byte_size);
        }
        return true;
    }
    default:
        return false;
    }
}

// 解析ONNX AttributeProto并转换为内部Attribute结构
static Attribute ParseAttribute(const onnx::AttributeProto &onnx_attr) {
    Attribute attr;
    switch (onnx_attr.type()) {
    case onnx::AttributeProto::FLOAT:
        attr.kind  = AttributeKind::FLOAT;
        attr.value = onnx_attr.f();
        break;
    case onnx::AttributeProto::INT:
        attr.kind  = AttributeKind::INT;
        attr.value = onnx_attr.i();
        break;
    case onnx::AttributeProto::STRING:
        attr.kind  = AttributeKind::STRING;
        attr.value = onnx_attr.s();
        break;
    case onnx::AttributeProto::FLOATS: {
        attr.kind = AttributeKind::FLOATS;
        std::vector<float> floats(onnx_attr.floats().begin(), onnx_attr.floats().end());
        attr.value = std::move(floats);
        break;
    }
    case onnx::AttributeProto::INTS: {
        attr.kind = AttributeKind::INTS;
        std::vector<int64_t> ints(onnx_attr.ints().begin(), onnx_attr.ints().end());
        attr.value = std::move(ints);
        break;
    }
    case onnx::AttributeProto::GRAPH:
        attr.kind  = AttributeKind::GRAPH;
        attr.value = nullptr;
        LOG_WARN("Attribute 包含子图(GRAPH)，当前版本暂未实现深度解析。");
        break;
    default:
        LOG_WARN("未处理的 ONNX Attribute 类型: %d", onnx_attr.type());
        break;
    }
    return attr;
}

// 从ONNX模型文件构建内部图IR表示
int Graph::BuildFromONNX(const std::string &file_name) {
    LOG_INFO("使用模型文件 %s", file_name.c_str());

    GOOGLE_PROTOBUF_VERIFY_VERSION;
    onnx::ModelProto model;

    std::fstream input(file_name, std::ios::in | std::ios::binary);
    if (!input) {
        LOG_ERROR("打开 onnx 文件失败: %s", file_name.c_str());
        return -1;
    }
    if (!model.ParseFromIstream(&input)) {
        LOG_ERROR("解析 ONNX 文件失败");
        return -1;
    }
    input.close();

    const onnx::GraphProto &onnx_graph = model.graph();

    // 符号表：映射张量名字到 Edge 指针
    std::unordered_map<std::string, Edge *> edge_symbol_table;

    // Lambda 辅助函数：通过名字获取 Edge，如果不存在则创建（用于确保所有的边都被正确追踪）
    auto get_or_create_edge = [&](const std::string &name) -> Edge * {
        if (name.empty()) {
            return nullptr;
        }
        if (edge_symbol_table.find(name) == edge_symbol_table.end()) {
            edge_symbol_table[name] = this->CreateEdge(name);
        }
        return edge_symbol_table[name];
    };

    // 解析 Initializers
    for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
        const onnx::TensorProto &tensor = onnx_graph.initializer(i);
        LOG_DEBUG("add tensor : %s", tensor.name().c_str());

        Edge *edge = get_or_create_edge(tensor.name());
        if (edge == nullptr) {
            LOG_ERROR("initializer 名称为空，无法导入");
            return -1;
        }

        edge->is_constant = true;
        edge->dtype       = OnnxTypeToDataType(tensor.data_type());
        edge->has_dtype   = (edge->dtype != DataType::UNKNOWN);

        edge->shape.clear();
        for (int j = 0; j < tensor.dims_size(); ++j) {
            edge->shape.push_back(tensor.dims(j));
        }
        edge->has_shape = true;

        if (!LoadTensorRawData(tensor, &edge->weight_data)) {
            LOG_ERROR("暂不支持该 initializer 数据格式: %s", tensor.name().c_str());
            return -1;
        }
    }

    // 解析 Graph Inputs
    for (int i = 0; i < onnx_graph.input_size(); ++i) {
        const onnx::ValueInfoProto &onnx_input = onnx_graph.input(i);
        Edge                       *edge       = get_or_create_edge(onnx_input.name());
        if (edge == nullptr) {
            LOG_ERROR("graph input 名称为空，无法导入");
            return -1;
        }

        MergeShapeAndType(onnx_input, edge, default_max_dynamic_size_);

        // 我们只把非 constant 的 edge 加入到图的逻辑输入中。
        if (!edge->is_constant) {
            edge->is_graph_input = true;
            this->graph_inputs.push_back(edge);
        }
    }

    // 解析 Nodes
    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto &onnx_node = onnx_graph.node(i);
        LOG_DEBUG("add node : %s", onnx_node.name().c_str());

        Node *node   = this->CreateNode(onnx_node.op_type(), onnx_node.name());
        node->domain = onnx_node.domain();

        // 解析 Attributes
        for (int j = 0; j < onnx_node.attribute_size(); ++j) {
            const onnx::AttributeProto &onnx_attr = onnx_node.attribute(j);
            node->attributes[onnx_attr.name()]    = ParseAttribute(onnx_attr);
        }

        // 解析输入边 (Inputs) 并建立双向连接
        for (int j = 0; j < onnx_node.input_size(); ++j) {
            const std::string &input_name = onnx_node.input(j);
            Edge              *input_edge = get_or_create_edge(input_name);
            if (input_edge) {
                node->input_edges.push_back(input_edge);
                input_edge->consumers.push_back(node); // 【关键】谁在消费这个数据
            }
        }

        // 解析输出边 (Outputs) 并建立双向连接
        for (int j = 0; j < onnx_node.output_size(); ++j) {
            const std::string &output_name = onnx_node.output(j);
            Edge              *output_edge = get_or_create_edge(output_name);
            if (output_edge) {
                if (output_edge->producer != nullptr) {
                    LOG_ERROR("检测到重复 producer，tensor `%s` 被多个节点定义", output_name.c_str());
                    return -1;
                }
                node->output_edges.push_back(output_edge);
                output_edge->producer = node;
            }
        }
    }

    // 解析 Graph Outputs
    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const onnx::ValueInfoProto &onnx_output = onnx_graph.output(i);
        Edge                       *edge        = get_or_create_edge(onnx_output.name());
        if (edge == nullptr) {
            LOG_ERROR("graph output 名称为空，无法导入");
            return -1;
        }

        MergeShapeAndType(onnx_output, edge, default_max_dynamic_size_);
        edge->is_graph_output = true;
        this->graph_outputs.push_back(edge);
    }

    // 再遍历一次 onnx_graph.value_info() 来补充中间 Tensor 的 Shape 和类型信息
    for (int i = 0; i < onnx_graph.value_info_size(); ++i) {
        const onnx::ValueInfoProto &value_info = onnx_graph.value_info(i);
        if (edge_symbol_table.find(value_info.name()) != edge_symbol_table.end()) {
            Edge *edge = edge_symbol_table[value_info.name()];
            MergeShapeAndType(value_info, edge, default_max_dynamic_size_);
        }
    }

    if (Validate() != 0) {
        LOG_ERROR("Graph 校验失败");
        return -1;
    }

    DumpSummary();
    LOG_INFO("ONNX 模型解析完成: 包含 %zu 个节点, %zu 条边", nodes.size(), edges.size());
    return 0;
}
