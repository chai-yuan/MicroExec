#include "graph/graph.h"

#include "common/log.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_map>

static bool GetAttrInt(const Node *node, const char *name, int64_t default_value, int64_t *out_value) {
    if (out_value == nullptr) {
        return false;
    }
    *out_value = default_value;
    auto it    = node->attributes.find(name);
    if (it == node->attributes.end()) {
        return true;
    }
    if (it->second.kind != AttributeKind::INT || !std::holds_alternative<int64_t>(it->second.value)) {
        return false;
    }
    *out_value = std::get<int64_t>(it->second.value);
    return true;
}

static bool GetAttrInts(const Node *node, const char *name, std::vector<int64_t> *out_values) {
    if (out_values == nullptr) {
        return false;
    }
    auto it = node->attributes.find(name);
    if (it == node->attributes.end()) {
        return false;
    }
    if (it->second.kind != AttributeKind::INTS || !std::holds_alternative<std::vector<int64_t>>(it->second.value)) {
        return false;
    }
    *out_values = std::get<std::vector<int64_t>>(it->second.value);
    return true;
}

static bool ReadShapeTensor(const Edge *shape_edge, std::vector<int64_t> *out_dims) {
    if (shape_edge == nullptr || out_dims == nullptr || !shape_edge->is_constant || shape_edge->weight_data.empty()) {
        return false;
    }
    out_dims->clear();

    if (shape_edge->dtype == DataType::INT64) {
        if (shape_edge->weight_data.size() % sizeof(int64_t) != 0) {
            return false;
        }
        const size_t count = shape_edge->weight_data.size() / sizeof(int64_t);
        out_dims->resize(count);
        std::memcpy(out_dims->data(), shape_edge->weight_data.data(), shape_edge->weight_data.size());
        return true;
    }
    if (shape_edge->dtype == DataType::INT32) {
        if (shape_edge->weight_data.size() % sizeof(int32_t) != 0) {
            return false;
        }
        const size_t   count = shape_edge->weight_data.size() / sizeof(int32_t);
        const int32_t *ptr   = reinterpret_cast<const int32_t *>(shape_edge->weight_data.data());
        out_dims->reserve(count);
        for (size_t i = 0; i < count; ++i) {
            out_dims->push_back(static_cast<int64_t>(ptr[i]));
        }
        return true;
    }
    return false;
}

static bool InferConvLikeOutput(const std::vector<int64_t> &input_shape, int64_t out_channels,
                                const std::vector<int64_t> &kernel_spatial_shape, const std::vector<int64_t> &pads,
                                const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations,
                                std::vector<int64_t> *out_shape) {
    if (out_shape == nullptr || input_shape.size() < 3 || kernel_spatial_shape.size() + 2 != input_shape.size() ||
        out_channels <= 0) {
        return false;
    }
    const size_t spatial_rank = input_shape.size() - 2;
    out_shape->assign(input_shape.size(), -1);
    (*out_shape)[0] = input_shape[0];
    (*out_shape)[1] = out_channels;

    for (size_t i = 0; i < spatial_rank; ++i) {
        const int64_t in_dim = input_shape[2 + i];
        const int64_t k_dim  = kernel_spatial_shape[i];
        if (k_dim <= 0) {
            return false;
        }
        const int64_t pad_l    = (i < pads.size()) ? pads[i] : 0;
        const int64_t pad_r    = (i + spatial_rank < pads.size()) ? pads[i + spatial_rank] : pad_l;
        const int64_t stride   = (i < strides.size() && strides[i] > 0) ? strides[i] : 1;
        const int64_t dilation = (i < dilations.size() && dilations[i] > 0) ? dilations[i] : 1;

        if (in_dim <= 0) {
            (*out_shape)[2 + i] = -1;
            continue;
        }

        const int64_t effective_kernel = dilation * (k_dim - 1) + 1;
        const int64_t numerator        = in_dim + pad_l + pad_r - effective_kernel;
        if (numerator < 0) {
            return false;
        }
        (*out_shape)[2 + i] = numerator / stride + 1;
    }
    return true;
}

static bool InferReshapeOutput(const std::vector<int64_t> &input_shape, const std::vector<int64_t> &target_shape_raw,
                               bool allow_zero, std::vector<int64_t> *out_shape) {
    if (out_shape == nullptr || input_shape.empty() || target_shape_raw.empty()) {
        return false;
    }
    bool    input_elems_known = true;
    int64_t input_elems       = 1;
    for (int64_t d : input_shape) {
        if (d <= 0) {
            input_elems_known = false;
            continue;
        }
        input_elems *= d;
    }

    out_shape->clear();
    out_shape->reserve(target_shape_raw.size());
    int64_t known_product = 1;
    int64_t infer_index   = -1;
    for (size_t i = 0; i < target_shape_raw.size(); ++i) {
        int64_t dim = target_shape_raw[i];
        if (dim == 0 && !allow_zero) {
            if (i >= input_shape.size()) {
                return false;
            }
            dim = input_shape[i];
        } else if (dim == -1) {
            if (infer_index != -1) {
                return false;
            }
            infer_index = static_cast<int64_t>(i);
            out_shape->push_back(-1);
            continue;
        } else if (dim <= 0) {
            return false;
        }
        out_shape->push_back(dim);
        if (dim > 0) {
            known_product *= dim;
        }
    }
    if (infer_index >= 0) {
        if (input_elems_known) {
            if (known_product <= 0 || input_elems % known_product != 0) {
                return false;
            }
            (*out_shape)[static_cast<size_t>(infer_index)] = input_elems / known_product;
        } else {
            (*out_shape)[static_cast<size_t>(infer_index)] = -1;
        }
    } else if (input_elems_known && known_product != input_elems) {
        return false;
    }
    return true;
}

static bool InferNodeShape(Node *node) {
    if (node == nullptr || node->output_edges.empty()) {
        return true;
    }
    if (node->op_type == "Relu") {
        if (node->input_edges.empty() || !node->input_edges[0]->has_shape) {
            return false;
        }
        for (Edge *out : node->output_edges) {
            out->shape     = node->input_edges[0]->shape;
            out->has_shape = true;
            if (!out->has_dtype) {
                out->dtype     = node->input_edges[0]->dtype;
                out->has_dtype = node->input_edges[0]->has_dtype;
            }
        }
        return true;
    }

    if (node->op_type == "Conv") {
        if (node->input_edges.size() < 2 || !node->input_edges[0]->has_shape || !node->input_edges[1]->has_shape) {
            return false;
        }
        std::vector<int64_t> pads;
        std::vector<int64_t> strides;
        std::vector<int64_t> dilations;
        GetAttrInts(node, "pads", &pads);
        GetAttrInts(node, "strides", &strides);
        GetAttrInts(node, "dilations", &dilations);

        const auto &weight_shape = node->input_edges[1]->shape;
        if (weight_shape.size() != node->input_edges[0]->shape.size() || weight_shape.size() < 3) {
            return false;
        }
        std::vector<int64_t> kernel_spatial(weight_shape.begin() + 2, weight_shape.end());
        std::vector<int64_t> out_shape;
        if (!InferConvLikeOutput(node->input_edges[0]->shape, weight_shape[0], kernel_spatial, pads, strides, dilations,
                                 &out_shape)) {
            return false;
        }
        Edge *out      = node->output_edges[0];
        out->shape     = std::move(out_shape);
        out->has_shape = true;
        if (!out->has_dtype) {
            out->dtype     = node->input_edges[0]->dtype;
            out->has_dtype = node->input_edges[0]->has_dtype;
        }
        return true;
    }

    if (node->op_type == "MaxPool") {
        if (node->input_edges.empty() || !node->input_edges[0]->has_shape) {
            return false;
        }
        std::vector<int64_t> kernel_shape;
        if (!GetAttrInts(node, "kernel_shape", &kernel_shape) || kernel_shape.empty()) {
            return false;
        }
        std::vector<int64_t> pads;
        std::vector<int64_t> strides;
        std::vector<int64_t> dilations;
        GetAttrInts(node, "pads", &pads);
        GetAttrInts(node, "strides", &strides);
        GetAttrInts(node, "dilations", &dilations);

        std::vector<int64_t> out_shape;
        if (!InferConvLikeOutput(node->input_edges[0]->shape, node->input_edges[0]->shape[1], kernel_shape, pads,
                                 strides, dilations, &out_shape)) {
            return false;
        }

        Edge *out      = node->output_edges[0];
        out->shape     = std::move(out_shape);
        out->has_shape = true;
        if (!out->has_dtype) {
            out->dtype     = node->input_edges[0]->dtype;
            out->has_dtype = node->input_edges[0]->has_dtype;
        }
        return true;
    }

    if (node->op_type == "Reshape") {
        if (node->input_edges.size() < 2 || !node->input_edges[0]->has_shape) {
            return false;
        }
        std::vector<int64_t> shape_values;
        if (!ReadShapeTensor(node->input_edges[1], &shape_values) || shape_values.empty()) {
            return false;
        }
        int64_t allow_zero_i64 = 0;
        if (!GetAttrInt(node, "allowzero", 0, &allow_zero_i64)) {
            return false;
        }
        std::vector<int64_t> out_shape;
        if (!InferReshapeOutput(node->input_edges[0]->shape, shape_values, allow_zero_i64 != 0, &out_shape)) {
            return false;
        }
        Edge *out      = node->output_edges[0];
        out->shape     = std::move(out_shape);
        out->has_shape = true;
        if (!out->has_dtype) {
            out->dtype     = node->input_edges[0]->dtype;
            out->has_dtype = node->input_edges[0]->has_dtype;
        }
        return true;
    }

    if (node->op_type == "Gemm") {
        if (node->input_edges.size() < 2 || !node->input_edges[0]->has_shape || !node->input_edges[1]->has_shape) {
            return false;
        }
        const auto &a_shape = node->input_edges[0]->shape;
        const auto &b_shape = node->input_edges[1]->shape;
        if (a_shape.size() != 2 || b_shape.size() != 2) {
            return false;
        }

        int64_t trans_a = 0;
        int64_t trans_b = 0;
        if (!GetAttrInt(node, "transA", 0, &trans_a) || !GetAttrInt(node, "transB", 0, &trans_b)) {
            return false;
        }

        const int64_t a_m = (trans_a != 0) ? a_shape[1] : a_shape[0];
        const int64_t a_k = (trans_a != 0) ? a_shape[0] : a_shape[1];
        const int64_t b_k = (trans_b != 0) ? b_shape[1] : b_shape[0];
        const int64_t b_n = (trans_b != 0) ? b_shape[0] : b_shape[1];

        if (a_k > 0 && b_k > 0 && a_k != b_k) {
            return false;
        }

        Edge *out      = node->output_edges[0];
        out->shape     = {(a_m > 0 ? a_m : -1), (b_n > 0 ? b_n : -1)};
        out->has_shape = true;
        if (!out->has_dtype) {
            out->dtype     = node->input_edges[0]->dtype;
            out->has_dtype = node->input_edges[0]->has_dtype;
        }
        return true;
    }

    // 未覆盖算子：尝试单输入逐元素广播（保守退化到同形状）。
    if (!node->input_edges.empty() && node->input_edges[0]->has_shape) {
        for (Edge *out : node->output_edges) {
            out->shape     = node->input_edges[0]->shape;
            out->has_shape = true;
            if (!out->has_dtype) {
                out->dtype     = node->input_edges[0]->dtype;
                out->has_dtype = node->input_edges[0]->has_dtype;
            }
        }
        LOG_WARN("算子 `%s` 尚未实现专用 shape rule，退化为输入同形状传播", node->op_type.c_str());
        return true;
    }
    return false;
}

Node *Graph::CreateNode(const std::string &op_type, const std::string &name) {
    Node *node    = arena.AllocNode();
    node->id      = node_id_counter++;
    node->op_type = op_type;
    node->name    = name;
    nodes.push_back(node);
    return node;
}

Edge *Graph::CreateEdge(const std::string &name) {
    Edge *edge            = arena.AllocEdge();
    edge->id              = edge_id_counter++;
    edge->name            = name;
    edge->has_shape       = false;
    edge->has_dtype       = false;
    edge->is_graph_input  = false;
    edge->is_graph_output = false;
    edges.push_back(edge);
    return edge;
}

int Graph::InferShapes() {
    std::unordered_map<uint32_t, uint32_t>              in_degree;
    std::unordered_map<uint32_t, std::vector<uint32_t>> successors;
    std::unordered_map<uint32_t, Node *>                id_to_node;

    for (Node *node : nodes) {
        in_degree[node->id]  = 0;
        id_to_node[node->id] = node;
    }
    for (Node *node : nodes) {
        for (const Edge *input_edge : node->input_edges) {
            if (input_edge->producer != nullptr) {
                successors[input_edge->producer->id].push_back(node->id);
                in_degree[node->id]++;
            }
        }
    }

    std::queue<uint32_t> ready;
    for (const auto &[id, degree] : in_degree) {
        if (degree == 0) {
            ready.push(id);
        }
    }

    std::vector<Node *> topo;
    topo.reserve(nodes.size());
    while (!ready.empty()) {
        const uint32_t id = ready.front();
        ready.pop();
        topo.push_back(id_to_node[id]);
        for (uint32_t succ : successors[id]) {
            if (--in_degree[succ] == 0) {
                ready.push(succ);
            }
        }
    }
    if (topo.size() != nodes.size()) {
        LOG_ERROR("Shape Inference 失败：图中存在环路");
        return -1;
    }

    for (Node *node : topo) {
        if (!InferNodeShape(node)) {
            LOG_ERROR("Shape Inference 失败：算子 `%s` (`%s`) 的输入信息不足或规则不满足", node->op_type.c_str(),
                      node->name.c_str());
            return -1;
        }
    }

    for (Edge *edge : edges) {
        if (!edge->has_dtype && edge->producer != nullptr && !edge->producer->input_edges.empty()) {
            Edge *in0 = edge->producer->input_edges[0];
            if (in0->has_dtype) {
                edge->dtype     = in0->dtype;
                edge->has_dtype = true;
            }
        }
    }

    LOG_INFO("Shape Inference 完成");
    return 0;
}

int Graph::Validate() const {
    for (size_t i = 0; i < edges.size(); ++i) {
        const Edge *edge = edges[i];
        if (edge == nullptr) {
            LOG_ERROR("Graph Validate failed: edges[%zu] is nullptr", i);
            return -1;
        }

        if (edge->name.empty()) {
            LOG_WARN("Graph Validate warn: edge id=%u has empty name", edge->id);
        }

        if (edge->is_constant && !edge->weight_data.empty() && !edge->has_dtype) {
            LOG_ERROR("Graph Validate failed: constant edge `%s` missing dtype", edge->name.c_str());
            return -1;
        }

        if (edge->producer == nullptr && !edge->is_constant && !edge->is_graph_input) {
            LOG_WARN("Graph Validate warn: edge `%s` has no producer and is not marked as graph input",
                     edge->name.c_str());
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node *node = nodes[i];
        if (node == nullptr) {
            LOG_ERROR("Graph Validate failed: nodes[%zu] is nullptr", i);
            return -1;
        }

        for (size_t j = 0; j < node->output_edges.size(); ++j) {
            const Edge *edge = node->output_edges[j];
            if (edge == nullptr) {
                LOG_ERROR("Graph Validate failed: node `%s` output_edges[%zu] is nullptr", node->name.c_str(), j);
                return -1;
            }
            if (edge->producer != node) {
                LOG_ERROR("Graph Validate failed: edge `%s` producer mismatch", edge->name.c_str());
                return -1;
            }
        }

        for (size_t j = 0; j < node->input_edges.size(); ++j) {
            const Edge *edge = node->input_edges[j];
            if (edge == nullptr) {
                LOG_ERROR("Graph Validate failed: node `%s` input_edges[%zu] is nullptr", node->name.c_str(), j);
                return -1;
            }
        }
    }

    for (size_t i = 0; i < graph_inputs.size(); ++i) {
        const Edge *edge = graph_inputs[i];
        if (edge == nullptr) {
            LOG_ERROR("Graph Validate failed: graph_inputs[%zu] is nullptr", i);
            return -1;
        }
        if (!edge->is_graph_input) {
            LOG_ERROR("Graph Validate failed: edge `%s` missing graph input flag", edge->name.c_str());
            return -1;
        }
    }

    for (size_t i = 0; i < graph_outputs.size(); ++i) {
        const Edge *edge = graph_outputs[i];
        if (edge == nullptr) {
            LOG_ERROR("Graph Validate failed: graph_outputs[%zu] is nullptr", i);
            return -1;
        }
        if (!edge->is_graph_output) {
            LOG_ERROR("Graph Validate failed: edge `%s` missing graph output flag", edge->name.c_str());
            return -1;
        }
        if (edge->producer == nullptr && !edge->is_constant && !edge->is_graph_input) {
            LOG_ERROR("Graph Validate failed: graph output `%s` has no producer", edge->name.c_str());
            return -1;
        }
    }

    return 0;
}

void Graph::DumpSummary() const {
    size_t constant_count     = 0;
    size_t graph_input_count  = 0;
    size_t graph_output_count = 0;
    size_t dtype_known_count  = 0;
    size_t shape_known_count  = 0;

    for (size_t i = 0; i < edges.size(); ++i) {
        const Edge *edge = edges[i];
        if (edge == nullptr) {
            continue;
        }

        if (edge->is_constant) {
            constant_count++;
        }
        if (edge->is_graph_input) {
            graph_input_count++;
        }
        if (edge->is_graph_output) {
            graph_output_count++;
        }
        if (edge->has_dtype) {
            dtype_known_count++;
        }
        if (edge->has_shape) {
            shape_known_count++;
        }
    }

    LOG_INFO("Graph Summary: %zu nodes, %zu edges, %zu inputs, %zu outputs", nodes.size(), edges.size(),
             graph_inputs.size(), graph_outputs.size());
    LOG_INFO("Graph Edge Stats: %zu constants, %zu graph_inputs, %zu graph_outputs, %zu typed, %zu shaped",
             constant_count, graph_input_count, graph_output_count, dtype_known_count, shape_known_count);
}

Node *Graph::Arena::AllocNode() {
    node_storage.push_back(std::make_unique<Node>());
    return node_storage.back().get();
}

Edge *Graph::Arena::AllocEdge() {
    edge_storage.push_back(std::make_unique<Edge>());
    return edge_storage.back().get();
}
