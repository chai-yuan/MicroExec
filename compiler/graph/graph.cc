#include "graph/graph.h"

#include "common/log.h"

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
