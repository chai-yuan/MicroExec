#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "common/graph_types.h"

class Graph;
class Node;

struct Attribute {
    AttributeKind kind = AttributeKind::UNDEFINED;

    std::variant<int64_t, float, std::string, std::vector<int64_t>, std::vector<float>, Graph *> value;
};

class Edge {
  public:
    uint32_t             id = 0;
    std::string          name;
    std::vector<int64_t> shape;
    DataType             dtype = DataType::UNKNOWN;

    bool has_shape       = false;
    bool has_dtype       = false;
    bool is_constant     = false;
    bool is_graph_input  = false;
    bool is_graph_output = false;

    std::vector<uint8_t> weight_data; // 如果是常量，存储权重数据

    Node               *producer = nullptr; // 谁生产了这个数据（为空表示是图的全局输入或常量）
    std::vector<Node *> consumers;           // 谁在使用这个数据
};

class Node {
  public:
    uint32_t    id = 0;
    std::string name;
    std::string op_type;
    std::string domain;

    std::vector<Edge *> input_edges;
    std::vector<Edge *> output_edges;

    std::map<std::string, Attribute> attributes;
};

class Graph {
  public:
    int  BuildFromONNX(const std::string &file_name);
    int  InferShapes();
    int  Validate() const;
    void DumpSummary() const;

    Node *CreateNode(const std::string &op_type, const std::string &name);
    Edge *CreateEdge(const std::string &name);

    const std::vector<Node *> &GetNodes() const { return nodes; }
    const std::vector<Edge *> &GetEdges() const { return edges; }
    const std::vector<Edge *> &GetGraphInputs() const { return graph_inputs; }
    const std::vector<Edge *> &GetGraphOutputs() const { return graph_outputs; }

  private:
    class Arena {
      public:
        Node *AllocNode();
        Edge *AllocEdge();

      private:
        std::vector<std::unique_ptr<Node>> node_storage;
        std::vector<std::unique_ptr<Edge>> edge_storage;
    };
    Arena arena;

    uint32_t node_id_counter = 0;
    uint32_t edge_id_counter = 0;

    std::vector<Node *> nodes;
    std::vector<Edge *> edges;

    std::vector<Edge *> graph_inputs;
    std::vector<Edge *> graph_outputs;
};
