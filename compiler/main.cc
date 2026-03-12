#include "backend/vm_program.h"
#include "execution/exec_program.h"
#include "graph/graph.h"

#include <cstdio>

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.onnx> [output.mvmp]\n", argv[0]);
        return -1;
    }
    const char *output_path = (argc >= 3) ? argv[2] : "build/program.mvmp";

    Graph graph;
    if (graph.BuildFromONNX(argv[1]) != 0) {
        return -1;
    }
    if (graph.InferShapes() != 0) {
        return -1;
    }

    ExecProgram exec;
    if (exec.BuildFromGraph(graph) != 0) {
        return -1;
    }
    if (exec.AnalyzeLifetimes() != 0) {
        return -1;
    }
    if (exec.PlanMemory() != 0) {
        return -1;
    }
    if (exec.Validate() != 0) {
        return -1;
    }
    exec.Dump();

    ProgramBuilder builder;
    if (builder.BuildFromExecProgram(exec) != 0) {
        return -1;
    }
    if (builder.Serialize(output_path) != 0) {
        return -1;
    }

    return 0;
}
