#include "execution/exec_program.h"
#include "graph/graph.h"

#include <cstdio>

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        return -1;
    }

    Graph graph;
    if (graph.BuildFromONNX(argv[1]) != 0) {
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

    return 0;
}
