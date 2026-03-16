#include "backend/vm_program.h"
#include "common/log.h"
#include "execution/exec_program.h"
#include "graph/graph.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.onnx> [output.mvmp] [--max-dynamic-size N]\n", argv[0]);
        return -1;
    }

    const char *model_path       = nullptr;
    const char *output_path      = "build/program.mvmp";
    int64_t     max_dynamic_size = 2;

    for (int i = 1; i < argc; ++i) {
        if (model_path == nullptr) {
            model_path = argv[i];
        } else {
            output_path = argv[i];
        }
    }

    if (model_path == nullptr) {
        std::fprintf(stderr, "error: no model file specified\n");
        return -1;
    }

    Graph graph;
    graph.SetDefaultMaxDynamicSize(max_dynamic_size);

    if (graph.BuildFromONNX(model_path) != 0) {
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

    const auto &plan         = exec.GetMemoryPlan();
    uint64_t    total_memory = plan.constant_pool_size + plan.runtime_pool_size;
    LOG_INFO("Compilation complete. Max memory required: %lu bytes (%.2f KB / %.2f MB)", (unsigned long)total_memory,
             total_memory / 1024.0, total_memory / (1024.0 * 1024.0));
    LOG_INFO("Compilation complete. Runtime memory required: %lu bytes (%.2f KB / %.2f MB)",
             (unsigned long)plan.runtime_pool_size, plan.runtime_pool_size / 1024.0,
             plan.runtime_pool_size / (1024.0 * 1024.0));

    return 0;
}
