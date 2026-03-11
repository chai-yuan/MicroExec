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

    return 0;
}
