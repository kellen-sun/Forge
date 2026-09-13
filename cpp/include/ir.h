#pragma once

#include "graph.h"

// The unit a Pass rewrites. Graph is the whole object (arena, pipelines)
// IR is just nodes + which value is the output.
struct IR {
    std::vector<Node> nodes;
    int output_index = 0;
};
