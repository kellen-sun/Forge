#pragma once
#include "graph.h"

std::vector<Node> optimize_graph(std::vector<Node> raw_nodes);

void generateKernels(Graph& graph);

void compile_metal(Graph& graph);
