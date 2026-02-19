#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "array_handle.h"
#include "graph.h"

namespace nb = nanobind;

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(
    nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu> arr, std::vector<int64_t> shape,
    ForgeHandle* FH);

nb::object array_to_list(const ArrayHandle& h);

std::vector<Node> parse_nodes(nb::list flat_nodes);

std::shared_ptr<Graph> make_graph(nb::list flat_nodes, int output_index);
