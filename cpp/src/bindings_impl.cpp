#include "../include/array_handle.h"
#include "../include/bindings.h"
#include "../include/compiler.h"
#include "../include/graph.h"
#include "../include/memory_arena.h"

namespace nb = nanobind;

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(
    nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu> arr, std::vector<int64_t> shape,
    ForgeHandle* FH) {
    int64_t total = numel_from_shape(shape);
    if (arr.size() != total) {
        throw std::runtime_error(
            "create_array_from_buffer: buffer length doesn't match given shape");
    }
    float* src_ptr = arr.data();
    void* dev = FH ? FH->device_ptr() : get_default_forge()->device_ptr();
    return std::make_shared<ArrayHandle>(src_ptr, shape, dev);
}

nb::object array_to_list(const ArrayHandle& h) {
    const_cast<ArrayHandle&>(h).synchronize();
    const std::vector<int64_t> shape = h.shape();
    const std::vector<int64_t> strides = h.strides();
    const std::span<const float> data = h.data();
    if (shape.empty() || strides.empty()) {
        return nb::cast(data.size() ? data[h.offset()] : 0.0f);
    }

    std::function<nb::object(size_t, size_t)> build;
    build = [&](size_t dim, size_t offset) -> nb::object {
        int64_t stride = strides[dim];
        if (dim + 1 == shape.size()) {
            nb::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(nb::float_(data[offset + i * stride]));
            return lst;
        } else {
            nb::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(build(dim + 1, offset + i * stride));
            return lst;
        }
    };
    return build(0, h.offset());
}

static std::vector<int64_t> infer_recursive(nb::handle x, std::vector<float>& flat) {
    PyObject* obj = x.ptr();

    if (PyFloat_Check(obj)) {
        flat.push_back(static_cast<float>(PyFloat_AsDouble(obj)));
        return {};
    }
    if (PyLong_Check(obj)) {
        double d = PyLong_AsDouble(obj);
        if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
        flat.push_back(static_cast<float>(d));
        return {};
    }

    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        bool is_list = PyList_Check(obj);
        Py_ssize_t len = is_list ? PyList_GET_SIZE(obj) : PyTuple_GET_SIZE(obj);
        if (len == 0) return {0};

        std::vector<int64_t> first_sub;
        for (Py_ssize_t i = 0; i < len; ++i) {
            PyObject* item = is_list ? PyList_GET_ITEM(obj, i) : PyTuple_GET_ITEM(obj, i);
            std::vector<int64_t> sub = infer_recursive(nb::handle(item), flat);
            if (i == 0) {
                first_sub = std::move(sub);
            } else if (sub != first_sub) {
                PyErr_SetString(PyExc_ValueError, "ragged nested lists: differing inner shapes");
                throw nb::python_error();
            }
        }
        std::vector<int64_t> result;
        result.reserve(first_sub.size() + 1);
        result.push_back(static_cast<int64_t>(len));
        result.insert(result.end(), first_sub.begin(), first_sub.end());
        return result;
    }

    if (PyBytes_Check(obj) || PyByteArray_Check(obj) || PyMemoryView_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "shape required for raw bytes; use Array.from_buffer(buf, shape) and "
                        "specify shape");
        throw nb::python_error();
    }

    static PyObject* array_type = []() -> PyObject* {
        PyObject* mod = PyImport_ImportModule("array");
        if (!mod) {
            PyErr_Clear();
            return nullptr;
        }
        PyObject* t = PyObject_GetAttrString(mod, "array");
        Py_DECREF(mod);
        if (!t) PyErr_Clear();
        return t;
    }();
    if (array_type && PyObject_IsInstance(obj, array_type) == 1) {
        PyObject* typecode_obj = PyObject_GetAttrString(obj, "typecode");
        if (!typecode_obj) throw nb::python_error();
        const char* typecode = PyUnicode_AsUTF8(typecode_obj);
        bool is_float = typecode && std::string(typecode) == "f";
        Py_DECREF(typecode_obj);
        if (!is_float) {
            PyErr_SetString(PyExc_TypeError, "array must be of type 'float32'");
            throw nb::python_error();
        }
        Py_buffer buf;
        if (PyObject_GetBuffer(obj, &buf, PyBUF_SIMPLE) != 0) throw nb::python_error();
        Py_ssize_t n = buf.len / static_cast<Py_ssize_t>(sizeof(float));
        const float* src = static_cast<const float*>(buf.buf);
        flat.insert(flat.end(), src, src + n);
        PyBuffer_Release(&buf);
        return {static_cast<int64_t>(n)};
    }

    std::string msg = std::string("unsupported input type: ") +
                      nb::str(nb::handle(reinterpret_cast<PyObject*>(Py_TYPE(obj)))).c_str();
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    throw nb::python_error();
}

std::pair<std::vector<int64_t>, std::shared_ptr<ArrayHandle>> infer_shape_and_flatten_py(
    nb::object data) {
    std::vector<float> flat;
    std::vector<int64_t> shape = infer_recursive(data, flat);
    void* dev = get_default_forge()->device_ptr();
    auto handle = std::make_shared<ArrayHandle>(flat.data(), shape, dev);
    return {std::move(shape), std::move(handle)};
}

std::vector<Node> parse_nodes(nb::list flat_nodes) {
    std::vector<Node> nodes;
    nodes.reserve(flat_nodes.size());

    for (auto handle : flat_nodes) {
        auto t = nb::cast<nb::tuple>(handle);

        Node n;
        n.op = static_cast<OpCode>(nb::cast<int>(t[0]));
        n.inputs = nb::cast<std::vector<int>>(t[1]);
        n.shape = nb::cast<std::vector<int64_t>>(t[2]);
        n.offset = nb::cast<int64_t>(t[3]);
        n.strides = nb::cast<std::vector<int64_t>>(t[4]);
        nb::tuple py_args = nb::cast<nb::tuple>(t[5]);

        // different operation add more later, if they take args
        // consider using a switch statement
        if (n.op == OpCode::UPDATE) {
            // py_args = (shape, strides, offset)
            auto s = nb::cast<std::vector<int64_t>>(py_args[0]);
            auto st = nb::cast<std::vector<int64_t>>(py_args[1]);
            int64_t off = nb::cast<int64_t>(py_args[2]);
            // flatten
            n.args.insert(n.args.end(), s.begin(), s.end());
            n.args.insert(n.args.end(), st.begin(), st.end());
            n.args.push_back(off);
        }
        nodes.push_back(n);
    }
    return nodes;
}

std::shared_ptr<Graph> make_graph(nb::list flat_nodes, int output_index) {
    // 1. Get the basic graph
    std::vector<Node> raw_nodes = parse_nodes(flat_nodes);
    // 2. Optimize graph
    std::vector<Node> optimized_nodes = optimize_graph(raw_nodes);
    // 3. Make graph
    // possible that output_index changes after compiling no?
    auto graph = std::make_shared<Graph>(std::move(optimized_nodes), output_index);
    // 4. Get shared memory map (with some Data struct)
    graph->arena = std::make_shared<MemoryArena>(*graph);
    // 5. Compile Graph, to get strings of the relevant kernels and associated info
    generateKernels(*graph);
    // 6. Pre-Compile Metal (MSL -> MTLComputePipelineState)
    compile_metal(*graph);
    return graph;
}
