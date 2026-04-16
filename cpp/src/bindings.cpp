#include "../include/bindings.h"

#include "../include/array_binops.h"
#include "../include/array_handle.h"
#include "../include/array_inplaceops.h"
#include "../include/array_nullaryops.h"
#include "../include/array_sum.h"
#include "../include/array_unaryops.h"
#include "../include/compiler.h"
#include "../include/graph.h"

namespace nb = nanobind;

NB_MODULE(_backend, m) {
    // DOC //
    m.doc() = "Forge";

    // ARRAY HANDLE //
    nb::class_<ArrayHandle>(m, "ArrayHandle")
        .def_prop_ro("shape", [](const ArrayHandle& h) { return h.shape(); })
        .def_prop_ro("strides", [](const ArrayHandle& h) { return h.strides(); })
        .def_prop_ro("offset", [](const ArrayHandle& h) { return h.offset(); })
        .def_prop_ro("data", [](const ArrayHandle& h) { return h.data(); })
        .def("item", [](ArrayHandle& h) -> float {
            if (!h.shape().empty()) {
                throw std::runtime_error("item(): can only convert scalar arrays to float");
            }
            h.synchronize();
            return h.data()[h.offset()];
        });
    m.def(
        "create_array_from_buffer",
        [](nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu> arr,
           std::vector<int64_t> shape) {
            return create_array_from_buffer_py(arr, shape, /*FH=*/nullptr);
        },
        nb::arg("arr"), nb::arg("shape"));
    m.def("make_view", [](std::shared_ptr<ArrayHandle> h, std::vector<int64_t> shape,
                          std::vector<int64_t> strides, size_t offset) {
        return std::make_shared<ArrayHandle>(h, shape, strides, offset);
    });
    m.def("copy_to_view", [](std::shared_ptr<ArrayHandle> h, std::shared_ptr<ArrayHandle> other,
                             std::vector<int64_t> shape, std::vector<int64_t> strides,
                             size_t offset) { h->copy_from(other, shape, strides, offset); });
    m.def("reshape", &array_reshape);
    m.def("array_shape", &array_shape);
    m.def("array_to_list", &array_to_list);
    m.def("set_seed", [](int32_t seed) { return get_default_forge()->set_seed(seed); });

    // OPERATIONS //
    // nullary_ops //
    m.def("rand",
          [](const std::vector<int64_t>& shape) { return array_nullaryops(shape, "rand"); });
    m.def("randn",
          [](const std::vector<int64_t>& shape) { return array_nullaryops(shape, "randn"); });
    m.def("zeros",
          [](const std::vector<int64_t>& shape) { return array_nullaryops(shape, "zeros"); });

    // unary_ops //
    m.def("exp", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "exp"); });
    m.def("exp2", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "exp2"); });
    m.def("exp10",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "exp10"); });
    m.def("log", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "log"); });
    m.def("log2", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "log2"); });
    m.def("log10",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "log10"); });
    m.def("sqrt", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "sqrt"); });
    m.def("rsqrt",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "rsqrt"); });
    m.def("abs", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "abs"); });
    m.def("sign", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "sign"); });
    m.def("ceil", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "ceil"); });
    m.def("floor",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "floor"); });
    m.def("round",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "round"); });
    m.def("trunc",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "trunc"); });
    m.def("fract",
          [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "fract"); });
    m.def("sin", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "sin"); });
    m.def("cos", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "cos"); });
    m.def("tan", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "tan"); });
    m.def("asin", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "asin"); });
    m.def("acos", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "acos"); });
    m.def("atan", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "atan"); });
    m.def("sinh", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "sinh"); });
    m.def("cosh", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "cosh"); });
    m.def("tanh", [](const std::shared_ptr<ArrayHandle>& a) { return array_unaryops(a, "tanh"); });

    // inplace_ops //
    m.def("iadd", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_inplaceops(a, b, "iadd");
    });
    m.def("isub", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_inplaceops(a, b, "isub");
    });
    m.def("imul", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_inplaceops(a, b, "imul");
    });
    m.def("idiv", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_inplaceops(a, b, "idiv");
    });

    // binary_ops //
    m.def("add", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "add");
    });
    m.def("sub", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "sub");
    });
    m.def("mul", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "mul");
    });
    m.def("div", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "div");
    });
    m.def("matmul", [](const std::shared_ptr<ArrayHandle>& a,
                       const std::shared_ptr<ArrayHandle>& b) { return array_matmul(a, b); });

    // reduction_ops //
    m.def("sum_global", &sum_global);
    m.def("sum_axis", &sum_axis);

    // COMPILE AND RUN //
    nb::class_<Graph>(m, "Graph").def("execute", &Graph::execute);
    m.def("make_graph", &make_graph);
}
