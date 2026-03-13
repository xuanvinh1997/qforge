#include "bind_backend.h"
#include "qforge/metal_backend.h"

PYBIND11_MODULE(_qforge_metal, m) {
    m.doc() = "Qsun Metal GPU acceleration engine (Apple Silicon)";

    auto cls = py::class_<qforge::MetalBackend>(m, "MetalStateVector");
    cls.def(py::init<int>(), py::arg("n_qubits"));
    bind_backend_methods(cls);

    m.attr("backend_type") = "metal";
}
