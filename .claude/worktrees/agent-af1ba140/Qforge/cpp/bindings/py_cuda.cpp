#include "bind_backend.h"
#include "qforge/cuda_backend.h"

PYBIND11_MODULE(_qforge_cuda, m) {
    m.doc() = "Qforge CUDA GPU acceleration engine (NVIDIA)";

    auto cls = py::class_<qforge::CudaBackend>(m, "CudaStateVector");
    cls.def(py::init<int>(), py::arg("n_qubits"));
    bind_backend_methods(cls);

    m.attr("backend_type") = "cuda";
}
