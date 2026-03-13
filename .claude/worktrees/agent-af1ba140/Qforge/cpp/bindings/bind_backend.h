#pragma once
// Shared pybind11 binding helpers for Backend-derived classes.
// Include this in py_metal.cpp and py_cuda.cpp.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "qforge/backend.h"

namespace py = pybind11;

// Bind all gate/measurement methods + amplitude property to a py::class_
template <typename BackendClass>
void bind_backend_methods(py::class_<BackendClass>& cls) {
    cls
        .def_property_readonly("n_qubits", &BackendClass::n_qubits)
        .def_property_readonly("dim", &BackendClass::dim)
        .def("reset", &BackendClass::reset)

        // Zero-copy numpy amplitude property
        .def_property("amplitude",
            [](BackendClass& b) -> py::array_t<std::complex<double>> {
                auto* ptr = b.host_data();
                return py::array_t<std::complex<double>>(
                    {b.dim()},
                    {sizeof(std::complex<double>)},
                    ptr,
                    py::cast(&b)
                );
            },
            [](BackendClass& b, py::array_t<std::complex<double>> arr) {
                auto buf = arr.request();
                if (static_cast<size_t>(buf.size) != b.dim())
                    throw std::runtime_error("Size mismatch");
                std::memcpy(b.host_data(), buf.ptr,
                            b.dim() * sizeof(std::complex<double>));
                b.sync_to_device();
            }
        )

        // --- Single-qubit gates ---
        .def("H", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.H(t);
        }, py::arg("target"))
        .def("X", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.X(t);
        }, py::arg("target"))
        .def("Y", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.Y(t);
        }, py::arg("target"))
        .def("Z", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.Z(t);
        }, py::arg("target"))
        .def("RX", [](BackendClass& b, int t, double phi) {
            py::gil_scoped_release release;
            b.RX(t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("RY", [](BackendClass& b, int t, double phi) {
            py::gil_scoped_release release;
            b.RY(t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("RZ", [](BackendClass& b, int t, double phi) {
            py::gil_scoped_release release;
            b.RZ(t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("Phase", [](BackendClass& b, int t, double phi) {
            py::gil_scoped_release release;
            b.Phase(t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("S", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.S(t);
        }, py::arg("target"))
        .def("T", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.T(t);
        }, py::arg("target"))
        .def("Xsquare", [](BackendClass& b, int t) {
            py::gil_scoped_release release;
            b.Xsquare(t);
        }, py::arg("target"))

        // --- Controlled gates ---
        .def("CNOT", [](BackendClass& b, int c, int t) {
            py::gil_scoped_release release;
            b.CNOT(c, t);
        }, py::arg("control"), py::arg("target"))
        .def("CRX", [](BackendClass& b, int c, int t, double phi) {
            py::gil_scoped_release release;
            b.CRX(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CRY", [](BackendClass& b, int c, int t, double phi) {
            py::gil_scoped_release release;
            b.CRY(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CRZ", [](BackendClass& b, int c, int t, double phi) {
            py::gil_scoped_release release;
            b.CRZ(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CPhase", [](BackendClass& b, int c, int t, double phi) {
            py::gil_scoped_release release;
            b.CPhase(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CP", [](BackendClass& b, int c, int t, double phi) {
            py::gil_scoped_release release;
            b.CP(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)

        // --- Double-controlled gates ---
        .def("CCNOT", [](BackendClass& b, int c1, int c2, int t) {
            py::gil_scoped_release release;
            b.CCNOT(c1, c2, t);
        }, py::arg("c1"), py::arg("c2"), py::arg("target"))
        .def("OR", [](BackendClass& b, int c1, int c2, int t) {
            py::gil_scoped_release release;
            b.OR(c1, c2, t);
        }, py::arg("c1"), py::arg("c2"), py::arg("target"))

        // --- Swap gates ---
        .def("SWAP", [](BackendClass& b, int t1, int t2) {
            py::gil_scoped_release release;
            b.SWAP(t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("CSWAP", [](BackendClass& b, int c, int t1, int t2) {
            py::gil_scoped_release release;
            b.CSWAP(c, t1, t2);
        }, py::arg("control"), py::arg("t1"), py::arg("t2"))
        .def("ISWAP", [](BackendClass& b, int t1, int t2) {
            py::gil_scoped_release release;
            b.ISWAP(t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("SISWAP", [](BackendClass& b, int t1, int t2) {
            py::gil_scoped_release release;
            b.SISWAP(t1, t2);
        }, py::arg("t1"), py::arg("t2"))

        // --- Noise ---
        .def("E", [](BackendClass& b, double p, int t) {
            py::gil_scoped_release release;
            b.E(p, t);
        }, py::arg("p_noise"), py::arg("target"))
        .def("E_all", [](BackendClass& b, double p) {
            py::gil_scoped_release release;
            b.E_all(p);
        }, py::arg("p_noise"))

        // --- Measurement ---
        .def("measure_one_prob0", [](const BackendClass& b, int q) {
            py::gil_scoped_release release;
            return b.measure_one_prob0(q);
        }, py::arg("qubit"))
        .def("collapse_one", [](BackendClass& b, int q, int v) {
            py::gil_scoped_release release;
            b.collapse_one(q, v);
        }, py::arg("qubit"), py::arg("value"))
        .def("pauli_expectation", [](const BackendClass& b, int q, int pt) {
            py::gil_scoped_release release;
            return b.pauli_expectation(q, pt);
        }, py::arg("qubit"), py::arg("pauli_type"))
        .def("probabilities", [](const BackendClass& b) {
            auto result = py::array_t<double>(b.dim());
            auto buf = result.request();
            {
                py::gil_scoped_release release;
                b.probabilities(static_cast<double*>(buf.ptr));
            }
            return result;
        })
    ;
}
