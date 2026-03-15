// -*- coding: utf-8 -*-
// py_distributed.cpp — pybind11 bindings for _qforge_distributed
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "qforge/distributed_backend.h"

namespace py = pybind11;
using C = std::complex<double>;

PYBIND11_MODULE(_qforge_distributed, m) {
    m.doc() = "Qforge MPI-distributed state vector engine";

    py::class_<qforge::distributed::DistributedStateVector>(m, "DistributedStateVector")
        .def(py::init<int>(), py::arg("n_qubits"),
             "Create distributed state vector. Uses MPI_COMM_WORLD.")
        .def_property_readonly("n_qubits", &qforge::distributed::DistributedStateVector::n_qubits)
        .def_property_readonly("local_dim", &qforge::distributed::DistributedStateVector::local_dim)
        .def_property_readonly("rank", &qforge::distributed::DistributedStateVector::rank)
        .def_property_readonly("n_ranks", &qforge::distributed::DistributedStateVector::n_ranks)
        .def("reset", &qforge::distributed::DistributedStateVector::reset)

        // Amplitude property (local shard as numpy view)
        .def_property_readonly("amplitude",
            [](qforge::distributed::DistributedStateVector& sv) {
                return py::array_t<C>(
                    {sv.local_dim()},
                    {sizeof(C)},
                    sv.local_data(),
                    py::cast(&sv)
                );
            }, "Local amplitude shard as numpy view")

        // Single-qubit gates (with GIL release)
        .def("H", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.H(t);
        }, py::arg("target"))
        .def("X", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.X(t);
        }, py::arg("target"))
        .def("Y", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.Y(t);
        }, py::arg("target"))
        .def("Z", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.Z(t);
        }, py::arg("target"))
        .def("RX", [](qforge::distributed::DistributedStateVector& sv, int t, double theta) {
            py::gil_scoped_release rel; sv.RX(t, theta);
        }, py::arg("target"), py::arg("theta"))
        .def("RY", [](qforge::distributed::DistributedStateVector& sv, int t, double theta) {
            py::gil_scoped_release rel; sv.RY(t, theta);
        }, py::arg("target"), py::arg("theta"))
        .def("RZ", [](qforge::distributed::DistributedStateVector& sv, int t, double theta) {
            py::gil_scoped_release rel; sv.RZ(t, theta);
        }, py::arg("target"), py::arg("theta"))
        .def("Phase", [](qforge::distributed::DistributedStateVector& sv, int t, double phi) {
            py::gil_scoped_release rel; sv.Phase(t, phi);
        }, py::arg("target"), py::arg("phi"))
        .def("S", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.S(t);
        }, py::arg("target"))
        .def("T", [](qforge::distributed::DistributedStateVector& sv, int t) {
            py::gil_scoped_release rel; sv.T(t);
        }, py::arg("target"))

        // Controlled gates
        .def("CNOT", [](qforge::distributed::DistributedStateVector& sv, int c, int t) {
            py::gil_scoped_release rel; sv.CNOT(c, t);
        }, py::arg("control"), py::arg("target"))
        .def("CRX", [](qforge::distributed::DistributedStateVector& sv, int c, int t, double theta) {
            py::gil_scoped_release rel; sv.CRX(c, t, theta);
        }, py::arg("control"), py::arg("target"), py::arg("theta"))
        .def("CRY", [](qforge::distributed::DistributedStateVector& sv, int c, int t, double theta) {
            py::gil_scoped_release rel; sv.CRY(c, t, theta);
        }, py::arg("control"), py::arg("target"), py::arg("theta"))
        .def("CRZ", [](qforge::distributed::DistributedStateVector& sv, int c, int t, double theta) {
            py::gil_scoped_release rel; sv.CRZ(c, t, theta);
        }, py::arg("control"), py::arg("target"), py::arg("theta"))
        .def("CPhase", [](qforge::distributed::DistributedStateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release rel; sv.CPhase(c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi"))

        // Measurement
        .def("measure_one_prob0",
            [](qforge::distributed::DistributedStateVector& sv, int q) {
                py::gil_scoped_release rel;
                return sv.measure_one_prob0(q);
            }, py::arg("qubit"))
        .def("collapse_one",
            [](qforge::distributed::DistributedStateVector& sv, int q, int v) {
                py::gil_scoped_release rel;
                sv.collapse_one(q, v);
            }, py::arg("qubit"), py::arg("value"))
        .def("pauli_expectation",
            [](qforge::distributed::DistributedStateVector& sv, int q, int pt) {
                py::gil_scoped_release rel;
                return sv.pauli_expectation(q, pt);
            }, py::arg("qubit"), py::arg("pauli_type"))

        // Gather full state to rank 0 (for testing)
        .def("gather",
            [](const qforge::distributed::DistributedStateVector& sv) {
                size_t total = size_t(1) << sv.n_qubits();
                auto result = py::array_t<C>(total);
                auto buf = result.request();
                {
                    py::gil_scoped_release rel;
                    sv.gather_amplitudes(static_cast<C*>(buf.ptr));
                }
                return result;
            }, "Gather full amplitude vector to rank 0 (expensive)")
    ;
}
