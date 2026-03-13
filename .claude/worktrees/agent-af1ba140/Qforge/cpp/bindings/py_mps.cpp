// -*- coding: utf-8 -*-
// py_mps.cpp — pybind11 bindings for _qsun_mps (MPS + MPO + DMRG)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "qforge/mps.h"
#include "qforge/mpo.h"

namespace py = pybind11;

// Forward declare dmrg_sweep from dmrg_sweep.cpp
namespace qforge { namespace dmrg {
double dmrg_sweep(mps::MPS& psi, const mpo::MPO& H,
                  int max_chi, double eps, int lanczos_dim);
}}

PYBIND11_MODULE(_qforge_mps, m) {
    m.doc() = "Qsun MPS/DMRG C++ acceleration engine";

    // ================================================================
    // MPS class
    // ================================================================
    py::class_<qforge::mps::MPS>(m, "MPS")
        .def(py::init<int, int>(),
             py::arg("n_qubits"), py::arg("max_bond_dim") = 32,
             "Create MPS initialized to |00...0>")
        .def_property_readonly("n_qubits", &qforge::mps::MPS::n_qubits)
        .def_property_readonly("max_bond_dim", &qforge::mps::MPS::max_bond_dim)
        .def("reset", &qforge::mps::MPS::reset,
             "Reset to |00...0> product state")
        .def("bond_dimensions", [](const qforge::mps::MPS& mps) {
                std::vector<int> dims;
                dims.reserve(mps.n_qubits() - 1);
                for (int i = 0; i < mps.n_qubits() - 1; ++i)
                    dims.push_back(mps.bond_dim(i));
                return dims;
             }, "Return list of bond dimensions (N-1 values)")
        .def("max_current_chi", &qforge::mps::MPS::max_current_chi,
             "Maximum bond dimension currently in use")
        // --- State vector conversion ---
        .def("to_statevector",
             [](const qforge::mps::MPS& mps) {
                 size_t dim = size_t(1) << mps.n_qubits();
                 auto result = py::array_t<std::complex<double>>(dim);
                 auto buf = result.request();
                 {
                     py::gil_scoped_release rel;
                     qforge::mps::ops::to_statevector(
                         mps, static_cast<std::complex<double>*>(buf.ptr));
                 }
                 return result;
             }, "Contract MPS to full amplitude vector (exponential cost)")
        .def("from_statevector",
             [](qforge::mps::MPS& mps,
                py::array_t<std::complex<double>> amp,
                double eps) {
                 auto buf = amp.request();
                 qforge::mps::SVDWorkspace ws;
                 {
                     py::gil_scoped_release rel;
                     qforge::mps::ops::from_statevector(
                         mps,
                         static_cast<const std::complex<double>*>(buf.ptr),
                         static_cast<size_t>(buf.size),
                         mps.max_bond_dim(), eps, ws);
                 }
             }, py::arg("amp"), py::arg("eps") = 1e-10,
             "Set MPS from amplitude vector via sequential SVD")
        // --- Gate application ---
        .def("apply_single_qubit_gate",
             [](qforge::mps::MPS& mps, int site,
                py::array_t<std::complex<double>> gate) {
                 auto buf = gate.request();
                 if (buf.size != 4)
                     throw std::runtime_error("gate must be a 4-element (2x2) array");
                 {
                     py::gil_scoped_release rel;
                     qforge::mps::ops::apply_single_qubit_gate(
                         mps, site,
                         static_cast<const std::complex<double>*>(buf.ptr));
                 }
             }, py::arg("site"), py::arg("gate"),
             "Apply 2x2 gate to single qubit site (no truncation)")
        .def("apply_two_qubit_gate",
             [](qforge::mps::MPS& mps, int site_i,
                py::array_t<std::complex<double>> gate,
                int max_chi, double eps) {
                 auto buf = gate.request();
                 if (buf.size != 16)
                     throw std::runtime_error("gate must be a 16-element (4x4) array");
                 qforge::mps::SVDWorkspace ws;
                 {
                     py::gil_scoped_release rel;
                     return qforge::mps::ops::apply_two_qubit_gate(
                         mps, site_i,
                         static_cast<const std::complex<double>*>(buf.ptr),
                         max_chi, eps, ws);
                 }
             }, py::arg("site_i"), py::arg("gate"),
                py::arg("max_chi") = 32, py::arg("eps") = 1e-10,
             "Apply 4x4 gate to neighboring qubits site_i, site_i+1 with SVD truncation. Returns truncation error.")
        // --- Measurement ---
        .def("single_site_expectation",
             [](const qforge::mps::MPS& mps, int site,
                py::array_t<std::complex<double>> op) {
                 auto buf = op.request();
                 if (buf.size != 4)
                     throw std::runtime_error("op must be 4-element (2x2) array");
                 py::gil_scoped_release rel;
                 return qforge::mps::ops::single_site_expectation(
                     mps, site,
                     static_cast<const std::complex<double>*>(buf.ptr));
             }, py::arg("site"), py::arg("op"),
             "<psi|op_site|psi>")
        .def("two_site_expectation",
             [](const qforge::mps::MPS& mps, int si, int sj,
                py::array_t<std::complex<double>> opi,
                py::array_t<std::complex<double>> opj) {
                 auto bi = opi.request(), bj = opj.request();
                 py::gil_scoped_release rel;
                 return qforge::mps::ops::two_site_expectation(
                     mps, si, sj,
                     static_cast<const std::complex<double>*>(bi.ptr),
                     static_cast<const std::complex<double>*>(bj.ptr));
             }, py::arg("si"), py::arg("sj"), py::arg("opi"), py::arg("opj"),
             "<psi|opi_si * opj_sj|psi>")
        .def("measure_prob0",
             [](const qforge::mps::MPS& mps, int site) {
                 py::gil_scoped_release rel;
                 return qforge::mps::ops::measure_prob0(mps, site);
             }, py::arg("site"),
             "Probability of measuring qubit site in |0>")
        .def("entanglement_entropy",
             [](const qforge::mps::MPS& mps, int bond) {
                 py::gil_scoped_release rel;
                 return qforge::mps::ops::entanglement_entropy(mps, bond);
             }, py::arg("bond"),
             "Von Neumann entanglement entropy at bond (bond, bond+1)")
        .def("max_entanglement_entropy",
             [](const qforge::mps::MPS& mps) {
                 double mx = 0.0;
                 for (int b = 0; b < mps.n_qubits() - 1; ++b) {
                     py::gil_scoped_release rel;
                     double s = qforge::mps::ops::entanglement_entropy(mps, b);
                     if (s > mx) mx = s;
                 }
                 return mx;
             }, "Maximum entanglement entropy across all bonds")
        .def("norm",
             [](const qforge::mps::MPS& mps) {
                 py::gil_scoped_release rel;
                 return qforge::mps::ops::norm(mps);
             }, "Norm of MPS state: sqrt(<psi|psi>)")
    ;

    // ================================================================
    // MPO class
    // ================================================================
    py::class_<qforge::mpo::MPO>(m, "MPO")
        .def(py::init<int, int>(),
             py::arg("n_sites"), py::arg("bond_dim") = 5)
        .def_property_readonly("n_sites", &qforge::mpo::MPO::n_sites)
        .def_property_readonly("bond_dim", &qforge::mpo::MPO::bond_dim)
    ;

    // ================================================================
    // MPO factory functions
    // ================================================================
    m.def("build_heisenberg_mpo",
          [](int n, double J) {
              py::gil_scoped_release rel;
              return qforge::mpo::heisenberg(n, J);
          }, py::arg("n_sites"), py::arg("J") = 1.0,
          "H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})");

    m.def("build_ising_mpo",
          [](int n, double J, double h) {
              py::gil_scoped_release rel;
              return qforge::mpo::ising(n, J, h);
          }, py::arg("n_sites"), py::arg("J") = 1.0, py::arg("h") = 0.5,
          "H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i");

    m.def("build_xxz_mpo",
          [](int n, double Delta, double J) {
              py::gil_scoped_release rel;
              return qforge::mpo::xxz(n, Delta, J);
          }, py::arg("n_sites"), py::arg("Delta") = 1.0, py::arg("J") = 1.0,
          "H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1})");

    // ================================================================
    // DMRG sweep (module-level function)
    // ================================================================
    m.def("dmrg_sweep",
          [](qforge::mps::MPS& psi, const qforge::mpo::MPO& H,
             int max_chi, double eps, int lanczos_dim) {
              py::gil_scoped_release rel;
              return qforge::dmrg::dmrg_sweep(psi, H, max_chi, eps, lanczos_dim);
          },
          py::arg("psi"), py::arg("H"),
          py::arg("max_chi") = 32, py::arg("eps") = 1e-10,
          py::arg("lanczos_dim") = 20,
          "Run one full two-site DMRG sweep (left→right + right→left). Returns energy.");
}
