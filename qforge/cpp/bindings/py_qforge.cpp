#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "qforge/state_vector.h"
#include "qforge/gates.h"
#include "qforge/measurement.h"
#include "qforge/data_ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_qforge_core, m) {
    m.doc() = "Qforge C++ acceleration engine";

    // --- StateVector ---
    py::class_<qforge::StateVector>(m, "StateVector")
        .def(py::init<int>(), py::arg("n_qubits"))
        .def(py::init<int, int>(), py::arg("n_qudits"), py::arg("dimension"))
        .def_property_readonly("n_qubits", &qforge::StateVector::n_qubits)
        .def_property_readonly("n_qudits", &qforge::StateVector::n_qudits)
        .def_property_readonly("dimension", &qforge::StateVector::dimension)
        .def_property_readonly("dim", &qforge::StateVector::dim)
        .def("reset", &qforge::StateVector::reset)
        // Zero-copy numpy array access
        .def_property("amplitude",
            [](qforge::StateVector& sv) -> py::array_t<std::complex<double>> {
                return py::array_t<std::complex<double>>(
                    {sv.dim()},
                    {sizeof(std::complex<double>)},
                    sv.data(),
                    py::cast(&sv)  // prevent GC of StateVector
                );
            },
            [](qforge::StateVector& sv, py::array_t<std::complex<double>> arr) {
                auto buf = arr.request();
                if (static_cast<size_t>(buf.size) != sv.dim())
                    throw std::runtime_error("Size mismatch");
                std::memcpy(sv.data(), buf.ptr, sv.dim() * sizeof(std::complex<double>));
            }
        )
        // --- Gate methods on StateVector (unified dispatch interface) ---
        .def("H", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::H(sv, t);
        }, py::arg("target"))
        .def("X", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::X(sv, t);
        }, py::arg("target"))
        .def("Y", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::Y(sv, t);
        }, py::arg("target"))
        .def("Z", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::Z(sv, t);
        }, py::arg("target"))
        .def("RX", [](qforge::StateVector& sv, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::RX(sv, t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("RY", [](qforge::StateVector& sv, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::RY(sv, t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("RZ", [](qforge::StateVector& sv, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::RZ(sv, t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("Phase", [](qforge::StateVector& sv, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::Phase(sv, t, phi);
        }, py::arg("target"), py::arg("phi") = 0.0)
        .def("S", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::S(sv, t);
        }, py::arg("target"))
        .def("T", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::T(sv, t);
        }, py::arg("target"))
        .def("Xsquare", [](qforge::StateVector& sv, int t) {
            py::gil_scoped_release release;
            qforge::gates::Xsquare(sv, t);
        }, py::arg("target"))
        .def("CNOT", [](qforge::StateVector& sv, int c, int t) {
            py::gil_scoped_release release;
            qforge::gates::CNOT(sv, c, t);
        }, py::arg("control"), py::arg("target"))
        .def("CRX", [](qforge::StateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::CRX(sv, c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CRY", [](qforge::StateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::CRY(sv, c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CRZ", [](qforge::StateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::CRZ(sv, c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CPhase", [](qforge::StateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::CPhase(sv, c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CP", [](qforge::StateVector& sv, int c, int t, double phi) {
            py::gil_scoped_release release;
            qforge::gates::CP(sv, c, t, phi);
        }, py::arg("control"), py::arg("target"), py::arg("phi") = 0.0)
        .def("CCNOT", [](qforge::StateVector& sv, int c1, int c2, int t) {
            py::gil_scoped_release release;
            qforge::gates::CCNOT(sv, c1, c2, t);
        }, py::arg("c1"), py::arg("c2"), py::arg("target"))
        .def("OR", [](qforge::StateVector& sv, int c1, int c2, int t) {
            py::gil_scoped_release release;
            qforge::gates::OR(sv, c1, c2, t);
        }, py::arg("c1"), py::arg("c2"), py::arg("target"))
        .def("SWAP", [](qforge::StateVector& sv, int t1, int t2) {
            py::gil_scoped_release release;
            qforge::gates::SWAP(sv, t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("CSWAP", [](qforge::StateVector& sv, int c, int t1, int t2) {
            py::gil_scoped_release release;
            qforge::gates::CSWAP(sv, c, t1, t2);
        }, py::arg("control"), py::arg("t1"), py::arg("t2"))
        .def("ISWAP", [](qforge::StateVector& sv, int t1, int t2) {
            py::gil_scoped_release release;
            qforge::gates::ISWAP(sv, t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("SISWAP", [](qforge::StateVector& sv, int t1, int t2) {
            py::gil_scoped_release release;
            qforge::gates::SISWAP(sv, t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("E", [](qforge::StateVector& sv, double p, int t) {
            py::gil_scoped_release release;
            qforge::gates::E(sv, p, t);
        }, py::arg("p_noise"), py::arg("target"))
        .def("E_all", [](qforge::StateVector& sv, double p) {
            py::gil_scoped_release release;
            qforge::gates::E_all(sv, p);
        }, py::arg("p_noise"))
        .def("measure_one_prob0", [](const qforge::StateVector& sv, int q) {
            py::gil_scoped_release release;
            return qforge::measurement::measure_one_prob0(sv, q);
        }, py::arg("qubit"))
        .def("collapse_one", [](qforge::StateVector& sv, int q, int v) {
            py::gil_scoped_release release;
            qforge::measurement::collapse_one(sv, q, v);
        }, py::arg("qubit"), py::arg("value"))
        .def("pauli_expectation", [](const qforge::StateVector& sv, int q, int pt) {
            py::gil_scoped_release release;
            return qforge::measurement::pauli_expectation(sv, q, pt);
        }, py::arg("qubit"), py::arg("pauli_type"))
        // --- Qudit gate/measurement methods ---
        .def("apply_single_qudit_gate",
            [](qforge::StateVector& sv, int target, py::array_t<std::complex<double>> gate) {
                auto buf = gate.request();
                int d = sv.dimension();
                if (buf.size != d * d)
                    throw std::runtime_error("gate matrix must be d*d elements");
                py::gil_scoped_release release;
                qforge::gates::apply_single_qudit_gate(sv, target,
                    static_cast<const std::complex<double>*>(buf.ptr));
            }, py::arg("target"), py::arg("gate"))
        .def("apply_controlled_qudit_gate",
            [](qforge::StateVector& sv, int control, int ctrl_val, int target,
               py::array_t<std::complex<double>> gate) {
                auto buf = gate.request();
                int d = sv.dimension();
                if (buf.size != d * d)
                    throw std::runtime_error("gate matrix must be d*d elements");
                py::gil_scoped_release release;
                qforge::gates::apply_controlled_qudit_gate(sv, control, ctrl_val, target,
                    static_cast<const std::complex<double>*>(buf.ptr));
            }, py::arg("control"), py::arg("ctrl_val"), py::arg("target"), py::arg("gate"))
        .def("qudit_swap", [](qforge::StateVector& sv, int t1, int t2) {
            py::gil_scoped_release release;
            qforge::gates::qudit_swap(sv, t1, t2);
        }, py::arg("t1"), py::arg("t2"))
        .def("csum", [](qforge::StateVector& sv, int control, int target) {
            py::gil_scoped_release release;
            qforge::gates::csum(sv, control, target);
        }, py::arg("control"), py::arg("target"))
        .def("measure_qudit_probs", [](const qforge::StateVector& sv, int qudit) {
            py::gil_scoped_release release;
            return qforge::measurement::measure_qudit_probs(sv, qudit);
        }, py::arg("qudit"))
        .def("collapse_qudit", [](qforge::StateVector& sv, int qudit, int value) {
            py::gil_scoped_release release;
            qforge::measurement::collapse_qudit(sv, qudit, value);
        }, py::arg("qudit"), py::arg("value"))
        .def("qudit_expectation",
            [](const qforge::StateVector& sv, int qudit,
               py::array_t<std::complex<double>> op) {
                auto buf = op.request();
                int d = sv.dimension();
                if (buf.size != d * d)
                    throw std::runtime_error("operator matrix must be d*d elements");
                py::gil_scoped_release release;
                return qforge::measurement::qudit_expectation(sv, qudit,
                    static_cast<const std::complex<double>*>(buf.ptr));
            }, py::arg("qudit"), py::arg("op"))
    ;

    // --- Module-level gate functions (backward compat) ---
    m.def("H", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::H(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("X", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::X(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("Y", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::Y(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("Z", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::Z(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("RX", [](qforge::StateVector& sv, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::RX(sv, t, phi);
    }, py::arg("sv"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("RY", [](qforge::StateVector& sv, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::RY(sv, t, phi);
    }, py::arg("sv"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("RZ", [](qforge::StateVector& sv, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::RZ(sv, t, phi);
    }, py::arg("sv"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("Phase", [](qforge::StateVector& sv, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::Phase(sv, t, phi);
    }, py::arg("sv"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("S", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::S(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("T", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::T(sv, t);
    }, py::arg("sv"), py::arg("target"));

    m.def("Xsquare", [](qforge::StateVector& sv, int t) {
        py::gil_scoped_release release;
        qforge::gates::Xsquare(sv, t);
    }, py::arg("sv"), py::arg("target"));

    // --- Controlled gates ---
    m.def("CNOT", [](qforge::StateVector& sv, int c, int t) {
        py::gil_scoped_release release;
        qforge::gates::CNOT(sv, c, t);
    }, py::arg("sv"), py::arg("control"), py::arg("target"));

    m.def("CRX", [](qforge::StateVector& sv, int c, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::CRX(sv, c, t, phi);
    }, py::arg("sv"), py::arg("control"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("CRY", [](qforge::StateVector& sv, int c, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::CRY(sv, c, t, phi);
    }, py::arg("sv"), py::arg("control"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("CRZ", [](qforge::StateVector& sv, int c, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::CRZ(sv, c, t, phi);
    }, py::arg("sv"), py::arg("control"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("CPhase", [](qforge::StateVector& sv, int c, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::CPhase(sv, c, t, phi);
    }, py::arg("sv"), py::arg("control"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("CP", [](qforge::StateVector& sv, int c, int t, double phi) {
        py::gil_scoped_release release;
        qforge::gates::CP(sv, c, t, phi);
    }, py::arg("sv"), py::arg("control"), py::arg("target"), py::arg("phi") = 0.0);

    m.def("CCNOT", [](qforge::StateVector& sv, int c1, int c2, int t) {
        py::gil_scoped_release release;
        qforge::gates::CCNOT(sv, c1, c2, t);
    }, py::arg("sv"), py::arg("c1"), py::arg("c2"), py::arg("target"));

    m.def("OR", [](qforge::StateVector& sv, int c1, int c2, int t) {
        py::gil_scoped_release release;
        qforge::gates::OR(sv, c1, c2, t);
    }, py::arg("sv"), py::arg("c1"), py::arg("c2"), py::arg("target"));

    // --- Swap gates ---
    m.def("SWAP", [](qforge::StateVector& sv, int t1, int t2) {
        py::gil_scoped_release release;
        qforge::gates::SWAP(sv, t1, t2);
    }, py::arg("sv"), py::arg("t1"), py::arg("t2"));

    m.def("CSWAP", [](qforge::StateVector& sv, int c, int t1, int t2) {
        py::gil_scoped_release release;
        qforge::gates::CSWAP(sv, c, t1, t2);
    }, py::arg("sv"), py::arg("control"), py::arg("t1"), py::arg("t2"));

    m.def("ISWAP", [](qforge::StateVector& sv, int t1, int t2) {
        py::gil_scoped_release release;
        qforge::gates::ISWAP(sv, t1, t2);
    }, py::arg("sv"), py::arg("t1"), py::arg("t2"));

    m.def("SISWAP", [](qforge::StateVector& sv, int t1, int t2) {
        py::gil_scoped_release release;
        qforge::gates::SISWAP(sv, t1, t2);
    }, py::arg("sv"), py::arg("t1"), py::arg("t2"));

    // --- Noise ---
    m.def("E", [](qforge::StateVector& sv, double p, int t) {
        py::gil_scoped_release release;
        qforge::gates::E(sv, p, t);
    }, py::arg("sv"), py::arg("p_noise"), py::arg("target"));

    m.def("E_all", [](qforge::StateVector& sv, double p) {
        py::gil_scoped_release release;
        qforge::gates::E_all(sv, p);
    }, py::arg("sv"), py::arg("p_noise"));

    // --- Measurement ---
    m.def("measure_one_prob0", [](const qforge::StateVector& sv, int qubit) {
        py::gil_scoped_release release;
        return qforge::measurement::measure_one_prob0(sv, qubit);
    }, py::arg("sv"), py::arg("qubit"));

    m.def("collapse_one", [](qforge::StateVector& sv, int qubit, int value) {
        py::gil_scoped_release release;
        qforge::measurement::collapse_one(sv, qubit, value);
    }, py::arg("sv"), py::arg("qubit"), py::arg("value"));

    m.def("pauli_expectation", [](const qforge::StateVector& sv, int qubit, int pauli_type) {
        py::gil_scoped_release release;
        return qforge::measurement::pauli_expectation(sv, qubit, pauli_type);
    }, py::arg("sv"), py::arg("qubit"), py::arg("pauli_type"));

    m.def("probabilities", [](const qforge::StateVector& sv) {
        auto result = py::array_t<double>(sv.dim());
        auto buf = result.request();
        {
            py::gil_scoped_release release;
            qforge::measurement::probabilities(sv, static_cast<double*>(buf.ptr));
        }
        return result;
    }, py::arg("sv"));

    // --- Data operations ---
    auto validate_probs = [](const py::buffer_info& buf, int n_qubits) {
        size_t expected = size_t(1) << n_qubits;
        if (static_cast<size_t>(buf.size) != expected)
            throw std::runtime_error("probs size mismatch: expected " +
                std::to_string(expected) + ", got " + std::to_string(buf.size));
    };

    m.def("pauli_z_one_body", [validate_probs](py::array_t<double> probs, int n_qubits, int i) {
        auto buf = probs.request();
        validate_probs(buf, n_qubits);
        return qforge::data_ops::pauli_z_one_body(
            static_cast<double*>(buf.ptr), buf.size, n_qubits, i);
    }, py::arg("probs"), py::arg("n_qubits"), py::arg("i"));

    m.def("pauli_z_two_body", [validate_probs](py::array_t<double> probs, int n_qubits, int i, int j) {
        auto buf = probs.request();
        validate_probs(buf, n_qubits);
        return qforge::data_ops::pauli_z_two_body(
            static_cast<double*>(buf.ptr), buf.size, n_qubits, i, j);
    }, py::arg("probs"), py::arg("n_qubits"), py::arg("i"), py::arg("j"));

    m.def("pauli_z_three_body", [validate_probs](py::array_t<double> probs, int n_qubits, int i, int j, int k) {
        auto buf = probs.request();
        validate_probs(buf, n_qubits);
        return qforge::data_ops::pauli_z_three_body(
            static_cast<double*>(buf.ptr), buf.size, n_qubits, i, j, k);
    }, py::arg("probs"), py::arg("n_qubits"), py::arg("i"), py::arg("j"), py::arg("k"));

    m.def("pauli_z_four_body", [validate_probs](py::array_t<double> probs, int n_qubits, int i, int j, int k, int l) {
        auto buf = probs.request();
        validate_probs(buf, n_qubits);
        return qforge::data_ops::pauli_z_four_body(
            static_cast<double*>(buf.ptr), buf.size, n_qubits, i, j, k, l);
    }, py::arg("probs"), py::arg("n_qubits"), py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"));

    m.def("reduced_density_matrix",
        [](py::array_t<std::complex<double>> amp_arr, int n_qubits,
           std::vector<int> keep_qubits) {
            auto buf = amp_arr.request();
            size_t dim = buf.size;
            size_t dim_k = size_t(1) << keep_qubits.size();
            auto result = py::array_t<std::complex<double>>({dim_k, dim_k});
            auto rbuf = result.request();
            {
                py::gil_scoped_release release;
                qforge::data_ops::reduced_density_matrix(
                    static_cast<std::complex<double>*>(buf.ptr), dim, n_qubits,
                    keep_qubits,
                    static_cast<std::complex<double>*>(rbuf.ptr), dim_k);
            }
            return result;
        },
        py::arg("amplitudes"), py::arg("n_qubits"), py::arg("keep_qubits"));

    // --- Module-level qudit functions ---
    m.def("apply_single_qudit_gate",
        [](qforge::StateVector& sv, int target, py::array_t<std::complex<double>> gate) {
            auto buf = gate.request();
            int d = sv.dimension();
            if (buf.size != d * d)
                throw std::runtime_error("gate matrix must be d*d elements");
            py::gil_scoped_release release;
            qforge::gates::apply_single_qudit_gate(sv, target,
                static_cast<const std::complex<double>*>(buf.ptr));
        }, py::arg("sv"), py::arg("target"), py::arg("gate"));

    m.def("apply_controlled_qudit_gate",
        [](qforge::StateVector& sv, int control, int ctrl_val, int target,
           py::array_t<std::complex<double>> gate) {
            auto buf = gate.request();
            int d = sv.dimension();
            if (buf.size != d * d)
                throw std::runtime_error("gate matrix must be d*d elements");
            py::gil_scoped_release release;
            qforge::gates::apply_controlled_qudit_gate(sv, control, ctrl_val, target,
                static_cast<const std::complex<double>*>(buf.ptr));
        }, py::arg("sv"), py::arg("control"), py::arg("ctrl_val"),
           py::arg("target"), py::arg("gate"));

    m.def("qudit_swap", [](qforge::StateVector& sv, int t1, int t2) {
        py::gil_scoped_release release;
        qforge::gates::qudit_swap(sv, t1, t2);
    }, py::arg("sv"), py::arg("t1"), py::arg("t2"));

    m.def("csum", [](qforge::StateVector& sv, int control, int target) {
        py::gil_scoped_release release;
        qforge::gates::csum(sv, control, target);
    }, py::arg("sv"), py::arg("control"), py::arg("target"));

    m.def("measure_qudit_probs", [](const qforge::StateVector& sv, int qudit) {
        py::gil_scoped_release release;
        return qforge::measurement::measure_qudit_probs(sv, qudit);
    }, py::arg("sv"), py::arg("qudit"));

    m.def("collapse_qudit", [](qforge::StateVector& sv, int qudit, int value) {
        py::gil_scoped_release release;
        qforge::measurement::collapse_qudit(sv, qudit, value);
    }, py::arg("sv"), py::arg("qudit"), py::arg("value"));

    m.def("qudit_expectation",
        [](const qforge::StateVector& sv, int qudit,
           py::array_t<std::complex<double>> op) {
            auto buf = op.request();
            int d = sv.dimension();
            if (buf.size != d * d)
                throw std::runtime_error("operator matrix must be d*d elements");
            py::gil_scoped_release release;
            return qforge::measurement::qudit_expectation(sv, qudit,
                static_cast<const std::complex<double>*>(buf.ptr));
        }, py::arg("sv"), py::arg("qudit"), py::arg("op"));
}
