#include "qforge/cpu_backend.h"
#include "qforge/gates.h"
#include "qforge/measurement.h"

namespace qforge {

CpuBackend::CpuBackend(int n_qubits) : sv_(n_qubits) {}

// --- Single-qubit gates ---
void CpuBackend::apply_single_gate(int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    gates::apply_single_qubit_gate(sv_, target, m00, m01, m10, m11);
}

void CpuBackend::H(int target) { gates::H(sv_, target); }
void CpuBackend::X(int target) { gates::X(sv_, target); }
void CpuBackend::Y(int target) { gates::Y(sv_, target); }
void CpuBackend::Z(int target) { gates::Z(sv_, target); }
void CpuBackend::RX(int target, double phi) { gates::RX(sv_, target, phi); }
void CpuBackend::RY(int target, double phi) { gates::RY(sv_, target, phi); }
void CpuBackend::RZ(int target, double phi) { gates::RZ(sv_, target, phi); }
void CpuBackend::Phase(int target, double phi) { gates::Phase(sv_, target, phi); }
void CpuBackend::S(int target) { gates::S(sv_, target); }
void CpuBackend::T(int target) { gates::T(sv_, target); }
void CpuBackend::Xsquare(int target) { gates::Xsquare(sv_, target); }

// --- Controlled gates ---
void CpuBackend::apply_controlled_gate(int control, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    gates::apply_controlled_gate(sv_, control, target, m00, m01, m10, m11);
}

void CpuBackend::CNOT(int control, int target) { gates::CNOT(sv_, control, target); }
void CpuBackend::CRX(int control, int target, double phi) { gates::CRX(sv_, control, target, phi); }
void CpuBackend::CRY(int control, int target, double phi) { gates::CRY(sv_, control, target, phi); }
void CpuBackend::CRZ(int control, int target, double phi) { gates::CRZ(sv_, control, target, phi); }
void CpuBackend::CPhase(int control, int target, double phi) { gates::CPhase(sv_, control, target, phi); }
void CpuBackend::CP(int control, int target, double phi) { gates::CP(sv_, control, target, phi); }

// --- Double-controlled gates ---
void CpuBackend::apply_double_controlled_gate(int c1, int c2, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11) {
    gates::apply_double_controlled_gate(sv_, c1, c2, target, m00, m01, m10, m11);
}

void CpuBackend::CCNOT(int c1, int c2, int target) { gates::CCNOT(sv_, c1, c2, target); }
void CpuBackend::OR(int c1, int c2, int target) { gates::OR(sv_, c1, c2, target); }

// --- Swap gates ---
void CpuBackend::SWAP(int t1, int t2) { gates::SWAP(sv_, t1, t2); }
void CpuBackend::CSWAP(int control, int t1, int t2) { gates::CSWAP(sv_, control, t1, t2); }
void CpuBackend::ISWAP(int t1, int t2) { gates::ISWAP(sv_, t1, t2); }
void CpuBackend::SISWAP(int t1, int t2) { gates::SISWAP(sv_, t1, t2); }

// --- Noise ---
void CpuBackend::E(double p_noise, int target) { gates::E(sv_, p_noise, target); }
void CpuBackend::E_all(double p_noise) { gates::E_all(sv_, p_noise); }

// --- Measurement ---
double CpuBackend::measure_one_prob0(int qubit) const {
    return measurement::measure_one_prob0(sv_, qubit);
}
void CpuBackend::collapse_one(int qubit, int value) {
    measurement::collapse_one(sv_, qubit, value);
}
double CpuBackend::pauli_expectation(int qubit, int pauli_type) const {
    return measurement::pauli_expectation(sv_, qubit, pauli_type);
}
void CpuBackend::probabilities(double* out) const {
    measurement::probabilities(sv_, out);
}

} // namespace qforge
