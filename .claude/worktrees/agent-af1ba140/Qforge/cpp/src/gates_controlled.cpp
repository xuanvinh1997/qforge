#include "qforge/gates.h"
#include <cmath>
#include <stdexcept>

namespace qforge { namespace gates {

void apply_controlled_gate(StateVector& sv, int control, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    const int n = sv.n_qubits();
    if (control < 0 || control >= n || target < 0 || target >= n)
        throw std::out_of_range("Index is out of range");
    if (control == target)
        throw std::invalid_argument("Control qubit and target qubit must be distinct");

    const size_t dim = sv.dim();
    const size_t ctrl_mask = size_t(1) << (n - control - 1);
    const size_t tgt_mask  = size_t(1) << (n - target - 1);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t idx = 0; idx < dim; ++idx) {
        // Only process pairs where control=1 and target=0
        if ((idx & ctrl_mask) && !(idx & tgt_mask)) {
            size_t j = idx | tgt_mask;
            auto a0 = amp[idx];
            auto a1 = amp[j];
            amp[idx] = m00 * a0 + m01 * a1;
            amp[j]   = m10 * a0 + m11 * a1;
        }
    }
}

void apply_double_controlled_gate(StateVector& sv, int c1, int c2, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    const int n = sv.n_qubits();
    if (c1 < 0 || c1 >= n || c2 < 0 || c2 >= n || target < 0 || target >= n)
        throw std::out_of_range("Index is out of range");
    if (c1 == target || c2 == target || c1 == c2)
        throw std::invalid_argument("Control qubit and target qubit must be distinct");

    const size_t dim = sv.dim();
    const size_t c1_mask  = size_t(1) << (n - c1 - 1);
    const size_t c2_mask  = size_t(1) << (n - c2 - 1);
    const size_t tgt_mask = size_t(1) << (n - target - 1);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t idx = 0; idx < dim; ++idx) {
        // Only when both controls=1 and target=0
        if ((idx & c1_mask) && (idx & c2_mask) && !(idx & tgt_mask)) {
            size_t j = idx | tgt_mask;
            auto a0 = amp[idx];
            auto a1 = amp[j];
            amp[idx] = m00 * a0 + m01 * a1;
            amp[j]   = m10 * a0 + m11 * a1;
        }
    }
}

// CNOT: controlled-X = [[0,1],[1,0]] on target when control=1
void CNOT(StateVector& sv, int control, int target) {
    apply_controlled_gate(sv, control, target, 0.0, 1.0, 1.0, 0.0);
}

// CRX: controlled-RX
void CRX(StateVector& sv, int control, int target, double phi) {
    double c = std::cos(phi / 2.0);
    double s = std::sin(phi / 2.0);
    apply_controlled_gate(sv, control, target,
        {c, 0}, {0, -s},
        {0, -s}, {c, 0});
}

// CRY: controlled-RY
void CRY(StateVector& sv, int control, int target, double phi) {
    double c = std::cos(phi / 2.0);
    double s = std::sin(phi / 2.0);
    apply_controlled_gate(sv, control, target, c, -s, s, c);
}

// CRZ: controlled-RZ
void CRZ(StateVector& sv, int control, int target, double phi) {
    std::complex<double> em = std::exp(std::complex<double>(0, -phi / 2.0));
    std::complex<double> ep = std::exp(std::complex<double>(0, phi / 2.0));
    apply_controlled_gate(sv, control, target, em, 0.0, 0.0, ep);
}

// CPhase: controlled-Phase
void CPhase(StateVector& sv, int control, int target, double phi) {
    std::complex<double> ep = std::exp(std::complex<double>(0, phi));
    apply_controlled_gate(sv, control, target, 1.0, 0.0, 0.0, ep);
}

// CP: alias for CPhase
void CP(StateVector& sv, int control, int target, double phi) {
    CPhase(sv, control, target, phi);
}

// CCNOT: double-controlled-X (Toffoli)
void CCNOT(StateVector& sv, int c1, int c2, int target) {
    apply_double_controlled_gate(sv, c1, c2, target, 0.0, 1.0, 1.0, 0.0);
}

// OR: flip target if control_1 OR control_2 is |1>
void OR(StateVector& sv, int c1, int c2, int target) {
    const int n = sv.n_qubits();
    if (c1 < 0 || c1 >= n || c2 < 0 || c2 >= n || target < 0 || target >= n)
        throw std::out_of_range("Index is out of range");
    if (c1 == target || c2 == target || c1 == c2)
        throw std::invalid_argument("Control qubit and target qubit must be distinct");

    const size_t dim = sv.dim();
    const size_t c1_mask  = size_t(1) << (n - c1 - 1);
    const size_t c2_mask  = size_t(1) << (n - c2 - 1);
    const size_t tgt_mask = size_t(1) << (n - target - 1);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t idx = 0; idx < dim; ++idx) {
        // When either control=1 and target=0
        if (((idx & c1_mask) || (idx & c2_mask)) && !(idx & tgt_mask)) {
            size_t j = idx | tgt_mask;
            auto a0 = amp[idx];
            auto a1 = amp[j];
            // X gate on target: swap a0, a1
            amp[idx] = a1;
            amp[j]   = a0;
        }
    }
}

}} // namespace qforge::gates
