#include "qforge/gates.h"
#include <cmath>
#include <stdexcept>

namespace qforge { namespace gates {

void apply_single_qubit_gate(StateVector& sv, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    const int n = sv.n_qubits();
    if (target < 0 || target >= n)
        throw std::out_of_range("Index is out of range");

    const size_t dim = sv.dim();
    const size_t mask = size_t(1) << (n - target - 1);
    auto* amp = sv.data();

    // Iterate over (|0>, |1>) pairs for the target qubit
    // The outer loop steps by 2*mask, inner loop covers indices where target bit = 0
    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t lo = 0; lo < dim; lo += 2 * mask) {
        for (size_t i = lo; i < lo + mask; ++i) {
            size_t j = i | mask;  // same state but target bit = 1
            auto a0 = amp[i];
            auto a1 = amp[j];
            amp[i] = m00 * a0 + m01 * a1;
            amp[j] = m10 * a0 + m11 * a1;
        }
    }
}

// H = (1/sqrt2) * [[1, 1], [1, -1]]
void H(StateVector& sv, int target) {
    constexpr double r = 0.7071067811865475244;
    apply_single_qubit_gate(sv, target, r, r, r, -r);
}

// X = [[0, 1], [1, 0]]
void X(StateVector& sv, int target) {
    apply_single_qubit_gate(sv, target, 0.0, 1.0, 1.0, 0.0);
}

// Y = [[0, -i], [i, 0]]
void Y(StateVector& sv, int target) {
    apply_single_qubit_gate(sv, target,
        {0, 0}, {0, -1},
        {0, 1}, {0, 0});
}

// Z = [[1, 0], [0, -1]]
void Z(StateVector& sv, int target) {
    apply_single_qubit_gate(sv, target, 1.0, 0.0, 0.0, -1.0);
}

// RX(phi) = [[cos(phi/2), -i*sin(phi/2)], [-i*sin(phi/2), cos(phi/2)]]
void RX(StateVector& sv, int target, double phi) {
    double c = std::cos(phi / 2.0);
    double s = std::sin(phi / 2.0);
    apply_single_qubit_gate(sv, target,
        {c, 0}, {0, -s},
        {0, -s}, {c, 0});
}

// RY(phi) = [[cos(phi/2), -sin(phi/2)], [sin(phi/2), cos(phi/2)]]
void RY(StateVector& sv, int target, double phi) {
    double c = std::cos(phi / 2.0);
    double s = std::sin(phi / 2.0);
    apply_single_qubit_gate(sv, target, c, -s, s, c);
}

// RZ(phi) = [[exp(-i*phi/2), 0], [0, exp(i*phi/2)]]
void RZ(StateVector& sv, int target, double phi) {
    std::complex<double> em = std::exp(std::complex<double>(0, -phi / 2.0));
    std::complex<double> ep = std::exp(std::complex<double>(0, phi / 2.0));
    apply_single_qubit_gate(sv, target, em, 0.0, 0.0, ep);
}

// Phase(phi) = [[1, 0], [0, exp(i*phi)]]
void Phase(StateVector& sv, int target, double phi) {
    std::complex<double> ep = std::exp(std::complex<double>(0, phi));
    apply_single_qubit_gate(sv, target, 1.0, 0.0, 0.0, ep);
}

// S = Phase(pi/2) = [[1, 0], [0, i]]
void S(StateVector& sv, int target) {
    apply_single_qubit_gate(sv, target, 1.0, 0.0, 0.0, {0, 1});
}

// T = Phase(pi/4) = [[1, 0], [0, exp(i*pi/4)]]
void T(StateVector& sv, int target) {
    constexpr double r = 0.7071067811865475244;
    apply_single_qubit_gate(sv, target, 1.0, 0.0, 0.0, {r, r});
}

// Xsquare (sqrt NOT) = [[1+i, 1-i], [1-i, 1+i]] / 2
void Xsquare(StateVector& sv, int target) {
    apply_single_qubit_gate(sv, target,
        {0.5, 0.5}, {0.5, -0.5},
        {0.5, -0.5}, {0.5, 0.5});
}

}} // namespace qforge::gates
