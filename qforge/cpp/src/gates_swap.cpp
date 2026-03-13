#include "qforge/gates.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace qforge { namespace gates {

// SWAP: exchange amplitudes of states differing in t1,t2 bits
void SWAP(StateVector& sv, int t1, int t2) {
    const int n = sv.n_qubits();
    if (t1 < 0 || t1 >= n || t2 < 0 || t2 >= n)
        throw std::out_of_range("Index is out of range");
    if (t1 == t2)
        throw std::invalid_argument("Target qubits must be distinct");

    const size_t dim = sv.dim();
    const size_t m1 = size_t(1) << (n - t1 - 1);
    const size_t m2 = size_t(1) << (n - t2 - 1);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        // Only process canonical pair: bit at t1=0, bit at t2=1
        if (!(i & m1) && (i & m2)) {
            size_t j = i ^ m1 ^ m2;  // flip both bits
            std::swap(amp[i], amp[j]);
        }
    }
}

// CSWAP (Fredkin): controlled SWAP
void CSWAP(StateVector& sv, int control, int t1, int t2) {
    const int n = sv.n_qubits();
    if (control < 0 || control >= n || t1 < 0 || t1 >= n || t2 < 0 || t2 >= n)
        throw std::out_of_range("Index is out of range");
    if (control == t1 || control == t2 || t1 == t2)
        throw std::invalid_argument("Control qubit and target qubit must be distinct");

    const size_t dim = sv.dim();
    const size_t ctrl_mask = size_t(1) << (n - control - 1);
    const size_t m1 = size_t(1) << (n - t1 - 1);
    const size_t m2 = size_t(1) << (n - t2 - 1);
    auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        // control=1, canonical pair: t1 bit=0, t2 bit=1
        if ((i & ctrl_mask) && !(i & m1) && (i & m2)) {
            size_t j = i ^ m1 ^ m2;
            std::swap(amp[i], amp[j]);
        }
    }
}

// ISWAP: swap with i phase factor
// |00> -> |00>, |01> -> i|10>, |10> -> i|01>, |11> -> |11>
void ISWAP(StateVector& sv, int t1, int t2) {
    const int n = sv.n_qubits();
    if (t1 < 0 || t1 >= n || t2 < 0 || t2 >= n)
        throw std::out_of_range("Index is out of range");
    if (t1 == t2)
        throw std::invalid_argument("Target qubits must be distinct");

    const size_t dim = sv.dim();
    const size_t m1 = size_t(1) << (n - t1 - 1);
    const size_t m2 = size_t(1) << (n - t2 - 1);
    auto* amp = sv.data();
    const std::complex<double> imag_unit(0.0, 1.0);

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        // canonical pair: t1 bit=0, t2 bit=1
        if (!(i & m1) && (i & m2)) {
            size_t j = i ^ m1 ^ m2;  // t1 bit=1, t2 bit=0
            auto a = amp[i];
            auto b = amp[j];
            amp[i] = imag_unit * b;
            amp[j] = imag_unit * a;
        }
    }
}

// SISWAP (sqrt of ISWAP):
// |00> -> |00>, |11> -> |11>
// |01> -> (1/sqrt2)|01> + (i/sqrt2)|10>
// |10> -> (i/sqrt2)|01> + (1/sqrt2)|10>
void SISWAP(StateVector& sv, int t1, int t2) {
    const int n = sv.n_qubits();
    if (t1 < 0 || t1 >= n || t2 < 0 || t2 >= n)
        throw std::out_of_range("Index is out of range");
    if (t1 == t2)
        throw std::invalid_argument("Target qubits must be distinct");

    const size_t dim = sv.dim();
    const size_t m1 = size_t(1) << (n - t1 - 1);
    const size_t m2 = size_t(1) << (n - t2 - 1);
    auto* amp = sv.data();
    constexpr double c = 0.7071067811865475244;  // 1/sqrt(2)
    const std::complex<double> ic(0.0, c);        // i/sqrt(2)

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        // canonical pair: t1 bit=0, t2 bit=1
        if (!(i & m1) && (i & m2)) {
            size_t j = i ^ m1 ^ m2;
            auto a = amp[i];   // |01> subspace
            auto b = amp[j];   // |10> subspace
            amp[i] = c * a + ic * b;
            amp[j] = ic * a + c * b;
        }
    }
}

}} // namespace qforge::gates
