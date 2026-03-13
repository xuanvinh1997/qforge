#include "qforge/gates.h"
#include <cmath>
#include <stdexcept>

namespace qforge { namespace gates {

// Depolarizing channel: mixed probability/amplitude operation
// Operates on probabilities then takes sqrt to get amplitudes back
void E(StateVector& sv, double p_noise, int target) {
    const int n = sv.n_qubits();
    if (target < 0 || target >= n)
        throw std::out_of_range("Index is out of range");

    const size_t dim = sv.dim();
    const size_t mask = size_t(1) << (n - target - 1);
    auto* amp = sv.data();
    auto* scratch = sv.scratch();

    // Phase 1: compute new probabilities using a pair-based loop.
    // Each outer iteration owns a disjoint (i, i|mask) pair — conflict-free,
    // so the inner body parallelises with no atomics and no zero_scratch() pass.
    const double keep = 1.0 - p_noise / 2.0;
    const double xfer = p_noise / 2.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dim > 4096)
#endif
    for (size_t lo = 0; lo < dim; lo += 2 * mask) {
        for (size_t i = lo; i < lo + mask; ++i) {
            double p0 = std::norm(amp[i]);
            double p1 = std::norm(amp[i | mask]);
            scratch[i]        = std::complex<double>(keep * p0 + xfer * p1, 0.0);
            scratch[i | mask] = std::complex<double>(xfer * p0 + keep * p1, 0.0);
        }
    }

    // Phase 2: take sqrt, preserving sign from original amplitude
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dim > 4096)
#endif
    for (size_t i = 0; i < dim; ++i) {
        double new_prob = scratch[i].real();
        double sign = (amp[i].real() < 0) ? -1.0 : 1.0;
        amp[i] = sign * std::sqrt(std::max(new_prob, 0.0));
    }
}

void E_all(StateVector& sv, double p_noise) {
    if (p_noise > 0) {
        for (int i = 0; i < sv.n_qubits(); ++i) {
            E(sv, p_noise, i);
        }
    }
}

}} // namespace qforge::gates
