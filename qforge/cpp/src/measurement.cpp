#include "qforge/measurement.h"
#include <cmath>
#include <stdexcept>

namespace qforge { namespace measurement {

double measure_one_prob0(const StateVector& sv, int qubit) {
    const int n = sv.n_qubits();
    if (qubit < 0 || qubit >= n)
        throw std::out_of_range("Index is out of range");

    const size_t dim = sv.dim();
    const size_t mask = size_t(1) << (n - qubit - 1);
    const auto* amp = sv.data();
    double prob_0 = 0.0;

    #pragma omp parallel for reduction(+:prob_0) schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        if (!(i & mask)) {
            prob_0 += std::norm(amp[i]);
        }
    }
    return prob_0;
}

void collapse_one(StateVector& sv, int qubit, int value) {
    const int n = sv.n_qubits();
    if (qubit < 0 || qubit >= n)
        throw std::out_of_range("Index is out of range");
    if (value != 0 && value != 1)
        throw std::invalid_argument("value must be 0 or 1");

    const size_t dim = sv.dim();
    const size_t mask = size_t(1) << (n - qubit - 1);
    auto* amp = sv.data();

    // Compute probability of the target value
    double prob = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        bool bit_is_one = (i & mask) != 0;
        if ((value == 0 && !bit_is_one) || (value == 1 && bit_is_one)) {
            prob += std::norm(amp[i]);
        }
    }

    if (prob < 1e-15)
        throw std::runtime_error("Cannot collapse to a state with zero probability");

    double norm_factor = 1.0 / std::sqrt(prob);

    // Normalize matching states, zero out non-matching
    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        bool bit_is_one = (i & mask) != 0;
        if ((value == 0 && !bit_is_one) || (value == 1 && bit_is_one)) {
            amp[i] *= norm_factor;
        } else {
            amp[i] = {0.0, 0.0};
        }
    }
}

double pauli_expectation(const StateVector& sv, int qubit, int pauli_type) {
    const int n = sv.n_qubits();
    if (qubit < 0 || qubit >= n)
        throw std::out_of_range("Index is out of range");

    const size_t dim = sv.dim();
    const size_t mask = size_t(1) << (n - qubit - 1);
    const auto* amp = sv.data();
    double expectation = 0.0;

    if (pauli_type == 2) {
        // Z: <Z> = P(|0>) - P(|1>)
        #pragma omp parallel for reduction(+:expectation) schedule(static) if(dim > 4096)
        for (size_t i = 0; i < dim; ++i) {
            double prob = std::norm(amp[i]);
            if (!(i & mask))
                expectation += prob;
            else
                expectation -= prob;
        }
    } else if (pauli_type == 0) {
        // X: <X> = 2 * Re(sum conj(amp[i]) * amp[j]) for pairs where bit differs
        #pragma omp parallel for reduction(+:expectation) schedule(static) if(dim > 4096)
        for (size_t i = 0; i < dim; ++i) {
            if (!(i & mask)) {
                size_t j = i | mask;
                expectation += 2.0 * (amp[i].real() * amp[j].real() + amp[i].imag() * amp[j].imag());
            }
        }
    } else if (pauli_type == 1) {
        // Y: <Y> = 2 * Re(-i * conj(amp[i]) * amp[j])
        //        = 2 * (amp[i].real()*amp[j].imag() - amp[i].imag()*amp[j].real())
        #pragma omp parallel for reduction(+:expectation) schedule(static) if(dim > 4096)
        for (size_t i = 0; i < dim; ++i) {
            if (!(i & mask)) {
                size_t j = i | mask;
                expectation += 2.0 * (amp[i].real() * amp[j].imag() - amp[i].imag() * amp[j].real());
            }
        }
    } else {
        throw std::invalid_argument("pauli_type must be 0(X), 1(Y), or 2(Z)");
    }

    return expectation;
}

void probabilities(const StateVector& sv, double* out) {
    const size_t dim = sv.dim();
    const auto* amp = sv.data();

    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        out[i] = std::norm(amp[i]);
    }
}

// --- Qudit measurement (any d) ---

std::vector<double> measure_qudit_probs(const StateVector& sv, int qudit) {
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (qudit < 0 || qudit >= n)
        throw std::out_of_range("qudit index out of range");

    const size_t dim = sv.dim();
    const size_t stride = sv.stride(qudit);
    const auto* amp = sv.data();

    std::vector<double> probs(d, 0.0);

    for (size_t i = 0; i < dim; ++i) {
        int val = extract_qudit(i, stride, d);
        probs[val] += std::norm(amp[i]);
    }
    return probs;
}

void collapse_qudit(StateVector& sv, int qudit, int value) {
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (qudit < 0 || qudit >= n)
        throw std::out_of_range("qudit index out of range");
    if (value < 0 || value >= d)
        throw std::invalid_argument("value must be in [0, d)");

    const size_t dim = sv.dim();
    const size_t stride = sv.stride(qudit);
    auto* amp = sv.data();

    // Compute probability of the target value
    double prob = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        if (extract_qudit(i, stride, d) == value)
            prob += std::norm(amp[i]);
    }

    if (prob < 1e-15)
        throw std::runtime_error("Cannot collapse to a state with zero probability");

    double norm_factor = 1.0 / std::sqrt(prob);

    // Normalize matching states, zero out non-matching
    #pragma omp parallel for schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        if (extract_qudit(i, stride, d) == value)
            amp[i] *= norm_factor;
        else
            amp[i] = {0.0, 0.0};
    }
}

std::complex<double> qudit_expectation(const StateVector& sv, int qudit,
    const std::complex<double>* op)
{
    const int n = sv.n_qudits();
    const int d = sv.dimension();
    if (qudit < 0 || qudit >= n)
        throw std::out_of_range("qudit index out of range");

    const size_t dim = sv.dim();
    const size_t stride = sv.stride(qudit);
    const auto* amp = sv.data();

    // <psi| O |psi> = sum_{i} sum_{s,t} conj(a_{i,s}) * O_{s,t} * a_{i,t}
    // where i runs over all other qudits, s and t are the target qudit values
    std::complex<double> result = {0.0, 0.0};

    for (size_t idx = 0; idx < dim; ++idx) {
        int s = extract_qudit(idx, stride, d);
        if (s != 0) continue;  // canonical: only process when target qudit = 0

        for (int row = 0; row < d; ++row) {
            size_t idx_row = replace_qudit(idx, stride, 0, row);
            for (int col = 0; col < d; ++col) {
                size_t idx_col = replace_qudit(idx, stride, 0, col);
                result += std::conj(amp[idx_row]) * op[row * d + col] * amp[idx_col];
            }
        }
    }

    return result;
}

}} // namespace qforge::measurement
