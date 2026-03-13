#include "qforge/data_ops.h"
#include <cstring>
#include <stdexcept>
#include <bit>

namespace qforge { namespace data_ops {

// Helper: extract bits at given positions from index, pack into reduced index
static inline size_t extract_bits(size_t idx, const std::vector<int>& positions, int n_qubits) {
    size_t result = 0;
    for (size_t k = 0; k < positions.size(); ++k) {
        int q = positions[k];
        bool bit = (idx >> (n_qubits - q - 1)) & 1;
        if (bit) result |= size_t(1) << (positions.size() - k - 1);
    }
    return result;
}

// Helper: build bitmask for given qubit positions
static inline size_t build_mask(const std::vector<int>& qubits, int n_qubits) {
    size_t mask = 0;
    for (int q : qubits) {
        mask |= size_t(1) << (n_qubits - q - 1);
    }
    return mask;
}

// PauliZ n-body: expectation = sum_i probs[i] * (-1)^(popcount(i & qubit_mask))
static double pauli_z_generic(const double* probs, size_t dim, int n_qubits,
                              const std::vector<int>& qubits) {
    size_t mask = build_mask(qubits, n_qubits);
    double result = 0.0;

    #pragma omp parallel for reduction(+:result) schedule(static) if(dim > 4096)
    for (size_t i = 0; i < dim; ++i) {
        size_t bits = i & mask;
        int parity = __builtin_popcountll(bits) & 1;
        result += parity ? -probs[i] : probs[i];
    }
    return result;
}

double pauli_z_one_body(const double* probs, size_t dim, int n_qubits, int i) {
    return pauli_z_generic(probs, dim, n_qubits, {i});
}

double pauli_z_two_body(const double* probs, size_t dim, int n_qubits, int i, int j) {
    return pauli_z_generic(probs, dim, n_qubits, {i, j});
}

double pauli_z_three_body(const double* probs, size_t dim, int n_qubits, int i, int j, int k) {
    return pauli_z_generic(probs, dim, n_qubits, {i, j, k});
}

double pauli_z_four_body(const double* probs, size_t dim, int n_qubits, int i, int j, int k, int l) {
    return pauli_z_generic(probs, dim, n_qubits, {i, j, k, l});
}

void reduced_density_matrix(
    const std::complex<double>* amp, size_t dim, int n_qubits,
    const std::vector<int>& keep_qubits,
    std::complex<double>* rho_out, size_t dim_k)
{
    // Zero output
    std::memset(rho_out, 0, dim_k * dim_k * sizeof(std::complex<double>));

    // Build mask for traced-out qubits
    std::vector<int> trace_qubits;
    for (int q = 0; q < n_qubits; ++q) {
        bool kept = false;
        for (int kq : keep_qubits) {
            if (kq == q) { kept = true; break; }
        }
        if (!kept) trace_qubits.push_back(q);
    }

    size_t n_trace = trace_qubits.size();
    size_t trace_dim = size_t(1) << n_trace;

    // For each traced-out configuration, accumulate outer product of
    // reduced amplitudes
    // Group states by their traced-out bits
    // For each trace configuration t:
    //   For each keep configuration k:
    //     full_index = reconstruct(k, t)
    //     reduced_amp[k] = amp[full_index]
    //   rho += outer(reduced_amp, reduced_amp*)

    // Precompute bit positions
    std::vector<size_t> keep_masks(keep_qubits.size());
    for (size_t i = 0; i < keep_qubits.size(); ++i)
        keep_masks[i] = size_t(1) << (n_qubits - keep_qubits[i] - 1);

    std::vector<size_t> trace_masks(trace_qubits.size());
    for (size_t i = 0; i < trace_qubits.size(); ++i)
        trace_masks[i] = size_t(1) << (n_qubits - trace_qubits[i] - 1);

    // Helper to reconstruct full index from keep_idx and trace_idx
    auto reconstruct = [&](size_t keep_idx, size_t trace_idx) -> size_t {
        size_t full = 0;
        for (size_t i = 0; i < keep_qubits.size(); ++i) {
            if (keep_idx & (size_t(1) << (keep_qubits.size() - i - 1)))
                full |= keep_masks[i];
        }
        for (size_t i = 0; i < trace_qubits.size(); ++i) {
            if (trace_idx & (size_t(1) << (trace_qubits.size() - i - 1)))
                full |= trace_masks[i];
        }
        return full;
    };

    // Accumulate: for each trace configuration, build reduced vector and outer product
    for (size_t t = 0; t < trace_dim; ++t) {
        for (size_t ki = 0; ki < dim_k; ++ki) {
            size_t fi = reconstruct(ki, t);
            auto ai = amp[fi];
            if (ai == std::complex<double>(0.0, 0.0)) continue;
            for (size_t kj = 0; kj < dim_k; ++kj) {
                size_t fj = reconstruct(kj, t);
                rho_out[ki * dim_k + kj] += ai * std::conj(amp[fj]);
            }
        }
    }
}

}} // namespace qforge::data_ops
