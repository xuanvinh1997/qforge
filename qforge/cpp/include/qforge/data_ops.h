#pragma once
#include <complex>
#include <cstddef>
#include <vector>

namespace qforge { namespace data_ops {

// PauliZ expectation values using bitwise parity
// probs: array of |amplitude|^2, dim: 2^n_qubits, n_qubits: number of qubits
double pauli_z_one_body(const double* probs, size_t dim, int n_qubits, int i);
double pauli_z_two_body(const double* probs, size_t dim, int n_qubits, int i, int j);
double pauli_z_three_body(const double* probs, size_t dim, int n_qubits, int i, int j, int k);
double pauli_z_four_body(const double* probs, size_t dim, int n_qubits, int i, int j, int k, int l);

// Reduced density matrix
// amp: state vector amplitudes, dim: 2^n_qubits
// keep_qubits: indices of qubits to keep
// rho_out: output matrix of size dim_k x dim_k where dim_k = 2^|keep_qubits|
void reduced_density_matrix(
    const std::complex<double>* amp, size_t dim, int n_qubits,
    const std::vector<int>& keep_qubits,
    std::complex<double>* rho_out, size_t dim_k);

}} // namespace qforge::data_ops
