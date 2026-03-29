#pragma once
#include "state_vector.h"
#include <vector>

namespace qforge { namespace measurement {

// Get probability of qubit being |0>
double measure_one_prob0(const StateVector& sv, int qubit);

// Collapse qubit to value (0 or 1), normalize
void collapse_one(StateVector& sv, int qubit, int value);

// Pauli expectation value for a single qubit
// pauli_type: 0=X, 1=Y, 2=Z
double pauli_expectation(const StateVector& sv, int qubit, int pauli_type);

// Compute |amplitude|^2 for all states, write to output array
void probabilities(const StateVector& sv, double* out);

// --- Qudit measurement (any d) ---

/// Return probabilities [P(0), P(1), ..., P(d-1)] for a single qudit.
std::vector<double> measure_qudit_probs(const StateVector& sv, int qudit);

/// Collapse qudit to value in [0, d), normalize.
void collapse_qudit(StateVector& sv, int qudit, int value);

/// Expectation value of a d×d Hermitian operator on a single qudit.
/// `op` is row-major d×d.
std::complex<double> qudit_expectation(const StateVector& sv, int qudit,
    const std::complex<double>* op);

}} // namespace qforge::measurement
