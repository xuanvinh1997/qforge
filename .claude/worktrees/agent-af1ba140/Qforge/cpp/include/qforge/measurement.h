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

}} // namespace qforge::measurement
