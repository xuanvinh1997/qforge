#pragma once
// -*- coding: utf-8 -*-
// distributed_backend.h — MPI-distributed state vector for multi-node simulation
//
// Shards the 2^n amplitude vector across MPI ranks. Each rank holds a
// contiguous block of 2^(n-k) amplitudes where k = log2(n_ranks).
//
// Gates on "local" qubits (index >= k) operate without communication.
// Gates on "global" qubits (index < k) use MPI_Sendrecv with a partner rank.

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef QFORGE_USE_MPI
#include <mpi.h>
#endif

namespace qforge { namespace distributed {

using C = std::complex<double>;

class DistributedStateVector {
public:
    explicit DistributedStateVector(int n_qubits
#ifdef QFORGE_USE_MPI
        , MPI_Comm comm = MPI_COMM_WORLD
#endif
    );
    ~DistributedStateVector();

    // Disable copy (MPI resources)
    DistributedStateVector(const DistributedStateVector&) = delete;
    DistributedStateVector& operator=(const DistributedStateVector&) = delete;

    int n_qubits() const { return n_qubits_; }
    size_t local_dim() const { return local_dim_; }
    int rank() const { return rank_; }
    int n_ranks() const { return n_ranks_; }
    int n_global_qubits() const { return n_global_; }
    C* local_data() { return local_amp_; }
    const C* local_data() const { return local_amp_; }

    void reset();  // Reset to |00...0>

    // ---- Single-qubit gates ----
    void apply_single_gate(int target, C m00, C m01, C m10, C m11);
    void H(int target);
    void X(int target);
    void Y(int target);
    void Z(int target);
    void RX(int target, double theta);
    void RY(int target, double theta);
    void RZ(int target, double theta);
    void Phase(int target, double phi);
    void S(int target);
    void T(int target);

    // ---- Controlled gates ----
    void apply_controlled_gate(int control, int target,
                                C m00, C m01, C m10, C m11);
    void CNOT(int control, int target);
    void CRX(int control, int target, double theta);
    void CRY(int control, int target, double theta);
    void CRZ(int control, int target, double theta);
    void CPhase(int control, int target, double phi);

    // ---- Measurement ----
    double measure_one_prob0(int qubit);
    void collapse_one(int qubit, int value);
    double pauli_expectation(int qubit, int pauli_type);  // 0=X, 1=Y, 2=Z

    // ---- Full amplitude access (for testing) ----
    // Gathers full state to rank 0. Expensive!
    void gather_amplitudes(C* out) const;

private:
    int n_qubits_;
    int n_global_;       // log2(n_ranks) — qubits that span ranks
    int n_local_;        // n_qubits - n_global
    size_t local_dim_;   // 2^n_local
    C* local_amp_;       // 64-byte aligned
    C* scratch_;         // communication buffer
    int rank_;
    int n_ranks_;

#ifdef QFORGE_USE_MPI
    MPI_Comm comm_;
#endif

    // Local gate kernel (no MPI)
    void apply_local_single_gate(int local_target, C m00, C m01, C m10, C m11);
    // Global gate kernel (MPI communication)
    void apply_global_single_gate(int global_target, C m00, C m01, C m10, C m11);
    // Local controlled gate kernel
    void apply_local_controlled_gate(int ctrl_local, int tgt_local,
                                      C m00, C m01, C m10, C m11);

    bool is_local(int qubit) const { return qubit >= n_global_; }
    int local_index(int qubit) const { return qubit - n_global_; }

    double local_prob0(int qubit);
    double allreduce_sum(double val);
};

}} // namespace qforge::distributed
