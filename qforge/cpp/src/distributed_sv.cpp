// -*- coding: utf-8 -*-
// distributed_sv.cpp — MPI-distributed state vector implementation
//
// Each rank owns local_dim = 2^(n-k) amplitudes, where k = log2(n_ranks).
// Qubit 0 = MSB. Global qubits [0..k-1] address the rank.
// Local qubits [k..n-1] are within each rank's shard.

#include "qforge/distributed_backend.h"
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qforge { namespace distributed {

static constexpr double INV_SQRT2 = 0.70710678118654752440;

// Aligned allocation
static C* aligned_alloc_complex(size_t count) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(count * sizeof(C), 64);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, 64, count * sizeof(C)) != 0)
        throw std::bad_alloc();
#endif
    return static_cast<C*>(ptr);
}

static void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int ilog2(int v) {
    int r = 0;
    while ((1 << r) < v) ++r;
    return r;
}

DistributedStateVector::DistributedStateVector(int n_qubits
#ifdef QFORGE_USE_MPI
    , MPI_Comm comm
#endif
)
    : n_qubits_(n_qubits), local_amp_(nullptr), scratch_(nullptr)
{
#ifdef QFORGE_USE_MPI
    comm_ = comm;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &n_ranks_);
#else
    rank_ = 0;
    n_ranks_ = 1;
#endif

    if (n_ranks_ < 1 || (n_ranks_ & (n_ranks_ - 1)) != 0)
        throw std::invalid_argument("n_ranks must be a power of 2");

    n_global_ = ilog2(n_ranks_);
    n_local_ = n_qubits_ - n_global_;
    if (n_local_ < 1)
        throw std::invalid_argument("Too many ranks for qubit count");

    local_dim_ = size_t(1) << n_local_;
    local_amp_ = aligned_alloc_complex(local_dim_);
    scratch_ = aligned_alloc_complex(local_dim_);

    reset();
}

DistributedStateVector::~DistributedStateVector() {
    if (local_amp_) aligned_free(local_amp_);
    if (scratch_) aligned_free(scratch_);
}

void DistributedStateVector::reset() {
    std::memset(local_amp_, 0, local_dim_ * sizeof(C));
    if (rank_ == 0)
        local_amp_[0] = {1.0, 0.0};
}

// ============================================================
// Local single-qubit gate (no MPI communication)
// Same kernel as gates_single.cpp
// ============================================================
void DistributedStateVector::apply_local_single_gate(
    int local_target, C m00, C m01, C m10, C m11
) {
    const size_t mask = size_t(1) << (n_local_ - local_target - 1);

#ifdef _OPENMP
    const bool use_omp = local_dim_ > 4096;
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (size_t lo = 0; lo < local_dim_; lo += 2 * mask) {
        for (size_t i = lo; i < lo + mask; ++i) {
            size_t j = i | mask;
            C a0 = local_amp_[i];
            C a1 = local_amp_[j];
            local_amp_[i] = m00 * a0 + m01 * a1;
            local_amp_[j] = m10 * a0 + m11 * a1;
        }
    }
}

// ============================================================
// Global single-qubit gate (MPI pairwise exchange)
// ============================================================
void DistributedStateVector::apply_global_single_gate(
    int global_target, C m00, C m01, C m10, C m11
) {
    // Partner rank differs in bit `global_target`
    // Global qubit 0 is MSB of rank index (big-endian)
    const int partner_bit = 1 << (n_global_ - global_target - 1);
    const int partner_rank = rank_ ^ partner_bit;

    // Determine if this rank has bit=0 or bit=1 for this qubit
    const bool bit_is_zero = (rank_ & partner_bit) == 0;

#ifdef QFORGE_USE_MPI
    // Exchange all local amplitudes with partner
    MPI_Sendrecv(
        local_amp_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        partner_rank, 0,
        scratch_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        partner_rank, 0,
        comm_, MPI_STATUS_IGNORE
    );

    // Apply 2x2 gate: this rank has |0> or |1> block, partner has the other
    if (bit_is_zero) {
        // This rank has |0> amplitudes, scratch has |1> from partner
        #ifdef _OPENMP
        #pragma omp parallel for if(local_dim_ > 4096) schedule(static)
        #endif
        for (size_t i = 0; i < local_dim_; ++i) {
            C a0 = local_amp_[i];
            C a1 = scratch_[i];
            local_amp_[i] = m00 * a0 + m01 * a1;
        }
    } else {
        // This rank has |1> amplitudes, scratch has |0> from partner
        #ifdef _OPENMP
        #pragma omp parallel for if(local_dim_ > 4096) schedule(static)
        #endif
        for (size_t i = 0; i < local_dim_; ++i) {
            C a0 = scratch_[i];
            C a1 = local_amp_[i];
            local_amp_[i] = m10 * a0 + m11 * a1;
        }
    }
#else
    // Without MPI, single rank — all qubits are local
    // This shouldn't happen (n_global_ == 0 means no global qubits)
    (void)partner_bit; (void)partner_rank; (void)bit_is_zero;
    (void)m00; (void)m01; (void)m10; (void)m11;
#endif
}

void DistributedStateVector::apply_single_gate(
    int target, C m00, C m01, C m10, C m11
) {
    if (is_local(target))
        apply_local_single_gate(local_index(target), m00, m01, m10, m11);
    else
        apply_global_single_gate(target, m00, m01, m10, m11);
}

// ============================================================
// Local controlled gate
// ============================================================
void DistributedStateVector::apply_local_controlled_gate(
    int ctrl_local, int tgt_local, C m00, C m01, C m10, C m11
) {
    const size_t ctrl_mask = size_t(1) << (n_local_ - ctrl_local - 1);
    const size_t tgt_mask  = size_t(1) << (n_local_ - tgt_local - 1);

#ifdef _OPENMP
    #pragma omp parallel for if(local_dim_ > 4096) schedule(static)
#endif
    for (size_t idx = 0; idx < local_dim_; ++idx) {
        if ((idx & ctrl_mask) && !(idx & tgt_mask)) {
            size_t j = idx | tgt_mask;
            C a0 = local_amp_[idx];
            C a1 = local_amp_[j];
            local_amp_[idx] = m00 * a0 + m01 * a1;
            local_amp_[j]   = m10 * a0 + m11 * a1;
        }
    }
}

void DistributedStateVector::apply_controlled_gate(
    int control, int target, C m00, C m01, C m10, C m11
) {
    bool ctrl_local = is_local(control);
    bool tgt_local  = is_local(target);

    if (ctrl_local && tgt_local) {
        apply_local_controlled_gate(
            local_index(control), local_index(target),
            m00, m01, m10, m11);
    } else if (ctrl_local && !tgt_local) {
        // Control is local, target is global
        // Check if control qubit is |1> on this rank; if mixed, need gate
        // For each amplitude: if control=1, apply gate on global target
        // This requires exchange with partner rank for the target qubit
#ifdef QFORGE_USE_MPI
        const int partner_bit = 1 << (n_global_ - target - 1);
        const int partner_rank = rank_ ^ partner_bit;
        const bool tgt_zero = (rank_ & partner_bit) == 0;
        const size_t ctrl_mask = size_t(1) << (n_local_ - local_index(control) - 1);

        MPI_Sendrecv(
            local_amp_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
            partner_rank, 0,
            scratch_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
            partner_rank, 0,
            comm_, MPI_STATUS_IGNORE
        );

        for (size_t i = 0; i < local_dim_; ++i) {
            if (!(i & ctrl_mask)) continue;  // control=0, skip
            if (tgt_zero) {
                C a0 = local_amp_[i];
                C a1 = scratch_[i];
                local_amp_[i] = m00 * a0 + m01 * a1;
            } else {
                C a0 = scratch_[i];
                C a1 = local_amp_[i];
                local_amp_[i] = m10 * a0 + m11 * a1;
            }
        }
#endif
    } else if (!ctrl_local && tgt_local) {
        // Control is global — check rank bit
        const int ctrl_bit = 1 << (n_global_ - control - 1);
        if (rank_ & ctrl_bit) {
            // Control=1 on this rank, apply gate locally on target
            apply_local_single_gate(local_index(target), m00, m01, m10, m11);
        }
        // Control=0: do nothing
    } else {
        // Both global — check control bit, then apply global gate on target
        const int ctrl_bit = 1 << (n_global_ - control - 1);
        if (rank_ & ctrl_bit) {
            apply_global_single_gate(target, m00, m01, m10, m11);
        }
    }
}

// ============================================================
// Named gates
// ============================================================
void DistributedStateVector::H(int t) {
    apply_single_gate(t, {INV_SQRT2,0}, {INV_SQRT2,0}, {INV_SQRT2,0}, {-INV_SQRT2,0});
}

void DistributedStateVector::X(int t) {
    apply_single_gate(t, {0,0}, {1,0}, {1,0}, {0,0});
}

void DistributedStateVector::Y(int t) {
    apply_single_gate(t, {0,0}, {0,-1}, {0,1}, {0,0});
}

void DistributedStateVector::Z(int t) {
    apply_single_gate(t, {1,0}, {0,0}, {0,0}, {-1,0});
}

void DistributedStateVector::RX(int t, double theta) {
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    apply_single_gate(t, {c,0}, {0,-s}, {0,-s}, {c,0});
}

void DistributedStateVector::RY(int t, double theta) {
    double c = std::cos(theta / 2), s = std::sin(theta / 2);
    apply_single_gate(t, {c,0}, {-s,0}, {s,0}, {c,0});
}

void DistributedStateVector::RZ(int t, double theta) {
    apply_single_gate(t, std::exp(C(0, -theta/2)), {0,0}, {0,0}, std::exp(C(0, theta/2)));
}

void DistributedStateVector::Phase(int t, double phi) {
    apply_single_gate(t, {1,0}, {0,0}, {0,0}, std::exp(C(0, phi)));
}

void DistributedStateVector::S(int t) {
    apply_single_gate(t, {1,0}, {0,0}, {0,0}, {0,1});
}

void DistributedStateVector::T(int t) {
    double c = INV_SQRT2;
    apply_single_gate(t, {1,0}, {0,0}, {0,0}, {c, c});
}

void DistributedStateVector::CNOT(int c, int t) {
    apply_controlled_gate(c, t, {0,0}, {1,0}, {1,0}, {0,0});
}

void DistributedStateVector::CRX(int c, int t, double theta) {
    double cs = std::cos(theta / 2), sn = std::sin(theta / 2);
    apply_controlled_gate(c, t, {cs,0}, {0,-sn}, {0,-sn}, {cs,0});
}

void DistributedStateVector::CRY(int c, int t, double theta) {
    double cs = std::cos(theta / 2), sn = std::sin(theta / 2);
    apply_controlled_gate(c, t, {cs,0}, {-sn,0}, {sn,0}, {cs,0});
}

void DistributedStateVector::CRZ(int c, int t, double theta) {
    apply_controlled_gate(c, t, std::exp(C(0,-theta/2)), {0,0}, {0,0}, std::exp(C(0,theta/2)));
}

void DistributedStateVector::CPhase(int c, int t, double phi) {
    apply_controlled_gate(c, t, {1,0}, {0,0}, {0,0}, std::exp(C(0, phi)));
}

// ============================================================
// Measurement
// ============================================================
double DistributedStateVector::allreduce_sum(double val) {
#ifdef QFORGE_USE_MPI
    double result;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return result;
#else
    return val;
#endif
}

double DistributedStateVector::local_prob0(int qubit) {
    double prob = 0.0;
    if (is_local(qubit)) {
        int lq = local_index(qubit);
        const size_t mask = size_t(1) << (n_local_ - lq - 1);
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:prob) if(local_dim_ > 4096)
        #endif
        for (size_t i = 0; i < local_dim_; ++i) {
            if (!(i & mask))
                prob += std::norm(local_amp_[i]);
        }
    } else {
        // Global qubit: check rank bit
        int gq = qubit;
        const int bit = 1 << (n_global_ - gq - 1);
        if (!(rank_ & bit)) {
            // This rank has qubit=0: sum all local amplitudes
            for (size_t i = 0; i < local_dim_; ++i)
                prob += std::norm(local_amp_[i]);
        }
    }
    return prob;
}

double DistributedStateVector::measure_one_prob0(int qubit) {
    return allreduce_sum(local_prob0(qubit));
}

void DistributedStateVector::collapse_one(int qubit, int value) {
    double prob = value == 0 ? measure_one_prob0(qubit) : (1.0 - measure_one_prob0(qubit));
    double scale = prob > 1e-14 ? 1.0 / std::sqrt(prob) : 0.0;

    if (is_local(qubit)) {
        int lq = local_index(qubit);
        const size_t mask = size_t(1) << (n_local_ - lq - 1);
        for (size_t i = 0; i < local_dim_; ++i) {
            bool bit_set = (i & mask) != 0;
            if ((value == 0 && bit_set) || (value == 1 && !bit_set))
                local_amp_[i] = {0.0, 0.0};
            else
                local_amp_[i] *= scale;
        }
    } else {
        int gq = qubit;
        const int bit = 1 << (n_global_ - gq - 1);
        bool rank_has_target = ((rank_ & bit) != 0) == (value == 1);
        if (rank_has_target) {
            for (size_t i = 0; i < local_dim_; ++i)
                local_amp_[i] *= scale;
        } else {
            std::memset(local_amp_, 0, local_dim_ * sizeof(C));
        }
    }
}

double DistributedStateVector::pauli_expectation(int qubit, int pauli_type) {
    if (pauli_type == 2) {
        // Z: prob0 - prob1 = 2*prob0 - 1
        double p0 = measure_one_prob0(qubit);
        return 2.0 * p0 - 1.0;
    }
    // X and Y require pair-wise amplitude access
    // For local qubits, straightforward; for global, needs exchange
    double result = 0.0;

    if (is_local(qubit)) {
        int lq = local_index(qubit);
        const size_t mask = size_t(1) << (n_local_ - lq - 1);

        for (size_t i = 0; i < local_dim_; ++i) {
            if (!(i & mask)) {
                size_t j = i | mask;
                if (pauli_type == 0) // X
                    result += 2.0 * std::real(std::conj(local_amp_[i]) * local_amp_[j]);
                else // Y
                    result += 2.0 * std::real(std::conj(local_amp_[i]) * C(0, -1) * local_amp_[j]);
            }
        }
        return allreduce_sum(result);
    }

    // Global qubit: exchange with partner
#ifdef QFORGE_USE_MPI
    const int partner_bit = 1 << (n_global_ - qubit - 1);
    const int partner_rank = rank_ ^ partner_bit;
    const bool bit_zero = (rank_ & partner_bit) == 0;

    MPI_Sendrecv(
        local_amp_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        partner_rank, 0,
        scratch_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        partner_rank, 0,
        comm_, MPI_STATUS_IGNORE
    );

    if (bit_zero) {
        for (size_t i = 0; i < local_dim_; ++i) {
            if (pauli_type == 0)
                result += 2.0 * std::real(std::conj(local_amp_[i]) * scratch_[i]);
            else
                result += 2.0 * std::real(std::conj(local_amp_[i]) * C(0, -1) * scratch_[i]);
        }
    }
    return allreduce_sum(result);
#else
    return 0.0;
#endif
}

void DistributedStateVector::gather_amplitudes(C* out) const {
#ifdef QFORGE_USE_MPI
    MPI_Gather(
        local_amp_, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        out, static_cast<int>(local_dim_), MPI_C_DOUBLE_COMPLEX,
        0, comm_
    );
#else
    std::memcpy(out, local_amp_, local_dim_ * sizeof(C));
#endif
}

}} // namespace qforge::distributed
