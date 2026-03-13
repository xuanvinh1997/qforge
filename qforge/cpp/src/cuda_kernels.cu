/**
 * cuda_kernels.cu — Custom CUDA kernels for Qforge state-vector simulation.
 * No cuQuantum dependency: only cuda_runtime and cuComplex.
 *
 * Qubit convention (same as the rest of Qforge):
 *   Qubit index 0 is the MOST significant bit in the integer state index.
 *   For n_qubits=3, qubit 0 → bit 2, qubit 1 → bit 1, qubit 2 → bit 0.
 *   Callers pass bit positions already converted:
 *       bit = n_qubits - 1 - qubit
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdint.h>
#include <cstddef>

// ---------------------------------------------------------------------------
// Bit-manipulation helper: insert a 0 bit at position `bit` in `idx`.
// Bits [0..bit-1] stay in place; bits [bit..] shift up by 1.
// ---------------------------------------------------------------------------
__device__ __forceinline__ size_t insert_zero(size_t idx, int bit) {
    size_t lo = idx & ((1ULL << bit) - 1);
    size_t hi = idx >> bit;
    return (hi << (bit + 1)) | lo;
}

// ---------------------------------------------------------------------------
// 1. Single-qubit gate
//    Each thread handles one (i0, i1) amplitude pair.
//    half_dim = 2^(n_qubits-1)
// ---------------------------------------------------------------------------
__global__ void k_single(cuDoubleComplex* sv, int tgt_bit, size_t half,
                          cuDoubleComplex m00, cuDoubleComplex m01,
                          cuDoubleComplex m10, cuDoubleComplex m11) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half) return;

    size_t i0 = insert_zero(idx, tgt_bit);
    size_t i1 = i0 | (1ULL << tgt_bit);

    cuDoubleComplex a = sv[i0], b = sv[i1];
    sv[i0] = cuCadd(cuCmul(m00, a), cuCmul(m01, b));
    sv[i1] = cuCadd(cuCmul(m10, a), cuCmul(m11, b));
}

// ---------------------------------------------------------------------------
// 2. Controlled single-qubit gate (1 control)
//    quarter_dim = 2^(n_qubits-2)
//    Only acts on pairs where ctrl_bit == 1.
// ---------------------------------------------------------------------------
__global__ void k_controlled(cuDoubleComplex* sv,
                              int ctrl_bit, int tgt_bit, size_t quarter,
                              cuDoubleComplex m00, cuDoubleComplex m01,
                              cuDoubleComplex m10, cuDoubleComplex m11) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quarter) return;

    // Insert zeros at sorted bit positions
    int lo = ctrl_bit < tgt_bit ? ctrl_bit : tgt_bit;
    int hi = ctrl_bit < tgt_bit ? tgt_bit  : ctrl_bit;

    size_t tmp    = insert_zero(idx, lo);
    size_t i_base = insert_zero(tmp, hi);

    // ctrl=1, tgt=0/1
    size_t i0 = i_base | (1ULL << ctrl_bit);
    size_t i1 = i0     | (1ULL << tgt_bit);

    cuDoubleComplex a = sv[i0], b = sv[i1];
    sv[i0] = cuCadd(cuCmul(m00, a), cuCmul(m01, b));
    sv[i1] = cuCadd(cuCmul(m10, a), cuCmul(m11, b));
}

// ---------------------------------------------------------------------------
// 3. Double-controlled single-qubit gate (2 controls, e.g. Toffoli)
//    eighth_dim = 2^(n_qubits-3)
// ---------------------------------------------------------------------------
__global__ void k_double_controlled(cuDoubleComplex* sv,
                                     int c1_bit, int c2_bit, int tgt_bit,
                                     size_t eighth,
                                     cuDoubleComplex m00, cuDoubleComplex m01,
                                     cuDoubleComplex m10, cuDoubleComplex m11) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= eighth) return;

    // Sort the three bit positions
    int b0 = c1_bit, b1 = c2_bit, b2 = tgt_bit;
    if (b0 > b1) { int t = b0; b0 = b1; b1 = t; }
    if (b1 > b2) { int t = b1; b1 = b2; b2 = t; }
    if (b0 > b1) { int t = b0; b0 = b1; b1 = t; }

    size_t t1     = insert_zero(idx, b0);
    size_t t2     = insert_zero(t1,  b1);
    size_t i_base = insert_zero(t2,  b2);

    // c1=1, c2=1, tgt=0/1
    size_t i0 = i_base | (1ULL << c1_bit) | (1ULL << c2_bit);
    size_t i1 = i0     | (1ULL << tgt_bit);

    cuDoubleComplex a = sv[i0], b = sv[i1];
    sv[i0] = cuCadd(cuCmul(m00, a), cuCmul(m01, b));
    sv[i1] = cuCadd(cuCmul(m10, a), cuCmul(m11, b));
}

// ---------------------------------------------------------------------------
// 4. Two-qubit gate (4×4 matrix, no controls)
//    Used for ISWAP, SISWAP, etc.
//    quarter_dim = 2^(n_qubits-2)
//
//    Basis ordering: tgt1 is the MORE significant bit of the pair,
//    tgt2 is the LESS significant.  Row/col order: |00>,|01>,|10>,|11>.
//    This matches custatevec's target ordering (first listed = MSB).
// ---------------------------------------------------------------------------
__global__ void k_two_qubit(cuDoubleComplex* sv,
                             int tgt1_bit, int tgt2_bit, size_t quarter,
                             const cuDoubleComplex* mat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quarter) return;

    int lo = tgt1_bit < tgt2_bit ? tgt1_bit : tgt2_bit;
    int hi = tgt1_bit < tgt2_bit ? tgt2_bit : tgt1_bit;

    size_t tmp    = insert_zero(idx, lo);
    size_t i_base = insert_zero(tmp, hi);

    // State indices in the four-dimensional subspace
    size_t i00 = i_base;
    size_t i01 = i_base | (1ULL << tgt2_bit);
    size_t i10 = i_base | (1ULL << tgt1_bit);
    size_t i11 = i_base | (1ULL << tgt1_bit) | (1ULL << tgt2_bit);

    cuDoubleComplex s0 = sv[i00], s1 = sv[i01], s2 = sv[i10], s3 = sv[i11];

    // Row 0
    sv[i00] = cuCadd(cuCadd(cuCmul(mat[0],  s0), cuCmul(mat[1],  s1)),
                     cuCadd(cuCmul(mat[2],  s2), cuCmul(mat[3],  s3)));
    // Row 1
    sv[i01] = cuCadd(cuCadd(cuCmul(mat[4],  s0), cuCmul(mat[5],  s1)),
                     cuCadd(cuCmul(mat[6],  s2), cuCmul(mat[7],  s3)));
    // Row 2
    sv[i10] = cuCadd(cuCadd(cuCmul(mat[8],  s0), cuCmul(mat[9],  s1)),
                     cuCadd(cuCmul(mat[10], s2), cuCmul(mat[11], s3)));
    // Row 3
    sv[i11] = cuCadd(cuCadd(cuCmul(mat[12], s0), cuCmul(mat[13], s1)),
                     cuCadd(cuCmul(mat[14], s2), cuCmul(mat[15], s3)));
}

// ---------------------------------------------------------------------------
// 5. Probability-of-0 reduction (parallel sum)
//    Sums |amplitude|^2 for all states where qubit_bit == 0.
//    Uses a two-phase approach: per-block partial sums → host-side final sum.
// ---------------------------------------------------------------------------
__global__ void k_prob0(const cuDoubleComplex* sv, size_t dim,
                        int qubit_bit, double* out) {
    extern __shared__ double shmem[];

    size_t tid = threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + tid;

    double acc = 0.0;
    while (i < dim) {
        if (!(i & (1ULL << qubit_bit))) {
            double re = cuCreal(sv[i]);
            double im = cuCimag(sv[i]);
            acc += re * re + im * im;
        }
        i += stride;
    }
    shmem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shmem[0];
}

// ---------------------------------------------------------------------------
// 6. Pauli-Z expectation reduction
//    <Z> = Σ_i (bit==0 ? +|amp[i]|² : -|amp[i]|²)
// ---------------------------------------------------------------------------
__global__ void k_pauli_z(const cuDoubleComplex* sv, size_t dim,
                           int qubit_bit, double* out) {
    extern __shared__ double shmem[];
    size_t tid = threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + tid;

    double acc = 0.0;
    while (i < dim) {
        double re = cuCreal(sv[i]), im = cuCimag(sv[i]);
        double prob = re * re + im * im;
        acc += (i & (1ULL << qubit_bit)) ? -prob : prob;
        i += stride;
    }
    shmem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shmem[0];
}

// ---------------------------------------------------------------------------
// 7. Pauli-X expectation reduction
//    <X> = 2 * Re( Σ_{i: bit=0} conj(amp[i]) * amp[i|mask] )
//    Re(conj(a)*b) = a.re*b.re + a.im*b.im
// ---------------------------------------------------------------------------
__global__ void k_pauli_x(const cuDoubleComplex* sv, size_t dim,
                           size_t mask, double* out) {
    extern __shared__ double shmem[];
    size_t tid = threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + tid;

    double acc = 0.0;
    while (i < dim) {
        if (!(i & mask)) {
            cuDoubleComplex a = sv[i], b = sv[i | mask];
            acc += 2.0 * (cuCreal(a) * cuCreal(b) + cuCimag(a) * cuCimag(b));
        }
        i += stride;
    }
    shmem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shmem[0];
}

// ---------------------------------------------------------------------------
// 8. Pauli-Y expectation reduction
//    <Y> = 2 * Im( Σ_{i: bit=0} conj(amp[i]) * amp[i|mask] )
//    Im(conj(a)*b) = a.re*b.im - a.im*b.re
// ---------------------------------------------------------------------------
__global__ void k_pauli_y(const cuDoubleComplex* sv, size_t dim,
                           size_t mask, double* out) {
    extern __shared__ double shmem[];
    size_t tid = threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + tid;

    double acc = 0.0;
    while (i < dim) {
        if (!(i & mask)) {
            cuDoubleComplex a = sv[i], b = sv[i | mask];
            acc += 2.0 * (cuCreal(a) * cuCimag(b) - cuCimag(a) * cuCreal(b));
        }
        i += stride;
    }
    shmem[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shmem[0];
}

// ---------------------------------------------------------------------------
// 9a. Depolarizing channel — Phase 1: scatter probabilities into scratch.
//     Pair-based: each thread owns (i, i|mask) — no write conflicts, no atomics.
//     scratch[i]        = keep*|amp[i]|² + xfer*|amp[i|mask]|²
//     scratch[i|mask]   = xfer*|amp[i]|² + keep*|amp[i|mask]|²
// ---------------------------------------------------------------------------
__global__ void k_depolarize_scatter(const cuDoubleComplex* sv, double* scratch,
                                      size_t half, int qubit_bit,
                                      double keep, double xfer) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half) return;

    size_t i  = insert_zero(idx, qubit_bit);
    size_t i1 = i | (1ULL << qubit_bit);

    double re0 = cuCreal(sv[i]),  im0 = cuCimag(sv[i]);
    double re1 = cuCreal(sv[i1]), im1 = cuCimag(sv[i1]);
    double p0 = re0 * re0 + im0 * im0;
    double p1 = re1 * re1 + im1 * im1;

    scratch[i]  = keep * p0 + xfer * p1;
    scratch[i1] = xfer * p0 + keep * p1;
}

// ---------------------------------------------------------------------------
// 9b. Depolarizing channel — Phase 2: replace amplitudes with sqrt(new_prob),
//     preserving the sign of the original real part.
// ---------------------------------------------------------------------------
__global__ void k_depolarize_sqrt(cuDoubleComplex* sv, const double* scratch,
                                   size_t dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    double new_prob = scratch[i];
    double sign = (cuCreal(sv[i]) < 0.0) ? -1.0 : 1.0;
    double val = sign * sqrt(new_prob > 0.0 ? new_prob : 0.0);
    sv[i] = make_cuDoubleComplex(val, 0.0);
}

// ===========================================================================
// C-linkage launch wrappers (called from cuda_backend.cpp via extern "C")
// ===========================================================================
extern "C" {

static inline int grid(size_t n, int blk) {
    return (int)((n + blk - 1) / blk);
}

void qforge_launch_single(cuDoubleComplex* sv, int n_qubits, int tgt_qubit,
                         cuDoubleComplex m00, cuDoubleComplex m01,
                         cuDoubleComplex m10, cuDoubleComplex m11) {
    int tgt_bit = n_qubits - 1 - tgt_qubit;
    size_t half = 1ULL << (n_qubits - 1);
    const int blk = 256;
    k_single<<<grid(half, blk), blk>>>(sv, tgt_bit, half, m00, m01, m10, m11);
}

void qforge_launch_controlled(cuDoubleComplex* sv, int n_qubits,
                              int ctrl_qubit, int tgt_qubit,
                              cuDoubleComplex m00, cuDoubleComplex m01,
                              cuDoubleComplex m10, cuDoubleComplex m11) {
    int ctrl_bit = n_qubits - 1 - ctrl_qubit;
    int tgt_bit  = n_qubits - 1 - tgt_qubit;
    size_t quarter = 1ULL << (n_qubits - 2);
    const int blk = 256;
    k_controlled<<<grid(quarter, blk), blk>>>(sv, ctrl_bit, tgt_bit, quarter,
                                               m00, m01, m10, m11);
}

void qforge_launch_double_controlled(cuDoubleComplex* sv, int n_qubits,
                                    int c1_qubit, int c2_qubit, int tgt_qubit,
                                    cuDoubleComplex m00, cuDoubleComplex m01,
                                    cuDoubleComplex m10, cuDoubleComplex m11) {
    int c1_bit  = n_qubits - 1 - c1_qubit;
    int c2_bit  = n_qubits - 1 - c2_qubit;
    int tgt_bit = n_qubits - 1 - tgt_qubit;
    size_t eighth = 1ULL << (n_qubits - 3);
    const int blk = 256;
    k_double_controlled<<<grid(eighth, blk), blk>>>(sv, c1_bit, c2_bit, tgt_bit,
                                                     eighth, m00, m01, m10, m11);
}

void qforge_launch_two_qubit(cuDoubleComplex* sv, int n_qubits,
                            int tgt1_qubit, int tgt2_qubit,
                            const cuDoubleComplex* mat_device) {
    int tgt1_bit = n_qubits - 1 - tgt1_qubit;
    int tgt2_bit = n_qubits - 1 - tgt2_qubit;
    size_t quarter = 1ULL << (n_qubits - 2);
    const int blk = 256;
    k_two_qubit<<<grid(quarter, blk), blk>>>(sv, tgt1_bit, tgt2_bit, quarter,
                                              mat_device);
}

double qforge_launch_prob0(const cuDoubleComplex* sv, int n_qubits, int qubit) {
    int qubit_bit = n_qubits - 1 - qubit;
    size_t dim = 1ULL << n_qubits;
    const int blk = 256;
    int nblocks = grid(dim, blk);
    if (nblocks > 1024) nblocks = 1024;

    double* d_out;
    cudaMalloc(&d_out, nblocks * sizeof(double));
    k_prob0<<<nblocks, blk, blk * sizeof(double)>>>(sv, dim, qubit_bit, d_out);

    double h_out[1024];
    cudaMemcpy(h_out, d_out, nblocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    double prob0 = 0.0;
    for (int i = 0; i < nblocks; ++i) prob0 += h_out[i];
    return prob0;
}

// Reuses a caller-supplied device reduction buffer (avoids per-call cudaMalloc).
double qforge_launch_prob0_ex(const cuDoubleComplex* sv, int n_qubits, int qubit,
                             double* d_reduce_buf) {
    int qubit_bit = n_qubits - 1 - qubit;
    size_t dim = 1ULL << n_qubits;
    const int blk = 256;
    int nblocks = grid(dim, blk);
    if (nblocks > 1024) nblocks = 1024;

    k_prob0<<<nblocks, blk, blk * sizeof(double)>>>(sv, dim, qubit_bit, d_reduce_buf);

    double h_out[1024];
    cudaMemcpy(h_out, d_reduce_buf, nblocks * sizeof(double), cudaMemcpyDeviceToHost);

    double prob0 = 0.0;
    for (int i = 0; i < nblocks; ++i) prob0 += h_out[i];
    return prob0;
}

// pauli_type: 0 = X, 1 = Y, 2 = Z (matches CudaBackend::pauli_expectation convention)
double qforge_launch_pauli(const cuDoubleComplex* sv, int n_qubits, int qubit,
                          int pauli_type, double* d_reduce_buf) {
    int qubit_bit = n_qubits - 1 - qubit;
    size_t dim = 1ULL << n_qubits;
    size_t mask = 1ULL << qubit_bit;
    const int blk = 256;
    int nblocks = grid(dim, blk);
    if (nblocks > 1024) nblocks = 1024;

    if (pauli_type == 2) { // Z
        k_pauli_z<<<nblocks, blk, blk * sizeof(double)>>>(sv, dim, qubit_bit, d_reduce_buf);
    } else if (pauli_type == 0) { // X
        k_pauli_x<<<nblocks, blk, blk * sizeof(double)>>>(sv, dim, mask, d_reduce_buf);
    } else { // Y
        k_pauli_y<<<nblocks, blk, blk * sizeof(double)>>>(sv, dim, mask, d_reduce_buf);
    }

    double h_out[1024];
    cudaMemcpy(h_out, d_reduce_buf, nblocks * sizeof(double), cudaMemcpyDeviceToHost);

    double ev = 0.0;
    for (int i = 0; i < nblocks; ++i) ev += h_out[i];
    return ev;
}

// GPU depolarizing channel.  d_scratch must be at least dim doubles on device.
void qforge_launch_depolarize(cuDoubleComplex* sv, int n_qubits, int qubit,
                             double p_noise, double* d_scratch) {
    int qubit_bit = n_qubits - 1 - qubit;
    size_t dim  = 1ULL << n_qubits;
    size_t half = dim / 2;
    const int blk = 256;
    double keep = 1.0 - p_noise / 2.0;
    double xfer = p_noise / 2.0;

    k_depolarize_scatter<<<grid(half, blk), blk>>>(sv, d_scratch, half,
                                                    qubit_bit, keep, xfer);
    k_depolarize_sqrt<<<grid(dim, blk), blk>>>(sv, d_scratch, dim);
}

} // extern "C"
