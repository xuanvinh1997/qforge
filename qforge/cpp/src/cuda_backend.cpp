#include "qforge/cuda_backend.h"

#ifdef QFORGE_HAS_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Kernel launchers declared in cuda_kernels.cu (compiled by nvcc)
// ---------------------------------------------------------------------------
extern "C" {
void   qforge_launch_single(cuDoubleComplex* sv, int n_qubits, int tgt_qubit,
                            cuDoubleComplex m00, cuDoubleComplex m01,
                            cuDoubleComplex m10, cuDoubleComplex m11);
void   qforge_launch_controlled(cuDoubleComplex* sv, int n_qubits,
                                int ctrl_qubit, int tgt_qubit,
                                cuDoubleComplex m00, cuDoubleComplex m01,
                                cuDoubleComplex m10, cuDoubleComplex m11);
void   qforge_launch_double_controlled(cuDoubleComplex* sv, int n_qubits,
                                      int c1_qubit, int c2_qubit, int tgt_qubit,
                                      cuDoubleComplex m00, cuDoubleComplex m01,
                                      cuDoubleComplex m10, cuDoubleComplex m11);
void   qforge_launch_two_qubit(cuDoubleComplex* sv, int n_qubits,
                               int tgt1_qubit, int tgt2_qubit,
                               const cuDoubleComplex* mat_device);
double qforge_launch_prob0(const cuDoubleComplex* sv, int n_qubits, int qubit);
// Versions that reuse a caller-supplied 1024-double device reduction buffer:
double qforge_launch_prob0_ex(const cuDoubleComplex* sv, int n_qubits, int qubit,
                             double* d_reduce_buf);
double qforge_launch_pauli(const cuDoubleComplex* sv, int n_qubits, int qubit,
                          int pauli_type, double* d_reduce_buf);
// GPU depolarizing channel: d_scratch must be dim doubles on device.
void qforge_launch_depolarize(cuDoubleComplex* sv, int n_qubits, int qubit,
                             double p_noise, double* d_scratch);
}

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
} while(0)

// Helpers to cast between std::complex<double> and cuDoubleComplex
static inline cuDoubleComplex to_cu(std::complex<double> c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

namespace qforge {

// ============================================================
// Construction / Destruction
// ============================================================

CudaBackend::CudaBackend(int n_qubits)
    : n_qubits_(n_qubits),
      dim_(size_t(1) << n_qubits),
      d_sv_(nullptr),
      h_cache_(nullptr),
      host_dirty_(false),
      device_dirty_(false),
      d_mat2q_(nullptr),
      d_reduce_(nullptr),
      d_scratch_(nullptr),
      stream_(nullptr)
{
    if (n_qubits < 1 || n_qubits > 30)
        throw std::runtime_error("CudaBackend: n_qubits must be 1..30");

    h_cache_ = new std::complex<double>[dim_];
    CUDA_CHECK(cudaMalloc(&d_sv_, dim_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_mat2q_, 16 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_reduce_, 1024 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scratch_, dim_ * sizeof(double)));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = s;

    reset();
}

CudaBackend::~CudaBackend() {
    delete[] h_cache_;
    if (d_sv_)       cudaFree(d_sv_);
    if (d_mat2q_)    cudaFree(d_mat2q_);
    if (d_reduce_)   cudaFree(d_reduce_);
    if (d_scratch_)  cudaFree(d_scratch_);
    if (stream_)     cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ============================================================
// Data sync
// ============================================================

void CudaBackend::sync_to_host() {
    if (host_dirty_) {
        cudaStream_t s = static_cast<cudaStream_t>(stream_);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpyAsync(h_cache_, d_sv_,
                                   dim_ * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        host_dirty_ = false;
    }
}

void CudaBackend::sync_to_device() {
    if (device_dirty_) {
        cudaStream_t s = static_cast<cudaStream_t>(stream_);
        CUDA_CHECK(cudaMemcpyAsync(d_sv_, h_cache_,
                                   dim_ * sizeof(cuDoubleComplex),
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        device_dirty_ = false;
    }
}

std::complex<double>* CudaBackend::host_data() {
    sync_to_host();
    device_dirty_ = true;   // caller may write through the pointer
    return h_cache_;
}

const std::complex<double>* CudaBackend::host_data() const {
    const_cast<CudaBackend*>(this)->sync_to_host();
    return h_cache_;
}

void CudaBackend::reset() {
    std::memset(h_cache_, 0, dim_ * sizeof(std::complex<double>));
    h_cache_[0] = 1.0;
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_sv_, h_cache_,
                               dim_ * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    host_dirty_ = false;
    device_dirty_ = false;
}

// ============================================================
// Single-qubit gates
// ============================================================

void CudaBackend::apply_single_gate(int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    sync_to_device();
    qforge_launch_single(static_cast<cuDoubleComplex*>(d_sv_), n_qubits_, target,
                        to_cu(m00), to_cu(m01), to_cu(m10), to_cu(m11));
    host_dirty_ = true;
}

void CudaBackend::H(int t) {
    constexpr double r = 0.7071067811865475244;
    apply_single_gate(t, r, r, r, -r);
}
void CudaBackend::X(int t)  { apply_single_gate(t, 0, 1, 1, 0); }
void CudaBackend::Y(int t)  {
    using C = std::complex<double>;
    apply_single_gate(t, 0, C(0,-1), C(0,1), 0);
}
void CudaBackend::Z(int t)  { apply_single_gate(t, 1, 0, 0, -1); }

void CudaBackend::RX(int t, double phi) {
    using C = std::complex<double>;
    double c = std::cos(phi/2), s = std::sin(phi/2);
    apply_single_gate(t, c, C(0,-s), C(0,-s), c);
}
void CudaBackend::RY(int t, double phi) {
    double c = std::cos(phi/2), s = std::sin(phi/2);
    apply_single_gate(t, c, -s, s, c);
}
void CudaBackend::RZ(int t, double phi) {
    using C = std::complex<double>;
    apply_single_gate(t, std::exp(C(0,-phi/2)), 0, 0, std::exp(C(0,phi/2)));
}
void CudaBackend::Phase(int t, double phi) {
    using C = std::complex<double>;
    apply_single_gate(t, 1, 0, 0, std::exp(C(0,phi)));
}
void CudaBackend::S(int t) {
    using C = std::complex<double>;
    apply_single_gate(t, 1, 0, 0, C(0,1));
}
void CudaBackend::T(int t) {
    using C = std::complex<double>;
    constexpr double r = 0.7071067811865475244;
    apply_single_gate(t, 1, 0, 0, C(r,r));
}
void CudaBackend::Xsquare(int t) {
    using C = std::complex<double>;
    C a(0.5,0.5), b(0.5,-0.5);
    apply_single_gate(t, a, b, b, a);
}

// ============================================================
// Controlled gates
// ============================================================

void CudaBackend::apply_controlled_gate(int control, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    sync_to_device();
    qforge_launch_controlled(static_cast<cuDoubleComplex*>(d_sv_), n_qubits_,
                            control, target,
                            to_cu(m00), to_cu(m01), to_cu(m10), to_cu(m11));
    host_dirty_ = true;
}

void CudaBackend::CNOT(int c, int t)  { apply_controlled_gate(c, t, 0, 1, 1, 0); }
void CudaBackend::CRX(int c, int t, double phi) {
    using C = std::complex<double>;
    double cs = std::cos(phi/2), s = std::sin(phi/2);
    apply_controlled_gate(c, t, cs, C(0,-s), C(0,-s), cs);
}
void CudaBackend::CRY(int c, int t, double phi) {
    double cs = std::cos(phi/2), s = std::sin(phi/2);
    apply_controlled_gate(c, t, cs, -s, s, cs);
}
void CudaBackend::CRZ(int c, int t, double phi) {
    using C = std::complex<double>;
    apply_controlled_gate(c, t, std::exp(C(0,-phi/2)), 0, 0, std::exp(C(0,phi/2)));
}
void CudaBackend::CPhase(int c, int t, double phi) {
    using C = std::complex<double>;
    apply_controlled_gate(c, t, 1, 0, 0, std::exp(C(0,phi)));
}
void CudaBackend::CP(int c, int t, double phi) { CPhase(c, t, phi); }

// ============================================================
// Double-controlled gates
// ============================================================

void CudaBackend::apply_double_controlled_gate(int c1, int c2, int target,
    std::complex<double> m00, std::complex<double> m01,
    std::complex<double> m10, std::complex<double> m11)
{
    sync_to_device();
    qforge_launch_double_controlled(static_cast<cuDoubleComplex*>(d_sv_), n_qubits_,
                                   c1, c2, target,
                                   to_cu(m00), to_cu(m01), to_cu(m10), to_cu(m11));
    host_dirty_ = true;
}

void CudaBackend::CCNOT(int c1, int c2, int t) {
    apply_double_controlled_gate(c1, c2, t, 0, 1, 1, 0);
}
void CudaBackend::OR(int c1, int c2, int t) {
    X(c1); X(c2);
    CCNOT(c1, c2, t);
    X(c1); X(c2);
    X(t);
}

// ============================================================
// Swap gates
// ============================================================

void CudaBackend::SWAP(int t1, int t2) {
    CNOT(t1, t2); CNOT(t2, t1); CNOT(t1, t2);
}
void CudaBackend::CSWAP(int ctrl, int t1, int t2) {
    CNOT(t2, t1); CCNOT(ctrl, t1, t2); CNOT(t2, t1);
}

void CudaBackend::apply_two_qubit_gate(int t1, int t2,
                                        const std::complex<double>* mat4x4) {
    sync_to_device();
    // Reuse the persistent 128-byte matrix buffer — no per-call cudaMalloc/Free.
    cuDoubleComplex h_mat[16];
    for (int i = 0; i < 16; ++i)
        h_mat[i] = make_cuDoubleComplex(mat4x4[i].real(), mat4x4[i].imag());
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_mat2q_, h_mat, 16 * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, s));
    qforge_launch_two_qubit(static_cast<cuDoubleComplex*>(d_sv_), n_qubits_,
                           t1, t2, static_cast<const cuDoubleComplex*>(d_mat2q_));
    host_dirty_ = true;
}

void CudaBackend::ISWAP(int t1, int t2) {
    using C = std::complex<double>;
    const C mat[16] = {
        1, 0,      0,      0,
        0, 0,      C(0,1), 0,
        0, C(0,1), 0,      0,
        0, 0,      0,      1
    };
    apply_two_qubit_gate(t1, t2, mat);
}

void CudaBackend::SISWAP(int t1, int t2) {
    using C = std::complex<double>;
    constexpr double c = 0.7071067811865475244;
    const C mat[16] = {
        1, 0,      0,      0,
        0, c,      C(0,c), 0,
        0, C(0,c), c,      0,
        0, 0,      0,      1
    };
    apply_two_qubit_gate(t1, t2, mat);
}

// ============================================================
// Noise (host-side)
// ============================================================

void CudaBackend::E(double p_noise, int target) {
    sync_to_device();
    qforge_launch_depolarize(static_cast<cuDoubleComplex*>(d_sv_), n_qubits_,
                            target, p_noise,
                            static_cast<double*>(d_scratch_));
    host_dirty_ = true;
}

void CudaBackend::E_all(double p_noise) {
    if (p_noise > 0)
        for (int i = 0; i < n_qubits_; ++i) E(p_noise, i);
}

// ============================================================
// Measurement
// ============================================================

double CudaBackend::measure_one_prob0(int qubit) const {
    // Compute probability on GPU — no host sync required.
    const_cast<CudaBackend*>(this)->sync_to_device();
    return qforge_launch_prob0_ex(
        static_cast<const cuDoubleComplex*>(d_sv_), n_qubits_, qubit,
        static_cast<double*>(d_reduce_));
}

void CudaBackend::collapse_one(int qubit, int value) {
    sync_to_host();
    size_t mask = size_t(1) << (n_qubits_ - qubit - 1);
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
        bool bit_set = (i & mask) != 0;
        if ((value == 0 && bit_set) || (value == 1 && !bit_set))
            h_cache_[i] = 0;
        else
            norm_sq += std::norm(h_cache_[i]);
    }
    if (norm_sq > 0) {
        double inv = 1.0 / std::sqrt(norm_sq);
        for (size_t i = 0; i < dim_; ++i) h_cache_[i] *= inv;
    }
    device_dirty_ = true;
    host_dirty_ = false;
}

double CudaBackend::pauli_expectation(int qubit, int pauli_type) const {
    // Compute expectation on GPU — no host sync required.
    const_cast<CudaBackend*>(this)->sync_to_device();
    return qforge_launch_pauli(
        static_cast<const cuDoubleComplex*>(d_sv_), n_qubits_, qubit,
        pauli_type, static_cast<double*>(d_reduce_));
}

void CudaBackend::probabilities(double* out) const {
    const_cast<CudaBackend*>(this)->sync_to_host();
    for (size_t i = 0; i < dim_; ++i)
        out[i] = std::norm(h_cache_[i]);
}

} // namespace qforge

#else // !QFORGE_HAS_CUDA

namespace qforge {

#define CUDA_STUB throw std::runtime_error("CudaBackend: CUDA support not compiled")

CudaBackend::CudaBackend(int) { CUDA_STUB; }
CudaBackend::~CudaBackend() {}
void CudaBackend::sync_to_host()   { CUDA_STUB; }
void CudaBackend::sync_to_device() { CUDA_STUB; }
std::complex<double>* CudaBackend::host_data() { CUDA_STUB; }
const std::complex<double>* CudaBackend::host_data() const { CUDA_STUB; }
void CudaBackend::reset()          { CUDA_STUB; }
void CudaBackend::apply_single_gate(int, std::complex<double>, std::complex<double>,
                                     std::complex<double>, std::complex<double>) { CUDA_STUB; }
void CudaBackend::H(int)            { CUDA_STUB; }
void CudaBackend::X(int)            { CUDA_STUB; }
void CudaBackend::Y(int)            { CUDA_STUB; }
void CudaBackend::Z(int)            { CUDA_STUB; }
void CudaBackend::RX(int, double)   { CUDA_STUB; }
void CudaBackend::RY(int, double)   { CUDA_STUB; }
void CudaBackend::RZ(int, double)   { CUDA_STUB; }
void CudaBackend::Phase(int, double){ CUDA_STUB; }
void CudaBackend::S(int)            { CUDA_STUB; }
void CudaBackend::T(int)            { CUDA_STUB; }
void CudaBackend::Xsquare(int)      { CUDA_STUB; }
void CudaBackend::apply_controlled_gate(int, int, std::complex<double>,
    std::complex<double>, std::complex<double>, std::complex<double>) { CUDA_STUB; }
void CudaBackend::CNOT(int, int)    { CUDA_STUB; }
void CudaBackend::CRX(int, int, double) { CUDA_STUB; }
void CudaBackend::CRY(int, int, double) { CUDA_STUB; }
void CudaBackend::CRZ(int, int, double) { CUDA_STUB; }
void CudaBackend::CPhase(int, int, double) { CUDA_STUB; }
void CudaBackend::CP(int, int, double)    { CUDA_STUB; }
void CudaBackend::apply_double_controlled_gate(int, int, int, std::complex<double>,
    std::complex<double>, std::complex<double>, std::complex<double>) { CUDA_STUB; }
void CudaBackend::CCNOT(int, int, int) { CUDA_STUB; }
void CudaBackend::OR(int, int, int)    { CUDA_STUB; }
void CudaBackend::SWAP(int, int)       { CUDA_STUB; }
void CudaBackend::CSWAP(int, int, int) { CUDA_STUB; }
void CudaBackend::ISWAP(int, int)      { CUDA_STUB; }
void CudaBackend::SISWAP(int, int)     { CUDA_STUB; }
void CudaBackend::apply_two_qubit_gate(int, int, const std::complex<double>*) { CUDA_STUB; }
void CudaBackend::E(double, int)    { CUDA_STUB; }
void CudaBackend::E_all(double)     { CUDA_STUB; }
double CudaBackend::measure_one_prob0(int) const { CUDA_STUB; }
void CudaBackend::collapse_one(int, int)         { CUDA_STUB; }
double CudaBackend::pauli_expectation(int, int) const { CUDA_STUB; }
void CudaBackend::probabilities(double*) const   { CUDA_STUB; }

#undef CUDA_STUB
} // namespace qforge

#endif // QFORGE_HAS_CUDA
