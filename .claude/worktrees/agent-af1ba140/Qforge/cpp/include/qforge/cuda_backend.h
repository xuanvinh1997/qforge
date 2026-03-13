#pragma once
#include "backend.h"

namespace qforge {

class CudaBackend : public Backend {
public:
    explicit CudaBackend(int n_qubits);
    ~CudaBackend();

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    BackendType type() const noexcept override { return BackendType::CUDA; }
    int n_qubits() const noexcept override { return n_qubits_; }
    size_t dim() const noexcept override { return dim_; }

    std::complex<double>* host_data() override;
    const std::complex<double>* host_data() const override;

    void sync_to_host() override;
    void sync_to_device() override;
    void reset() override;

    // --- Single-qubit gates ---
    void apply_single_gate(int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) override;

    void H(int target) override;
    void X(int target) override;
    void Y(int target) override;
    void Z(int target) override;
    void RX(int target, double phi) override;
    void RY(int target, double phi) override;
    void RZ(int target, double phi) override;
    void Phase(int target, double phi) override;
    void S(int target) override;
    void T(int target) override;
    void Xsquare(int target) override;

    // --- Controlled gates ---
    void apply_controlled_gate(int control, int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) override;

    void CNOT(int control, int target) override;
    void CRX(int control, int target, double phi) override;
    void CRY(int control, int target, double phi) override;
    void CRZ(int control, int target, double phi) override;
    void CPhase(int control, int target, double phi) override;
    void CP(int control, int target, double phi) override;

    // --- Double-controlled gates ---
    void apply_double_controlled_gate(int c1, int c2, int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) override;

    void CCNOT(int c1, int c2, int target) override;
    void OR(int c1, int c2, int target) override;

    // --- Swap gates ---
    void SWAP(int t1, int t2) override;
    void CSWAP(int control, int t1, int t2) override;
    void ISWAP(int t1, int t2) override;
    void SISWAP(int t1, int t2) override;

    // --- Noise ---
    void E(double p_noise, int target) override;
    void E_all(double p_noise) override;

    // --- Measurement ---
    double measure_one_prob0(int qubit) const override;
    void collapse_one(int qubit, int value) override;
    double pauli_expectation(int qubit, int pauli_type) const override;
    void probabilities(double* out) const override;

private:
    int n_qubits_;
    size_t dim_;

    void* d_sv_;                      // cuDoubleComplex* on device
    mutable std::complex<double>* h_cache_;  // host-side cache
    mutable bool host_dirty_;         // GPU has newer data than host cache
    bool device_dirty_;               // host cache has newer data than GPU

    void* d_mat2q_;                   // persistent 16-element matrix buffer (no per-call cudaMalloc)
    void* d_reduce_;                  // persistent 1024-double reduction output buffer for GPU measurement
    void* d_scratch_;                 // persistent dim_-double scratch buffer for GPU depolarizing noise
    void* stream_;                    // cudaStream_t (typed as void* to avoid including cuda_runtime.h here)

    // Two-qubit gate helper (uploads 4x4 matrix to device, launches kernel)
    void apply_two_qubit_gate(int t1, int t2,
                               const std::complex<double>* mat4x4);
};

} // namespace qforge
