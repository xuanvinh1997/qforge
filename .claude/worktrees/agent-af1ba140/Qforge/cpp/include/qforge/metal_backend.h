#pragma once
#include "backend.h"
#include <string>

// Forward declarations for Metal types (avoid including Metal headers here)
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLComputePipelineState;
@protocol MTLBuffer;
@protocol MTLLibrary;
#else
// Opaque pointers for C++ compilation
typedef void* id;
#endif

namespace qforge {

class MetalBackend : public Backend {
public:
    explicit MetalBackend(int n_qubits);
    ~MetalBackend();

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    BackendType type() const noexcept override { return BackendType::METAL; }
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

    // Double-precision host mirror
    mutable std::complex<double>* h_cache_;
    mutable bool host_dirty_;  // true when GPU has newer data
    bool device_dirty_;        // true when host has newer data

    // Metal objects (stored as void* to avoid Obj-C in header)
    void* device_;             // id<MTLDevice>
    void* queue_;              // id<MTLCommandQueue>
    void* amp_buf_;            // id<MTLBuffer> - float2 amplitudes
    void* partial_buf_;        // id<MTLBuffer> - for reduction
    void* library_;            // id<MTLLibrary>

    // Pipeline states for each kernel
    void* pso_single_gate_;
    void* pso_controlled_gate_;
    void* pso_double_controlled_gate_;
    void* pso_swap_;
    void* pso_iswap_;
    void* pso_siswap_;
    void* pso_cswap_;
    void* pso_measure_prob0_;
    void* pso_probabilities_;

    // Command buffer batching
    void* pending_cmd_buf_;     // id<MTLCommandBuffer>
    void* pending_encoder_;     // id<MTLComputeCommandEncoder>

    void init_metal();
    void create_pipelines();
    void upload_f64_to_f32();
    void download_f32_to_f64() const;
    void flush();  // Commit pending GPU commands and wait
};

} // namespace qforge
