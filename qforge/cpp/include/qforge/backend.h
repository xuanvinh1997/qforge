#pragma once
#include <complex>
#include <cstddef>
#include <vector>

namespace qforge {

enum class BackendType { CPU, CUDA, METAL };

class Backend {
public:
    virtual ~Backend() = default;

    virtual BackendType type() const noexcept = 0;
    virtual int n_qubits() const noexcept = 0;
    virtual size_t dim() const noexcept = 0;

    // CPU-accessible amplitude pointer (may trigger device->host sync)
    virtual std::complex<double>* host_data() = 0;
    virtual const std::complex<double>* host_data() const = 0;

    // Synchronization (no-ops for CPU backend)
    virtual void sync_to_host() = 0;
    virtual void sync_to_device() = 0;

    // Reset to |00...0>
    virtual void reset() = 0;

    // --- Single-qubit gates ---
    virtual void apply_single_gate(int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) = 0;

    // Gate convenience methods
    virtual void H(int target) = 0;
    virtual void X(int target) = 0;
    virtual void Y(int target) = 0;
    virtual void Z(int target) = 0;
    virtual void RX(int target, double phi) = 0;
    virtual void RY(int target, double phi) = 0;
    virtual void RZ(int target, double phi) = 0;
    virtual void Phase(int target, double phi) = 0;
    virtual void S(int target) = 0;
    virtual void T(int target) = 0;
    virtual void Xsquare(int target) = 0;

    // --- Controlled gates ---
    virtual void apply_controlled_gate(int control, int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) = 0;

    virtual void CNOT(int control, int target) = 0;
    virtual void CRX(int control, int target, double phi) = 0;
    virtual void CRY(int control, int target, double phi) = 0;
    virtual void CRZ(int control, int target, double phi) = 0;
    virtual void CPhase(int control, int target, double phi) = 0;
    virtual void CP(int control, int target, double phi) = 0;

    // --- Double-controlled gates ---
    virtual void apply_double_controlled_gate(int c1, int c2, int target,
        std::complex<double> m00, std::complex<double> m01,
        std::complex<double> m10, std::complex<double> m11) = 0;

    virtual void CCNOT(int c1, int c2, int target) = 0;
    virtual void OR(int c1, int c2, int target) = 0;

    // --- Swap gates ---
    virtual void SWAP(int t1, int t2) = 0;
    virtual void CSWAP(int control, int t1, int t2) = 0;
    virtual void ISWAP(int t1, int t2) = 0;
    virtual void SISWAP(int t1, int t2) = 0;

    // --- Noise ---
    virtual void E(double p_noise, int target) = 0;
    virtual void E_all(double p_noise) = 0;

    // --- Measurement ---
    virtual double measure_one_prob0(int qubit) const = 0;
    virtual void collapse_one(int qubit, int value) = 0;
    virtual double pauli_expectation(int qubit, int pauli_type) const = 0;
    virtual void probabilities(double* out) const = 0;
};

} // namespace qforge
