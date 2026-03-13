#pragma once
#include "backend.h"
#include "state_vector.h"

namespace qforge {

class CpuBackend : public Backend {
public:
    explicit CpuBackend(int n_qubits);

    BackendType type() const noexcept override { return BackendType::CPU; }
    int n_qubits() const noexcept override { return sv_.n_qubits(); }
    size_t dim() const noexcept override { return sv_.dim(); }

    std::complex<double>* host_data() override { return sv_.data(); }
    const std::complex<double>* host_data() const override { return sv_.data(); }

    void sync_to_host() override {}   // no-op
    void sync_to_device() override {} // no-op
    void reset() override { sv_.reset(); }

    // Access underlying StateVector (for backward compat with _qsun_core)
    StateVector& state_vector() { return sv_; }
    const StateVector& state_vector() const { return sv_; }

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
    StateVector sv_;
};

} // namespace qforge
