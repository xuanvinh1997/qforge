#pragma once
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qforge {

class StateVector {
public:
    explicit StateVector(int n_qubits);
    StateVector(int n_qudits, int dimension);
    ~StateVector();

    StateVector(const StateVector&) = delete;
    StateVector& operator=(const StateVector&) = delete;
    StateVector(StateVector&& other) noexcept;
    StateVector& operator=(StateVector&& other) noexcept;

    int n_qubits() const { return n_qudits_; }
    int n_qudits() const { return n_qudits_; }
    int dimension() const { return dimension_; }
    size_t dim() const { return dim_; }

    /// Precomputed stride for qudit position i: d^(n - i - 1)
    size_t stride(int pos) const { return strides_[pos]; }

    std::complex<double>* data() { return amp_; }
    const std::complex<double>* data() const { return amp_; }

    std::complex<double>& operator[](size_t i) { return amp_[i]; }
    const std::complex<double>& operator[](size_t i) const { return amp_[i]; }

    // Reset to |00...0>
    void reset();

    // Scratch buffer for double-buffered operations
    std::complex<double>* scratch() { return scratch_; }
    void zero_scratch();
    void swap_buffers();

private:
    int n_qudits_;
    int dimension_;           // d=2 for qubits, d=3 for qutrits, etc.
    size_t dim_;              // d^n total amplitudes
    std::complex<double>* amp_;
    std::complex<double>* scratch_;
    std::vector<size_t> strides_;  // strides_[i] = d^(n - i - 1)

    void init(int n_qudits, int dimension);
    static std::complex<double>* alloc_aligned(size_t count);
    static void free_aligned(std::complex<double>* ptr);
};

// --- Qudit index utilities (inline for performance) ---

/// Extract the value of qudit at position `pos` from state index `idx`.
inline int extract_qudit(size_t idx, size_t stride, int d) {
    return static_cast<int>((idx / stride) % d);
}

/// Replace qudit value at position `pos`: change from old_val to new_val.
inline size_t replace_qudit(size_t idx, size_t stride, int old_val, int new_val) {
    return idx + static_cast<size_t>(new_val - old_val) * stride;
}

} // namespace qforge
