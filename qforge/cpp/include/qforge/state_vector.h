#pragma once
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qforge {

class StateVector {
public:
    explicit StateVector(int n_qubits);
    ~StateVector();

    StateVector(const StateVector&) = delete;
    StateVector& operator=(const StateVector&) = delete;
    StateVector(StateVector&& other) noexcept;
    StateVector& operator=(StateVector&& other) noexcept;

    int n_qubits() const { return n_qubits_; }
    size_t dim() const { return dim_; }

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
    int n_qubits_;
    size_t dim_;
    std::complex<double>* amp_;
    std::complex<double>* scratch_;

    static std::complex<double>* alloc_aligned(size_t count);
    static void free_aligned(std::complex<double>* ptr);
};

} // namespace qforge
