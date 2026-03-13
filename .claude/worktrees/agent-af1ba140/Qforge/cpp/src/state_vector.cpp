#include "qforge/state_vector.h"
#include <algorithm>

namespace qforge {

static constexpr size_t ALIGNMENT = 64;

std::complex<double>* StateVector::alloc_aligned(size_t count) {
    size_t bytes = count * sizeof(std::complex<double>);
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, ALIGNMENT);
#else
    if (posix_memalign(&ptr, ALIGNMENT, bytes) != 0) ptr = nullptr;
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<std::complex<double>*>(ptr);
}

void StateVector::free_aligned(std::complex<double>* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

StateVector::StateVector(int n_qubits)
    : n_qubits_(n_qubits)
    , dim_(size_t(1) << n_qubits)
    , amp_(alloc_aligned(size_t(1) << n_qubits))
    , scratch_(alloc_aligned(size_t(1) << n_qubits))
{
    if (n_qubits < 1 || n_qubits > 30)
        throw std::invalid_argument("n_qubits must be between 1 and 30");
    reset();
}

StateVector::~StateVector() {
    free_aligned(amp_);
    free_aligned(scratch_);
}

StateVector::StateVector(StateVector&& other) noexcept
    : n_qubits_(other.n_qubits_)
    , dim_(other.dim_)
    , amp_(other.amp_)
    , scratch_(other.scratch_)
{
    other.amp_ = nullptr;
    other.scratch_ = nullptr;
}

StateVector& StateVector::operator=(StateVector&& other) noexcept {
    if (this != &other) {
        free_aligned(amp_);
        free_aligned(scratch_);
        n_qubits_ = other.n_qubits_;
        dim_ = other.dim_;
        amp_ = other.amp_;
        scratch_ = other.scratch_;
        other.amp_ = nullptr;
        other.scratch_ = nullptr;
    }
    return *this;
}

void StateVector::reset() {
    std::memset(amp_, 0, dim_ * sizeof(std::complex<double>));
    amp_[0] = {1.0, 0.0};
    std::memset(scratch_, 0, dim_ * sizeof(std::complex<double>));
}

void StateVector::zero_scratch() {
    std::memset(scratch_, 0, dim_ * sizeof(std::complex<double>));
}

void StateVector::swap_buffers() {
    std::swap(amp_, scratch_);
}

} // namespace qforge
