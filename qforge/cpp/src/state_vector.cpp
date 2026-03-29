#include "qforge/state_vector.h"
#include <algorithm>
#include <cmath>

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

static size_t pow_int(int base, int exp) {
    size_t result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

void StateVector::init(int n_qudits, int dimension) {
    n_qudits_ = n_qudits;
    dimension_ = dimension;
    dim_ = pow_int(dimension, n_qudits);
    amp_ = alloc_aligned(dim_);
    scratch_ = alloc_aligned(dim_);

    // Precompute strides: strides_[i] = d^(n - i - 1)
    strides_.resize(n_qudits);
    for (int i = 0; i < n_qudits; ++i)
        strides_[i] = pow_int(dimension, n_qudits - i - 1);

    reset();
}

StateVector::StateVector(int n_qubits)
    : amp_(nullptr), scratch_(nullptr)
{
    if (n_qubits < 1 || n_qubits > 30)
        throw std::invalid_argument("n_qubits must be between 1 and 30");
    init(n_qubits, 2);
}

StateVector::StateVector(int n_qudits, int dimension)
    : amp_(nullptr), scratch_(nullptr)
{
    if (n_qudits < 1)
        throw std::invalid_argument("n_qudits must be >= 1");
    if (dimension < 2)
        throw std::invalid_argument("dimension must be >= 2");

    // Check that d^n won't exceed a reasonable memory limit (~16 GB)
    double log_dim = n_qudits * std::log2(dimension);
    if (log_dim > 30.0)
        throw std::invalid_argument("State space too large (d^n > 2^30)");

    init(n_qudits, dimension);
}

StateVector::~StateVector() {
    free_aligned(amp_);
    free_aligned(scratch_);
}

StateVector::StateVector(StateVector&& other) noexcept
    : n_qudits_(other.n_qudits_)
    , dimension_(other.dimension_)
    , dim_(other.dim_)
    , amp_(other.amp_)
    , scratch_(other.scratch_)
    , strides_(std::move(other.strides_))
{
    other.amp_ = nullptr;
    other.scratch_ = nullptr;
}

StateVector& StateVector::operator=(StateVector&& other) noexcept {
    if (this != &other) {
        free_aligned(amp_);
        free_aligned(scratch_);
        n_qudits_ = other.n_qudits_;
        dimension_ = other.dimension_;
        dim_ = other.dim_;
        amp_ = other.amp_;
        scratch_ = other.scratch_;
        strides_ = std::move(other.strides_);
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
