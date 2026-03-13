#pragma once
#include <complex>
#include <vector>
#include <string>
#include <utility>

namespace qforge { namespace mpo {

// ============================================================
// MPO tensor for site i — shape [w_left, w_right, d, d]
// w: MPO bond dimension (small: 3-5 for standard models)
// d: physical dimension = 2 (qubit)
// Index: wl * w_right * d * d + wr * d * d + s_bra * d + s_ket
// ============================================================
struct MPOTensor {
    std::vector<std::complex<double>> data;
    int w_left, w_right, d;

    MPOTensor() : w_left(1), w_right(1), d(2), data(4, {0.0, 0.0}) {}
    MPOTensor(int wl, int wr, int d_ = 2)
        : w_left(wl), w_right(wr), d(d_),
          data(static_cast<size_t>(wl) * wr * d_ * d_, {0.0, 0.0}) {}

    std::complex<double>& at(int wl, int wr, int sb, int sk) {
        return data[static_cast<size_t>(wl) * w_right * d * d
                  + static_cast<size_t>(wr) * d * d
                  + static_cast<size_t>(sb) * d + sk];
    }
    const std::complex<double>& at(int wl, int wr, int sb, int sk) const {
        return data[static_cast<size_t>(wl) * w_right * d * d
                  + static_cast<size_t>(wr) * d * d
                  + static_cast<size_t>(sb) * d + sk];
    }
    void zero() { std::fill(data.begin(), data.end(), std::complex<double>(0, 0)); }
};

// ============================================================
// MPO: chain of MPOTensors, one per site
// Boundary: w_left[0] = w_right[N-1] = 1
// ============================================================
class MPO {
public:
    explicit MPO(int n_sites, int bond_dim = 5);
    MPO(const MPO&) = default;
    MPO& operator=(const MPO&) = default;
    MPO(MPO&&) noexcept = default;
    MPO& operator=(MPO&&) noexcept = default;

    int n_sites() const { return n_sites_; }
    int bond_dim() const { return bond_dim_; }

    MPOTensor& site(int i) { return tensors_[i]; }
    const MPOTensor& site(int i) const { return tensors_[i]; }

    std::vector<MPOTensor> tensors_;
private:
    int n_sites_;
    int bond_dim_;
};

// ============================================================
// Built-in Hamiltonian factories
// ============================================================

// Heisenberg: H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
// MPO bond dim = 5
MPO heisenberg(int n_sites, double J = 1.0);

// Transverse-field Ising: H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
// MPO bond dim = 3
MPO ising(int n_sites, double J = 1.0, double h = 0.5);

// XXZ: H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1})
// MPO bond dim = 5
MPO xxz(int n_sites, double Delta = 1.0, double J = 1.0);

}} // namespace qforge::mpo
