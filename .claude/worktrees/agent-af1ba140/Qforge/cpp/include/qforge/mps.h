#pragma once
#include <complex>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace qforge { namespace mps {

// ============================================================
// Single MPS site tensor — shape [chi_left, d, chi_right]
// Stored row-major: index = cl * d * chi_right + s * chi_right + cr
// ============================================================
struct Tensor {
    std::vector<std::complex<double>> data;
    int chi_left, chi_right, d;

    Tensor() : chi_left(1), chi_right(1), d(2), data(2, {0.0, 0.0}) {}
    Tensor(int cl, int cr, int d_ = 2)
        : chi_left(cl), chi_right(cr), d(d_),
          data(static_cast<size_t>(cl) * d_ * cr, {0.0, 0.0}) {}

    std::complex<double>& at(int cl, int s, int cr) {
        return data[static_cast<size_t>(cl) * d * chi_right
                  + static_cast<size_t>(s) * chi_right + cr];
    }
    const std::complex<double>& at(int cl, int s, int cr) const {
        return data[static_cast<size_t>(cl) * d * chi_right
                  + static_cast<size_t>(s) * chi_right + cr];
    }
    size_t size() const { return data.size(); }
    void zero() { std::fill(data.begin(), data.end(), std::complex<double>(0, 0)); }
};

// ============================================================
// SVD workspace — reused across calls, zero per-gate allocation
// ============================================================
struct SVDWorkspace {
    std::vector<std::complex<double>> mat;   // input matrix copy
    std::vector<std::complex<double>> U;
    std::vector<std::complex<double>> Vt;
    std::vector<double> S;

    void resize(int m, int n) {
        int k = std::min(m, n);
        mat.resize(static_cast<size_t>(m) * n);
        U.resize(static_cast<size_t>(m) * k);
        Vt.resize(static_cast<size_t>(k) * n);
        S.resize(k);
    }
};

// ============================================================
// MPS: chain of tensors, qubit 0 = leftmost (MSB convention)
// Boundary: chi_left[0] = 1, chi_right[N-1] = 1
// ============================================================
class MPS {
public:
    explicit MPS(int n_qubits, int max_bond_dim = 32);
    MPS(const MPS&) = default;
    MPS& operator=(const MPS&) = default;
    MPS(MPS&&) noexcept = default;
    MPS& operator=(MPS&&) noexcept = default;

    int n_qubits() const { return n_qubits_; }
    int max_bond_dim() const { return max_bond_dim_; }
    int ortho_center() const { return ortho_center_; }
    void set_ortho_center(int c) { ortho_center_ = c; }

    Tensor& site(int i) { return tensors_[i]; }
    const Tensor& site(int i) const { return tensors_[i]; }

    int bond_dim(int bond) const {
        if (bond < 0 || bond >= n_qubits_ - 1)
            throw std::out_of_range("bond index out of range");
        return tensors_[bond].chi_right;
    }
    int max_current_chi() const {
        int mx = 1;
        for (int b = 0; b < n_qubits_ - 1; ++b)
            if (tensors_[b].chi_right > mx) mx = tensors_[b].chi_right;
        return mx;
    }

    // Reset to |00...0> product state
    void reset();

    std::vector<Tensor> tensors_;

private:
    int n_qubits_;
    int max_bond_dim_;
    int ortho_center_;  // -1 = not canonicalized
};

// ============================================================
// Operations namespace
// ============================================================
namespace ops {

// --- Canonicalization ---
void left_canonicalize(MPS& mps, int up_to_site, SVDWorkspace& ws);
void right_canonicalize(MPS& mps, int from_site, SVDWorkspace& ws);
void shift_ortho_center(MPS& mps, int target, SVDWorkspace& ws);
void canonicalize(MPS& mps, SVDWorkspace& ws);

// --- SVD split ---
// Merge two tensors at sites i, i+1 into theta [chi_l, d, d, chi_r],
// apply gate, SVD-split back. Returns truncation error.
double svd_split(
    const std::complex<double>* theta,  // [chi_l * d * d * chi_r] row-major
    int chi_l, int chi_r,
    Tensor& A,          // output [chi_l, d, chi_new]
    Tensor& B,          // output [chi_new, d, chi_r]
    int& new_chi,
    int max_chi,
    double eps,
    SVDWorkspace& ws
);

// --- Gate application ---
void apply_single_qubit_gate(MPS& mps, int site,
                              const std::complex<double> gate[4]);

// Returns truncation error
double apply_two_qubit_gate(MPS& mps, int site_i,
                             const std::complex<double> gate[16],
                             int max_chi, double eps, SVDWorkspace& ws);

// --- State vector conversion ---
void from_statevector(MPS& mps, const std::complex<double>* amp,
                      size_t dim, int max_chi, double eps, SVDWorkspace& ws);
void to_statevector(const MPS& mps, std::complex<double>* amp);

// --- Measurement ---
std::complex<double> single_site_expectation(const MPS& mps, int site,
                                              const std::complex<double> op[4]);
std::complex<double> two_site_expectation(const MPS& mps, int si, int sj,
                                           const std::complex<double> opi[4],
                                           const std::complex<double> opj[4]);
double measure_prob0(const MPS& mps, int site);
double entanglement_entropy(const MPS& mps, int bond);
double norm(const MPS& mps);

} // namespace ops
}} // namespace qforge::mps
