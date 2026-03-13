// -*- coding: utf-8 -*-
// mps_core.cpp — MPS class, canonicalization, from/to statevector
// Uses Eigen3 JacobiSVD for portability (no LAPACK link dependency).
#include "qforge/mps.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cstring>
#include <cmath>
#include <numeric>

namespace qforge { namespace mps {

// ============================================================
// MPS constructor / reset
// ============================================================

MPS::MPS(int n_qubits, int max_bond_dim)
    : n_qubits_(n_qubits), max_bond_dim_(max_bond_dim), ortho_center_(-1)
{
    if (n_qubits < 1)
        throw std::invalid_argument("n_qubits must be >= 1");
    tensors_.resize(n_qubits);
    reset();
}

void MPS::reset() {
    // Initialize to |00...0> product state
    // Each tensor has shape [1, 2, 1]: [[1, 0]] (|0> state)
    for (int i = 0; i < n_qubits_; ++i) {
        tensors_[i] = Tensor(1, 1, 2);
        tensors_[i].at(0, 0, 0) = {1.0, 0.0};  // |0>
        tensors_[i].at(0, 1, 0) = {0.0, 0.0};  // not |1>
    }
    ortho_center_ = -1;
}

// ============================================================
// Internal SVD helper using Eigen3
// Computes SVD of matrix M (m x n), returns U, S, Vt
// Truncates to max_chi singular values above eps * S[0]
// Returns truncation error (sum of discarded singular values squared)
// ============================================================
static double eigen_svd_truncate(
    const std::complex<double>* M_data, int m, int n,
    std::vector<std::complex<double>>& U_out,
    std::vector<double>& S_out,
    std::vector<std::complex<double>>& Vt_out,
    int max_chi, double eps
) {
    using CMatrix = Eigen::Matrix<std::complex<double>,
                                  Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;
    // Map input data into Eigen matrix
    Eigen::Map<const CMatrix> M(M_data, m, n);

    // Thin SVD
    Eigen::JacobiSVD<CMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const auto& sing = svd.singularValues();  // real, sorted descending
    int k_full = static_cast<int>(sing.size());

    // Determine truncation rank
    double threshold = (k_full > 0 && sing[0] > 0.0) ? eps * sing[0] : eps;
    int keep = 0;
    for (int i = 0; i < k_full; ++i) {
        if (sing[i] > threshold) keep++;
        else break;
    }
    keep = std::max(1, std::min(keep, max_chi));

    // Compute truncation error
    double trunc_err = 0.0;
    for (int i = keep; i < k_full; ++i)
        trunc_err += sing[i] * sing[i];

    // Store results
    S_out.resize(keep);
    for (int i = 0; i < keep; ++i)
        S_out[i] = sing[i];

    // U: m x keep (column-major from Eigen, convert to row-major)
    const auto& U_eig = svd.matrixU();
    U_out.resize(static_cast<size_t>(m) * keep);
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < keep; ++c)
            U_out[r * keep + c] = U_eig(r, c);

    // V: n x k_full (Eigen returns V, not Vt)
    // Vt: keep x n
    const auto& V_eig = svd.matrixV();
    Vt_out.resize(static_cast<size_t>(keep) * n);
    for (int r = 0; r < keep; ++r)
        for (int c = 0; c < n; ++c)
            Vt_out[r * n + c] = std::conj(V_eig(c, r));

    return trunc_err;
}

// ============================================================
// ops::svd_split
// theta: [chi_l, d, d, chi_r] → A [chi_l, d, chi_new], B [chi_new, d, chi_r]
// ============================================================
double ops::svd_split(
    const std::complex<double>* theta,
    int chi_l, int chi_r,
    Tensor& A, Tensor& B,
    int& new_chi,
    int max_chi, double eps,
    SVDWorkspace& ws
) {
    constexpr int d = 2;
    int m = chi_l * d;
    int n = d * chi_r;

    std::vector<std::complex<double>> U_flat;
    std::vector<double> S_flat;
    std::vector<std::complex<double>> Vt_flat;

    double trunc_err = eigen_svd_truncate(theta, m, n,
                                           U_flat, S_flat, Vt_flat,
                                           max_chi, eps);
    new_chi = static_cast<int>(S_flat.size());

    // A[chi_l, d, new_chi] = U * diag(S)  (absorb S into A, left-canonical convention)
    // Actually for two-site DMRG we keep U as left-canonical and S*Vt as right tensor
    // Here we absorb sqrt(S) into both for symmetric gauge, or S into B.
    // Convention: A = U (left-canonical), B = diag(S) * Vt (not normalized)
    A = Tensor(chi_l, new_chi, d);
    for (int cl = 0; cl < chi_l; ++cl)
        for (int s = 0; s < d; ++s)
            for (int cr = 0; cr < new_chi; ++cr)
                A.at(cl, s, cr) = U_flat[cl * d * new_chi + s * new_chi + cr];

    // B[new_chi, d, chi_r] = diag(S) * Vt  (Vt is [new_chi, d*chi_r])
    B = Tensor(new_chi, chi_r, d);
    for (int r = 0; r < new_chi; ++r)
        for (int s = 0; s < d; ++s)
            for (int c = 0; c < chi_r; ++c)
                B.at(r, s, c) = S_flat[r] * Vt_flat[r * n + s * chi_r + c];

    return trunc_err;
}

// ============================================================
// Left-canonicalize sites [0, up_to_site)
// After this, tensors 0..up_to_site-1 are left-isometries
// ============================================================
void ops::left_canonicalize(MPS& mps, int up_to_site, SVDWorkspace& ws) {
    const int N = mps.n_qubits();
    for (int i = 0; i < up_to_site && i < N - 1; ++i) {
        Tensor& A = mps.site(i);
        int chi_l = A.chi_left;
        int chi_r = A.chi_right;
        constexpr int d = 2;
        int m = chi_l * d;
        int n = chi_r;

        // Reshape A to [m, n] = [chi_l * d, chi_r]
        std::vector<std::complex<double>> M(m * n);
        for (int cl = 0; cl < chi_l; ++cl)
            for (int s = 0; s < d; ++s)
                for (int cr = 0; cr < chi_r; ++cr)
                    M[(cl * d + s) * n + cr] = A.at(cl, s, cr);

        std::vector<std::complex<double>> U_flat, Vt_flat;
        std::vector<double> S_flat;
        eigen_svd_truncate(M.data(), m, n, U_flat, S_flat, Vt_flat,
                           mps.max_bond_dim(), 1e-14);
        int new_chi = static_cast<int>(S_flat.size());

        // Update site i: A_new = U [chi_l, d, new_chi]
        A = Tensor(chi_l, new_chi, d);
        for (int cl = 0; cl < chi_l; ++cl)
            for (int s = 0; s < d; ++s)
                for (int cr = 0; cr < new_chi; ++cr)
                    A.at(cl, s, cr) = U_flat[(cl * d + s) * new_chi + cr];

        // Absorb S * Vt into site i+1
        Tensor& B = mps.site(i + 1);
        int old_chi_r = B.chi_right;
        Tensor B_new(new_chi, old_chi_r, d);
        // S * Vt is [new_chi, n=chi_r_old]; multiply into B
        // B_new[r, s, c] = sum_{k} (S[r] * Vt[r,k]) * B_old[k, s, c]
        for (int r = 0; r < new_chi; ++r)
            for (int s = 0; s < d; ++s)
                for (int c = 0; c < old_chi_r; ++c) {
                    std::complex<double> val = 0;
                    for (int k = 0; k < chi_r; ++k)
                        val += S_flat[r] * Vt_flat[r * chi_r + k] * B.at(k, s, c);
                    B_new.at(r, s, c) = val;
                }
        B = std::move(B_new);
    }
}

// ============================================================
// Right-canonicalize sites (from_site, N-1]
// ============================================================
void ops::right_canonicalize(MPS& mps, int from_site, SVDWorkspace& ws) {
    const int N = mps.n_qubits();
    for (int i = N - 1; i > from_site && i > 0; --i) {
        Tensor& A = mps.site(i);
        int chi_l = A.chi_left;
        int chi_r = A.chi_right;
        constexpr int d = 2;
        int m = chi_l;
        int n = d * chi_r;

        // Reshape A to [chi_l, d * chi_r]
        std::vector<std::complex<double>> M(m * n);
        for (int cl = 0; cl < chi_l; ++cl)
            for (int s = 0; s < d; ++s)
                for (int cr = 0; cr < chi_r; ++cr)
                    M[cl * n + s * chi_r + cr] = A.at(cl, s, cr);

        std::vector<std::complex<double>> U_flat, Vt_flat;
        std::vector<double> S_flat;
        eigen_svd_truncate(M.data(), m, n, U_flat, S_flat, Vt_flat,
                           mps.max_bond_dim(), 1e-14);
        int new_chi = static_cast<int>(S_flat.size());

        // site i becomes Vt [new_chi, d, chi_r] (right-isometry)
        A = Tensor(new_chi, chi_r, d);
        for (int r = 0; r < new_chi; ++r)
            for (int s = 0; s < d; ++s)
                for (int c = 0; c < chi_r; ++c)
                    A.at(r, s, c) = Vt_flat[r * n + s * chi_r + c];

        // Absorb U * S into site i-1
        Tensor& prev = mps.site(i - 1);
        int old_chi_l = prev.chi_left;
        Tensor prev_new(old_chi_l, new_chi, d);
        // prev_new[cl, s, r] = sum_k prev[cl, s, k] * U[k, r] * S[r]
        for (int cl = 0; cl < old_chi_l; ++cl)
            for (int s = 0; s < d; ++s)
                for (int r = 0; r < new_chi; ++r) {
                    std::complex<double> val = 0;
                    for (int k = 0; k < chi_l; ++k)
                        val += prev.at(cl, s, k) * U_flat[k * new_chi + r] * S_flat[r];
                    prev_new.at(cl, s, r) = val;
                }
        prev = std::move(prev_new);
    }
}

void ops::canonicalize(MPS& mps, SVDWorkspace& ws) {
    ops::left_canonicalize(mps, mps.n_qubits() - 1, ws);
    mps.set_ortho_center(mps.n_qubits() - 1);
}

void ops::shift_ortho_center(MPS& mps, int target, SVDWorkspace& ws) {
    int cur = mps.ortho_center();
    if (cur == target) return;
    if (cur < target)
        ops::left_canonicalize(mps, target, ws);
    else
        ops::right_canonicalize(mps, target, ws);
    mps.set_ortho_center(target);
}

// ============================================================
// from_statevector: amplitude vector → MPS via sequential SVD
// Amplitude ordering: qubit 0 = MSB (Qforge convention)
// ============================================================
void ops::from_statevector(MPS& mps, const std::complex<double>* amp,
                            size_t dim, int max_chi, double eps,
                            SVDWorkspace& ws) {
    const int N = mps.n_qubits();
    if (dim != (size_t(1) << N))
        throw std::invalid_argument("amplitude size mismatch");

    // Start with the full state reshaped as [1, 2^N]
    std::vector<std::complex<double>> psi(amp, amp + dim);

    int chi_l = 1;
    for (int i = 0; i < N - 1; ++i) {
        int d = 2;
        int cols = static_cast<int>(psi.size()) / (chi_l * d);
        int m = chi_l * d;
        int n = cols;

        std::vector<std::complex<double>> U_flat, Vt_flat;
        std::vector<double> S_flat;
        eigen_svd_truncate(psi.data(), m, n, U_flat, S_flat, Vt_flat, max_chi, eps);
        int new_chi = static_cast<int>(S_flat.size());

        // site i: [chi_l, d, new_chi]
        mps.site(i) = Tensor(chi_l, new_chi, d);
        for (int cl = 0; cl < chi_l; ++cl)
            for (int s = 0; s < d; ++s)
                for (int cr = 0; cr < new_chi; ++cr)
                    mps.site(i).at(cl, s, cr) = U_flat[(cl * d + s) * new_chi + cr];

        // psi = diag(S) * Vt, shape [new_chi, cols]
        psi.resize(static_cast<size_t>(new_chi) * n);
        for (int r = 0; r < new_chi; ++r)
            for (int c = 0; c < n; ++c)
                psi[r * n + c] = S_flat[r] * Vt_flat[r * n + c];

        chi_l = new_chi;
    }
    // Last site: psi has shape [chi_l, d]
    mps.site(N - 1) = Tensor(chi_l, 1, 2);
    for (int cl = 0; cl < chi_l; ++cl)
        for (int s = 0; s < 2; ++s)
            mps.site(N - 1).at(cl, s, 0) = psi[cl * 2 + s];

    mps.set_ortho_center(-1);
}

// ============================================================
// to_statevector: contract MPS → full amplitude vector
// ============================================================
void ops::to_statevector(const MPS& mps, std::complex<double>* amp) {
    const int N = mps.n_qubits();
    const size_t dim = size_t(1) << N;

    // Iterative contraction from left
    // result shape: [chi, 2^i] at step i
    const Tensor& t0 = mps.site(0);
    int chi_r = t0.chi_right;
    // Start: result[s, cr] = t0[0, s, cr], shape [2, chi_r] = [2*chi_r]
    std::vector<std::complex<double>> result(2 * chi_r);
    for (int s = 0; s < 2; ++s)
        for (int cr = 0; cr < chi_r; ++cr)
            result[s * chi_r + cr] = t0.at(0, s, cr);
    // result shape: [2^1, chi_r]

    for (int i = 1; i < N; ++i) {
        const Tensor& A = mps.site(i);
        int old_states = static_cast<int>(result.size()) / chi_r;  // = 2^i
        int new_chi_r = A.chi_right;
        // new_result[old_s * d + s, new_cr] = sum_{cr} result[old_s, cr] * A[cr, s, new_cr]
        int new_states = old_states * 2;
        std::vector<std::complex<double>> new_result(new_states * new_chi_r, {0, 0});
        for (int os = 0; os < old_states; ++os)
            for (int cr = 0; cr < chi_r; ++cr) {
                std::complex<double> r = result[os * chi_r + cr];
                if (r == std::complex<double>(0, 0)) continue;
                for (int s = 0; s < 2; ++s)
                    for (int nc = 0; nc < new_chi_r; ++nc)
                        new_result[(os * 2 + s) * new_chi_r + nc] += r * A.at(cr, s, nc);
            }
        result = std::move(new_result);
        chi_r = new_chi_r;
    }
    // result has shape [2^N, 1], chi_r should be 1
    for (size_t i = 0; i < dim; ++i)
        amp[i] = result[i];
}

}} // namespace qforge::mps
