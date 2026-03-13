// -*- coding: utf-8 -*-
// dmrg_sweep.cpp — Two-site DMRG sweep with Lanczos eigensolver
#include "qforge/mps.h"
#include "qforge/mpo.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <complex>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace qforge { namespace dmrg {

using C = std::complex<double>;
using CMatrix = Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using CVec = Eigen::Matrix<C, Eigen::Dynamic, 1>;

// ============================================================
// Environment tensor: [chi_mps, w_mpo, chi_mps] stored row-major
// Index: a * w * chi + w_idx * chi + b
// ============================================================
struct EnvTensor {
    std::vector<C> data;
    int chi, w;

    EnvTensor() : chi(1), w(1), data(1, {1.0, 0.0}) {}
    EnvTensor(int chi_, int w_)
        : chi(chi_), w(w_),
          data(static_cast<size_t>(chi_) * w_ * chi_, {0.0, 0.0}) {}

    C& at(int a, int wl, int b) {
        return data[static_cast<size_t>(a) * w * chi + wl * chi + b];
    }
    const C& at(int a, int wl, int b) const {
        return data[static_cast<size_t>(a) * w * chi + wl * chi + b];
    }
    void zero() { std::fill(data.begin(), data.end(), C(0, 0)); }
};

// ============================================================
// Update left environment by contracting one site
// L_new[a', wl', b'] = sum_{a,wl,b,s_bra,s_ket}
//   conj(A[a, s_bra, a']) * L[a, wl, b] * W[wl, wl', s_bra, s_ket] * A[b, s_ket, b']
// ============================================================
static EnvTensor update_left_env(
    const EnvTensor& L,
    const mps::Tensor& A,
    const mpo::MPOTensor& W
) {
    int chi_l = A.chi_left;
    int chi_r = A.chi_right;
    int w_l = W.w_left;
    int w_r = W.w_right;
    const int d = 2;

    EnvTensor L_new(chi_r, w_r);
    L_new.zero();

    for (int a = 0; a < chi_l; ++a)       // left MPS bond (bra)
        for (int b = 0; b < chi_l; ++b)   // left MPS bond (ket)
            for (int wl = 0; wl < w_l; ++wl) {
                C lval = L.at(a, wl, b);
                if (lval == C(0, 0)) continue;
                for (int s_bra = 0; s_bra < d; ++s_bra)
                    for (int s_ket = 0; s_ket < d; ++s_ket)
                        for (int wr = 0; wr < w_r; ++wr) {
                            C wval = W.at(wl, wr, s_bra, s_ket);
                            if (wval == C(0, 0)) continue;
                            for (int ap = 0; ap < chi_r; ++ap)  // right MPS bond (bra)
                                for (int bp = 0; bp < chi_r; ++bp)  // right MPS bond (ket)
                                    L_new.at(ap, wr, bp) +=
                                        std::conj(A.at(a, s_bra, ap))
                                        * lval * wval
                                        * A.at(b, s_ket, bp);
                        }
            }
    return L_new;
}

// ============================================================
// Update right environment by contracting one site
// R_new[a', wr', b'] = sum_{a,wr,b,s_bra,s_ket}
//   conj(A[a', s_bra, a]) * R[a, wr, b] * W[wr', wr, s_bra, s_ket] * A[b', s_ket, b]
// ============================================================
static EnvTensor update_right_env(
    const EnvTensor& R,
    const mps::Tensor& A,
    const mpo::MPOTensor& W
) {
    int chi_l = A.chi_left;
    int chi_r = A.chi_right;
    int w_l = W.w_left;
    int w_r = W.w_right;
    const int d = 2;

    EnvTensor R_new(chi_l, w_l);
    R_new.zero();

    for (int a = 0; a < chi_r; ++a)
        for (int b = 0; b < chi_r; ++b)
            for (int wr = 0; wr < w_r; ++wr) {
                C rval = R.at(a, wr, b);
                if (rval == C(0, 0)) continue;
                for (int s_bra = 0; s_bra < d; ++s_bra)
                    for (int s_ket = 0; s_ket < d; ++s_ket)
                        for (int wl = 0; wl < w_l; ++wl) {
                            C wval = W.at(wl, wr, s_bra, s_ket);
                            if (wval == C(0, 0)) continue;
                            for (int ap = 0; ap < chi_l; ++ap)
                                for (int bp = 0; bp < chi_l; ++bp)
                                    R_new.at(ap, wl, bp) +=
                                        std::conj(A.at(ap, s_bra, a))
                                        * rval * wval
                                        * A.at(bp, s_ket, b);
                        }
            }
    return R_new;
}

// ============================================================
// Build initial environments (full left/right sweeps)
// L_envs[i] = left environment including sites 0..i-1
// R_envs[i] = right environment including sites i+1..N-1
// ============================================================
static void build_environments(
    const mps::MPS& psi, const mpo::MPO& H,
    std::vector<EnvTensor>& L_envs, std::vector<EnvTensor>& R_envs
) {
    const int N = psi.n_qubits();

    // Initial boundary environments: 1x1 identity
    L_envs.resize(N + 1);
    R_envs.resize(N + 1);
    L_envs[0] = EnvTensor(1, 1);
    L_envs[0].at(0, 0, 0) = {1.0, 0.0};
    R_envs[N] = EnvTensor(1, 1);
    R_envs[N].at(0, 0, 0) = {1.0, 0.0};

    // Build left environments
    for (int i = 0; i < N; ++i)
        L_envs[i + 1] = update_left_env(L_envs[i], psi.site(i), H.site(i));

    // Build right environments
    for (int i = N - 1; i >= 0; --i)
        R_envs[i] = update_right_env(R_envs[i + 1], psi.site(i), H.site(i));
}

// ============================================================
// Apply effective two-site Hamiltonian H_eff to theta vector
// H_eff = L[i] ⊗ W[i] ⊗ W[i+1] ⊗ R[i+1]
// theta: [chi_l, d, d, chi_r] → same shape output
// ============================================================
static void apply_heff(
    const EnvTensor& L,       // [chi_l, w_l, chi_l]
    const mpo::MPOTensor& W1, // site i:   [w_l, w_m, d, d]
    const mpo::MPOTensor& W2, // site i+1: [w_m, w_r, d, d]
    const EnvTensor& R,       // [chi_r, w_r, chi_r]
    const C* theta_in,        // [chi_l * d * d * chi_r]
    C* theta_out,
    int chi_l, int chi_r
) {
    const int d = 2;
    const int D = d * d;  // two-site physical dimension
    int w_l = W1.w_left;
    int w_m = W1.w_right;  // == W2.w_left
    int w_r = W2.w_right;

    std::fill(theta_out, theta_out + chi_l * D * chi_r, C(0, 0));

    for (int al = 0; al < chi_l; ++al)       // left MPS index (ket)
        for (int s0_in = 0; s0_in < d; ++s0_in)
            for (int s1_in = 0; s1_in < d; ++s1_in)
                for (int ar = 0; ar < chi_r; ++ar) {
                    C t_in = theta_in[((al * d + s0_in) * d + s1_in) * chi_r + ar];
                    if (t_in == C(0, 0)) continue;

                    for (int wl = 0; wl < w_l; ++wl)
                        for (int alp = 0; alp < chi_l; ++alp) {  // left MPS index (bra)
                            C lval = L.at(alp, wl, al);
                            if (lval == C(0, 0)) continue;

                            for (int wm = 0; wm < w_m; ++wm)
                                for (int s0_out = 0; s0_out < d; ++s0_out) {
                                    C w1val = W1.at(wl, wm, s0_out, s0_in);
                                    if (w1val == C(0, 0)) continue;

                                    for (int wr = 0; wr < w_r; ++wr)
                                        for (int s1_out = 0; s1_out < d; ++s1_out) {
                                            C w2val = W2.at(wm, wr, s1_out, s1_in);
                                            if (w2val == C(0, 0)) continue;

                                            for (int arp = 0; arp < chi_r; ++arp) {
                                                C rval = R.at(ar, wr, arp);
                                                if (rval == C(0, 0)) continue;
                                                theta_out[((alp * d + s0_out) * d + s1_out) * chi_r + arp]
                                                    += lval * w1val * w2val * rval * t_in;
                                            }
                                        }
                                }
                        }
                }
}

// ============================================================
// Lanczos eigensolver — find lowest eigenvalue/eigenvector
// of hermitian operator A defined by matvec: x → A*x
// Returns: lowest eigenvalue
// vec: on input initial guess, on output eigenvector
// ============================================================
static double lanczos(
    std::function<void(const C*, C*)> matvec,
    C* vec, int dim,
    int max_iter, double tol
) {
    max_iter = std::min(max_iter, dim);

    // Normalize initial vector
    double norm_v = 0.0;
    for (int i = 0; i < dim; ++i) norm_v += std::norm(vec[i]);
    norm_v = std::sqrt(norm_v);
    if (norm_v < 1e-14) {
        // Random initialization
        for (int i = 0; i < dim; ++i) vec[i] = C(1.0 / std::sqrt(dim), 0);
        norm_v = 1.0;
    }
    for (int i = 0; i < dim; ++i) vec[i] /= norm_v;

    std::vector<std::vector<C>> V;     // Krylov basis
    std::vector<double> alpha;         // diagonal of tridiagonal matrix
    std::vector<double> beta;          // off-diagonal

    V.push_back(std::vector<C>(vec, vec + dim));

    std::vector<C> w(dim);
    double energy = 0.0;

    for (int j = 0; j < max_iter; ++j) {
        // w = A * v_j
        matvec(V[j].data(), w.data());

        // alpha_j = <v_j | w>
        double aj = 0.0;
        for (int i = 0; i < dim; ++i)
            aj += std::real(std::conj(V[j][i]) * w[i]);
        alpha.push_back(aj);

        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        for (int i = 0; i < dim; ++i) {
            w[i] -= aj * V[j][i];
            if (j > 0) w[i] -= beta[j - 1] * V[j - 1][i];
        }

        double bj = 0.0;
        for (int i = 0; i < dim; ++i) bj += std::norm(w[i]);
        bj = std::sqrt(bj);

        // Diagonalize tridiagonal matrix for eigenvalue convergence check
        int m = static_cast<int>(alpha.size());
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
        for (int k = 0; k < m; ++k) T(k, k) = alpha[k];
        for (int k = 0; k < m - 1; ++k) {
            T(k, k + 1) = (k < (int)beta.size()) ? beta[k] : 0.0;
            T(k + 1, k) = T(k, k + 1);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
        energy = eig.eigenvalues()(0);

        if (bj < tol || j == max_iter - 1) {
            // Reconstruct eigenvector from Krylov basis
            Eigen::VectorXd y = eig.eigenvectors().col(0);
            std::fill(vec, vec + dim, C(0, 0));
            for (int k = 0; k < m && k < (int)V.size(); ++k) {
                double yk = y(k);
                for (int i = 0; i < dim; ++i)
                    vec[i] += yk * V[k][i];
            }
            break;
        }

        beta.push_back(bj);
        V.push_back(std::vector<C>(dim));
        for (int i = 0; i < dim; ++i)
            V.back()[i] = w[i] / bj;
    }
    return energy;
}

// ============================================================
// Two-site DMRG sweep (left-to-right then right-to-left)
// Updates psi in-place, rebuilds environments incrementally
// Returns energy after the sweep
// ============================================================
double dmrg_sweep(
    mps::MPS& psi, const mpo::MPO& H,
    int max_chi, double eps, int lanczos_dim
) {
    const int N = psi.n_qubits();
    const int d = 2;

    // Build all environments from scratch
    std::vector<EnvTensor> L_envs, R_envs;
    build_environments(psi, H, L_envs, R_envs);

    double energy = 0.0;
    mps::SVDWorkspace ws;

    auto optimize_site = [&](int i, bool left_to_right) -> double {
        const EnvTensor& L = L_envs[i];
        const EnvTensor& R = R_envs[i + 2];
        const mpo::MPOTensor& W1 = H.site(i);
        const mpo::MPOTensor& W2 = H.site(i + 1);

        int chi_l = psi.site(i).chi_left;
        int chi_r = psi.site(i + 1).chi_right;
        int theta_dim = chi_l * d * d * chi_r;

        // Initial theta = A[i] * A[i+1]
        std::vector<C> theta(theta_dim, {0.0, 0.0});
        const mps::Tensor& A = psi.site(i);
        const mps::Tensor& B = psi.site(i + 1);
        int chi_m = A.chi_right;
        for (int cl = 0; cl < chi_l; ++cl)
            for (int s0 = 0; s0 < d; ++s0)
                for (int m = 0; m < chi_m; ++m) {
                    C a = A.at(cl, s0, m);
                    if (a == C(0, 0)) continue;
                    for (int s1 = 0; s1 < d; ++s1)
                        for (int cr = 0; cr < chi_r; ++cr)
                            theta[((cl * d + s0) * d + s1) * chi_r + cr] += a * B.at(m, s1, cr);
                }

        // Lanczos optimization
        std::vector<C> theta_out(theta_dim);
        auto matvec = [&](const C* in, C* out) {
            apply_heff(L, W1, W2, R, in, out, chi_l, chi_r);
        };

        double e = lanczos(matvec, theta.data(), theta_dim, lanczos_dim, 1e-10);

        // SVD split with truncation
        mps::Tensor A_new, B_new;
        int new_chi = 0;
        mps::ops::svd_split(theta.data(), chi_l, chi_r,
                             A_new, B_new, new_chi, max_chi, eps, ws);
        psi.site(i) = std::move(A_new);
        psi.site(i + 1) = std::move(B_new);

        return e;
    };

    // Left-to-right sweep
    for (int i = 0; i < N - 1; ++i) {
        energy = optimize_site(i, true);
        // Update left environment incrementally
        L_envs[i + 1] = update_left_env(L_envs[i], psi.site(i), H.site(i));
    }

    // Right-to-left sweep
    for (int i = N - 2; i >= 0; --i) {
        energy = optimize_site(i, false);
        // Update right environment incrementally
        R_envs[i + 1] = update_right_env(R_envs[i + 2], psi.site(i + 1), H.site(i + 1));
    }

    return energy;
}

}} // namespace qforge::dmrg
