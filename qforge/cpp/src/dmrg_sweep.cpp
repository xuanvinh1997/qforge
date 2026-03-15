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

// ============================================================
// Apply effective single-site Hamiltonian H_eff to theta vector
// H_eff = L[i] ⊗ W[i] ⊗ R[i+1]
// theta: [chi_l, d, chi_r] → same shape output
// ============================================================
static void apply_heff_1site(
    const EnvTensor& L,       // [chi_l, w_l, chi_l]
    const mpo::MPOTensor& W,  // [w_l, w_r, d, d]
    const EnvTensor& R,       // [chi_r, w_r, chi_r]
    const C* theta_in,        // [chi_l * d * chi_r]
    C* theta_out,
    int chi_l, int chi_r
) {
    const int d = 2;
    int w_l = W.w_left;
    int w_r = W.w_right;

    std::fill(theta_out, theta_out + chi_l * d * chi_r, C(0, 0));

    for (int al = 0; al < chi_l; ++al)
        for (int s_in = 0; s_in < d; ++s_in)
            for (int ar = 0; ar < chi_r; ++ar) {
                C t_in = theta_in[(al * d + s_in) * chi_r + ar];
                if (t_in == C(0, 0)) continue;

                for (int wl = 0; wl < w_l; ++wl)
                    for (int alp = 0; alp < chi_l; ++alp) {
                        C lval = L.at(alp, wl, al);
                        if (lval == C(0, 0)) continue;

                        for (int wr = 0; wr < w_r; ++wr)
                            for (int s_out = 0; s_out < d; ++s_out) {
                                C wval = W.at(wl, wr, s_out, s_in);
                                if (wval == C(0, 0)) continue;

                                for (int arp = 0; arp < chi_r; ++arp) {
                                    C rval = R.at(ar, wr, arp);
                                    if (rval == C(0, 0)) continue;
                                    theta_out[(alp * d + s_out) * chi_r + arp]
                                        += lval * wval * rval * t_in;
                                }
                            }
                    }
            }
}

// ============================================================
// Single-site DMRG sweep (left-to-right then right-to-left)
// More memory efficient than two-site: theta is [chi_l, d, chi_r]
// instead of [chi_l, d, d, chi_r]. Uses SVD to shift ortho center
// after each local optimization.
// Returns energy after the sweep.
// ============================================================
double dmrg_sweep_1site(
    mps::MPS& psi, const mpo::MPO& H,
    int max_chi, double eps, int lanczos_dim
) {
    const int N = psi.n_qubits();
    const int d = 2;
    (void)max_chi;  // single-site doesn't truncate (bond dim is fixed)
    (void)eps;

    std::vector<EnvTensor> L_envs, R_envs;
    build_environments(psi, H, L_envs, R_envs);

    double energy = 0.0;

    auto optimize_site_1s = [&](int i) -> double {
        const EnvTensor& L = L_envs[i];
        const EnvTensor& R = R_envs[i + 1];
        const mpo::MPOTensor& W = H.site(i);

        int chi_l = psi.site(i).chi_left;
        int chi_r = psi.site(i).chi_right;
        int theta_dim = chi_l * d * chi_r;

        // theta = current tensor at site i, flattened
        std::vector<C> theta(psi.site(i).data.begin(), psi.site(i).data.end());

        auto matvec = [&](const C* in, C* out) {
            apply_heff_1site(L, W, R, in, out, chi_l, chi_r);
        };

        double e = lanczos(matvec, theta.data(), theta_dim, lanczos_dim, 1e-10);

        // Write optimized tensor back
        std::copy(theta.begin(), theta.end(), psi.site(i).data.begin());

        return e;
    };

    // Helper: shift ortho center by QR from site i to i+1 (left-to-right)
    auto shift_right = [&](int i) {
        mps::Tensor& A = psi.site(i);
        mps::Tensor& B = psi.site(i + 1);
        int cl = A.chi_left;
        int cr = A.chi_right;

        // Reshape A as [cl*d, cr] and do QR
        int m = cl * d;
        int n = cr;
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(m, n);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c)
                M(r, c) = A.data[r * n + c];

        // Thin QR: Q [m, k], R [k, n] where k = min(m, n)
        Eigen::HouseholderQR<Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr(M);
        int k = std::min(m, n);
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q = qr.householderQ() * Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>::Identity(m, k);
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> R_mat = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();

        // Update A = Q reshaped [cl, d, k]
        A = mps::Tensor(cl, k);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < k; ++c)
                A.data[r * k + c] = Q(r, c);

        // Update B = R * B_old: [k, d, chi_right_B]
        mps::Tensor B_old = B;
        int cr_b = B_old.chi_right;
        B = mps::Tensor(k, cr_b);
        for (int a = 0; a < k; ++a)
            for (int s = 0; s < d; ++s)
                for (int c = 0; c < cr_b; ++c) {
                    C val = {0.0, 0.0};
                    for (int j = 0; j < n; ++j)
                        val += R_mat(a, j) * B_old.at(j, s, c);
                    B.at(a, s, c) = val;
                }
    };

    // Helper: shift ortho center by QR from site i to i-1 (right-to-left)
    auto shift_left = [&](int i) {
        mps::Tensor& A = psi.site(i - 1);
        mps::Tensor& B = psi.site(i);
        int cl = B.chi_left;
        int cr = B.chi_right;

        // Reshape B as [cl, d*cr] then transpose to [d*cr, cl] for QR
        int m = d * cr;
        int n = cl;
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mt(m, n);
        for (int s = 0; s < d; ++s)
            for (int c = 0; c < cr; ++c)
                for (int a = 0; a < cl; ++a)
                    Mt(s * cr + c, a) = B.at(a, s, c);

        int k = std::min(m, n);
        Eigen::HouseholderQR<Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr(Mt);
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q = qr.householderQ() * Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic>::Identity(m, k);
        Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> R_mat = qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();

        // B_new[a, s, c] = Q^T[a, s*cr+c] → B_new has shape [k, d, cr]
        B = mps::Tensor(k, cr);
        for (int a = 0; a < k; ++a)
            for (int s = 0; s < d; ++s)
                for (int c = 0; c < cr; ++c)
                    B.at(a, s, c) = Q(s * cr + c, a);

        // A_new = A_old * R^T: [chi_left_A, d, k]
        mps::Tensor A_old = A;
        int cl_a = A_old.chi_left;
        int cr_a = A_old.chi_right;  // == cl == n
        A = mps::Tensor(cl_a, k);
        for (int a = 0; a < cl_a; ++a)
            for (int s = 0; s < d; ++s)
                for (int c = 0; c < k; ++c) {
                    C val = {0.0, 0.0};
                    for (int j = 0; j < cr_a; ++j)
                        val += A_old.at(a, s, j) * R_mat(c, j);  // R^T[j,c] = R[c,j]
                    A.at(a, s, c) = val;
                }
    };

    // Left-to-right sweep
    for (int i = 0; i < N; ++i) {
        energy = optimize_site_1s(i);
        if (i < N - 1) {
            shift_right(i);
            // Update left environment
            L_envs[i + 1] = update_left_env(L_envs[i], psi.site(i), H.site(i));
        }
    }

    // Right-to-left sweep
    for (int i = N - 1; i >= 0; --i) {
        energy = optimize_site_1s(i);
        if (i > 0) {
            shift_left(i);
            // Update right environment
            R_envs[i] = update_right_env(R_envs[i + 1], psi.site(i), H.site(i));
        }
    }

    return energy;
}

// ============================================================
// Excited-state DMRG sweep via penalty method
// H_eff = H + weight * sum_j |psi_j><psi_j|
// prev_states: previously converged ground/excited MPS states
// ============================================================
double dmrg_sweep_excited(
    mps::MPS& psi, const mpo::MPO& H,
    int max_chi, double eps, int lanczos_dim,
    const std::vector<mps::MPS*>& prev_states, double weight
) {
    const int N = psi.n_qubits();
    const int d = 2;
    const int n_prev = static_cast<int>(prev_states.size());

    // Build environments for H
    std::vector<EnvTensor> L_envs, R_envs;
    build_environments(psi, H, L_envs, R_envs);

    // Build overlap environments for each previous state
    // P_L[j][i] = <prev_j| left contraction up to site i
    // P_R[j][i] = <prev_j| right contraction from site i
    // These are [chi_psi, chi_prev] matrices
    struct OverlapEnv {
        std::vector<C> data;
        int chi_psi, chi_prev;

        OverlapEnv() : chi_psi(1), chi_prev(1), data(1, {1.0, 0.0}) {}
        OverlapEnv(int cp, int cpv) : chi_psi(cp), chi_prev(cpv),
            data(static_cast<size_t>(cp) * cpv, {0.0, 0.0}) {}
        C& at(int a, int b) { return data[static_cast<size_t>(a) * chi_prev + b]; }
        const C& at(int a, int b) const { return data[static_cast<size_t>(a) * chi_prev + b]; }
        void zero() { std::fill(data.begin(), data.end(), C(0, 0)); }
    };

    // Build all overlap environments
    std::vector<std::vector<OverlapEnv>> P_L(n_prev), P_R(n_prev);

    auto build_overlap_left = [&](int j, int site) -> OverlapEnv {
        const OverlapEnv& Lp = P_L[j][site];
        const mps::Tensor& A_psi = psi.site(site);
        const mps::Tensor& A_prev = prev_states[j]->site(site);
        int cl_psi = A_psi.chi_left, cr_psi = A_psi.chi_right;
        int cl_prev = A_prev.chi_left, cr_prev = A_prev.chi_right;

        OverlapEnv Lnew(cr_psi, cr_prev);
        Lnew.zero();
        for (int a = 0; a < cl_psi; ++a)
            for (int b = 0; b < cl_prev; ++b) {
                C lval = Lp.at(a, b);
                if (lval == C(0, 0)) continue;
                for (int s = 0; s < d; ++s)
                    for (int ap = 0; ap < cr_psi; ++ap)
                        for (int bp = 0; bp < cr_prev; ++bp)
                            Lnew.at(ap, bp) += std::conj(A_psi.at(a, s, ap)) * lval * A_prev.at(b, s, bp);
            }
        return Lnew;
    };

    auto build_overlap_right = [&](int j, int site) -> OverlapEnv {
        const OverlapEnv& Rp = P_R[j][site + 1];
        const mps::Tensor& A_psi = psi.site(site);
        const mps::Tensor& A_prev = prev_states[j]->site(site);
        int cl_psi = A_psi.chi_left, cr_psi = A_psi.chi_right;
        int cl_prev = A_prev.chi_left, cr_prev = A_prev.chi_right;

        OverlapEnv Rnew(cl_psi, cl_prev);
        Rnew.zero();
        for (int a = 0; a < cr_psi; ++a)
            for (int b = 0; b < cr_prev; ++b) {
                C rval = Rp.at(a, b);
                if (rval == C(0, 0)) continue;
                for (int s = 0; s < d; ++s)
                    for (int ap = 0; ap < cl_psi; ++ap)
                        for (int bp = 0; bp < cl_prev; ++bp)
                            Rnew.at(ap, bp) += std::conj(A_psi.at(ap, s, a)) * rval * A_prev.at(bp, s, b);
            }
        return Rnew;
    };

    for (int j = 0; j < n_prev; ++j) {
        P_L[j].resize(N + 1);
        P_R[j].resize(N + 1);
        P_L[j][0] = OverlapEnv();
        P_R[j][N] = OverlapEnv();
        for (int i = 0; i < N; ++i)
            P_L[j][i + 1] = build_overlap_left(j, i);
        for (int i = N - 1; i >= 0; --i)
            P_R[j][i] = build_overlap_right(j, i);
    }

    double energy = 0.0;
    mps::SVDWorkspace ws;

    auto optimize_site_excited = [&](int i, bool left_to_right) -> double {
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

        // Build projected vectors for penalty: |phi_j> at sites i,i+1
        // phi_j[al, s0, s1, ar] = P_L[j][i][al,bl] * A_prev[j][i][bl,s0,bm] * B_prev[j][i+1][bm,s1,br] * P_R[j][i+2][ar,br]
        std::vector<std::vector<C>> prev_thetas(n_prev);
        for (int j = 0; j < n_prev; ++j) {
            prev_thetas[j].resize(theta_dim, {0.0, 0.0});
            const mps::Tensor& Ap = prev_states[j]->site(i);
            const mps::Tensor& Bp = prev_states[j]->site(i + 1);
            const auto& Lp = P_L[j][i];
            const auto& Rp = P_R[j][i + 2];
            int chi_mp = Ap.chi_right;

            for (int al = 0; al < chi_l; ++al)
                for (int bl = 0; bl < Ap.chi_left; ++bl) {
                    C lval = Lp.at(al, bl);
                    if (lval == C(0, 0)) continue;
                    for (int s0 = 0; s0 < d; ++s0)
                        for (int bm = 0; bm < chi_mp; ++bm) {
                            C aval = Ap.at(bl, s0, bm);
                            if (aval == C(0, 0)) continue;
                            for (int s1 = 0; s1 < d; ++s1)
                                for (int br = 0; br < Bp.chi_right; ++br) {
                                    C bval = Bp.at(bm, s1, br);
                                    if (bval == C(0, 0)) continue;
                                    for (int ar = 0; ar < chi_r; ++ar)
                                        prev_thetas[j][((al * d + s0) * d + s1) * chi_r + ar]
                                            += lval * aval * bval * Rp.at(ar, br);
                                }
                        }
                }
        }

        // Modified matvec: H_eff * theta + weight * sum_j <phi_j|theta> * |phi_j>
        auto matvec = [&](const C* in, C* out) {
            apply_heff(L, W1, W2, R, in, out, chi_l, chi_r);
            // Add penalty projections
            for (int j = 0; j < n_prev; ++j) {
                C overlap = {0.0, 0.0};
                for (int k = 0; k < theta_dim; ++k)
                    overlap += std::conj(prev_thetas[j][k]) * in[k];
                for (int k = 0; k < theta_dim; ++k)
                    out[k] += weight * overlap * prev_thetas[j][k];
            }
        };

        double e = lanczos(matvec, theta.data(), theta_dim, lanczos_dim, 1e-10);

        // SVD split
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
        energy = optimize_site_excited(i, true);
        L_envs[i + 1] = update_left_env(L_envs[i], psi.site(i), H.site(i));
        for (int j = 0; j < n_prev; ++j)
            P_L[j][i + 1] = build_overlap_left(j, i);
    }

    // Right-to-left sweep
    for (int i = N - 2; i >= 0; --i) {
        energy = optimize_site_excited(i, false);
        R_envs[i + 1] = update_right_env(R_envs[i + 2], psi.site(i + 1), H.site(i + 1));
        for (int j = 0; j < n_prev; ++j)
            P_R[j][i + 1] = build_overlap_right(j, i + 1);
    }

    return energy;
}

}} // namespace qforge::dmrg
