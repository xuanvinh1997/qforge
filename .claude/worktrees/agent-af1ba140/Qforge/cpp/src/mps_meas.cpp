// -*- coding: utf-8 -*-
// mps_meas.cpp — expectation values, entropy, norm for MPS
#include "qforge/mps.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
#include <complex>

namespace qforge { namespace mps { namespace ops {

// ============================================================
// Internal: build [chi_bra, chi_ket] left environment after contracting site i
// L_new[cr_bra, cr_ket] = sum_{cl_bra, cl_ket, s}
//     conj(A[cl_bra, s_bra, cr_bra]) * L[cl_bra, cl_ket] * op[s_bra, s_ket] * A[cl_ket, s_ket, cr_ket]
// With op = identity: s_bra == s_ket
// ============================================================
static void contract_left(
    const std::vector<std::complex<double>>& L,
    int chi_bra, int chi_ket,
    const Tensor& A,
    const std::complex<double> op[4],  // nullptr means identity
    std::vector<std::complex<double>>& L_new,
    int& new_bra, int& new_ket
) {
    new_bra = A.chi_right;
    new_ket = A.chi_right;
    L_new.assign(static_cast<size_t>(new_bra) * new_ket, {0.0, 0.0});
    const int d = 2;
    for (int cl_bra = 0; cl_bra < chi_bra; ++cl_bra)
        for (int cl_ket = 0; cl_ket < chi_ket; ++cl_ket) {
            std::complex<double> lval = L[cl_bra * chi_ket + cl_ket];
            if (lval == std::complex<double>(0, 0)) continue;
            for (int s_ket = 0; s_ket < d; ++s_ket) {
                for (int s_bra = 0; s_bra < d; ++s_bra) {
                    std::complex<double> o = (op == nullptr)
                        ? (s_bra == s_ket ? std::complex<double>(1, 0) : std::complex<double>(0, 0))
                        : op[s_bra * d + s_ket];
                    if (o == std::complex<double>(0, 0)) continue;
                    for (int cr_bra = 0; cr_bra < A.chi_right; ++cr_bra)
                        for (int cr_ket = 0; cr_ket < A.chi_right; ++cr_ket)
                            L_new[cr_bra * new_ket + cr_ket] +=
                                std::conj(A.at(cl_bra, s_bra, cr_bra))
                                * lval * o * A.at(cl_ket, s_ket, cr_ket);
                }
            }
        }
}

// ============================================================
// single_site_expectation: <psi|op_site|psi>
// ============================================================
std::complex<double> ops::single_site_expectation(
    const MPS& mps, int site,
    const std::complex<double> op[4]
) {
    const int N = mps.n_qubits();
    if (site < 0 || site >= N)
        throw std::out_of_range("site out of range");

    std::vector<std::complex<double>> L = {1.0};
    int cb = 1, ck = 1;
    int nb, nk;

    for (int i = 0; i < N; ++i) {
        std::vector<std::complex<double>> L_new;
        const std::complex<double>* use_op = (i == site) ? op : nullptr;
        contract_left(L, cb, ck, mps.site(i), use_op, L_new, nb, nk);
        L = std::move(L_new);
        cb = nb; ck = nk;
    }
    return L[0];
}

// ============================================================
// two_site_expectation: <psi|op_i * op_j|psi>, si < sj
// ============================================================
std::complex<double> ops::two_site_expectation(
    const MPS& mps, int si, int sj,
    const std::complex<double> opi[4],
    const std::complex<double> opj[4]
) {
    if (si >= sj)
        throw std::invalid_argument("si must be < sj");
    const int N = mps.n_qubits();

    std::vector<std::complex<double>> L = {1.0};
    int cb = 1, ck = 1;
    int nb, nk;

    for (int i = 0; i < N; ++i) {
        std::vector<std::complex<double>> L_new;
        const std::complex<double>* use_op =
            (i == si) ? opi : (i == sj) ? opj : nullptr;
        contract_left(L, cb, ck, mps.site(i), use_op, L_new, nb, nk);
        L = std::move(L_new);
        cb = nb; ck = nk;
    }
    return L[0];
}

// ============================================================
// measure_prob0: P(qubit=|0>) = <psi|P0_site|psi>
// ============================================================
double ops::measure_prob0(const MPS& mps, int site) {
    const std::complex<double> P0[4] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
    return ops::single_site_expectation(mps, site, P0).real();
}

// ============================================================
// norm: sqrt(<psi|psi>)
// ============================================================
double ops::norm(const MPS& mps) {
    const int N = mps.n_qubits();
    std::vector<std::complex<double>> L = {1.0};
    int cb = 1, ck = 1;
    for (int i = 0; i < N; ++i) {
        std::vector<std::complex<double>> L_new;
        int nb, nk;
        contract_left(L, cb, ck, mps.site(i), nullptr, L_new, nb, nk);
        L = std::move(L_new);
        cb = nb; ck = nk;
    }
    return std::sqrt(std::max(0.0, L[0].real()));
}

// ============================================================
// entanglement_entropy at bond (bond, bond+1)
// Contract left and right halves, form theta matrix, SVD
// ============================================================
double ops::entanglement_entropy(const MPS& mps, int bond) {
    const int N = mps.n_qubits();
    if (bond < 0 || bond >= N - 1)
        throw std::out_of_range("bond index out of range");

    // --- Left contraction: shape [left_states, chi_bond] ---
    const Tensor& t0 = mps.site(0);
    int chi = t0.chi_right;
    std::vector<std::complex<double>> left(2 * chi);
    for (int s = 0; s < 2; ++s)
        for (int cr = 0; cr < chi; ++cr)
            left[s * chi + cr] = t0.at(0, s, cr);
    int left_states = 2;

    for (int i = 1; i <= bond; ++i) {
        const Tensor& A = mps.site(i);
        int new_chi = A.chi_right;
        int new_states = left_states * 2;
        std::vector<std::complex<double>> new_left(new_states * new_chi, {0, 0});
        for (int os = 0; os < left_states; ++os)
            for (int c = 0; c < chi; ++c) {
                std::complex<double> r = left[os * chi + c];
                if (r == std::complex<double>(0, 0)) continue;
                for (int s = 0; s < 2; ++s)
                    for (int nc = 0; nc < new_chi; ++nc)
                        new_left[(os * 2 + s) * new_chi + nc] += r * A.at(c, s, nc);
            }
        left = std::move(new_left);
        left_states = new_states;
        chi = new_chi;
    }
    // left: [left_states, chi_bond]
    int chi_bond = chi;

    // --- Right contraction: shape [chi_bond, right_states] ---
    const Tensor& tLast = mps.site(N - 1);
    int right_chi = tLast.chi_left;
    int right_states = 2;
    std::vector<std::complex<double>> right(right_chi * right_states);
    for (int cl = 0; cl < right_chi; ++cl)
        for (int s = 0; s < 2; ++s)
            right[cl * right_states + s] = tLast.at(cl, s, 0);

    for (int i = N - 2; i >= bond + 1; --i) {
        const Tensor& A = mps.site(i);
        int new_chi = A.chi_left;
        int new_states = right_states * 2;
        std::vector<std::complex<double>> new_right(new_chi * new_states, {0, 0});
        for (int cl = 0; cl < new_chi; ++cl)
            for (int s = 0; s < 2; ++s)
                for (int c = 0; c < right_chi; ++c) {
                    std::complex<double> a = A.at(cl, s, c);
                    if (a == std::complex<double>(0, 0)) continue;
                    for (int rs = 0; rs < right_states; ++rs)
                        new_right[cl * new_states + s * right_states + rs] +=
                            a * right[c * right_states + rs];
                }
        right = std::move(new_right);
        right_chi = new_chi;
        right_states = new_states;
    }
    // right: [chi_bond, right_states]

    // --- theta = left @ right: [left_states, right_states] ---
    using CMatrix = Eigen::Matrix<std::complex<double>,
                                  Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;

    // Check chi_bond == right_chi
    if (chi_bond != right_chi) {
        // fallback: can happen if bond+1 == N-1 and right not contracted
        // should not happen with correct logic above
        return 0.0;
    }

    CMatrix L_mat = Eigen::Map<const CMatrix>(left.data(), left_states, chi_bond);
    CMatrix R_mat = Eigen::Map<const CMatrix>(right.data(), chi_bond, right_states);
    CMatrix theta = L_mat * R_mat;

    Eigen::JacobiSVD<CMatrix> svd(theta, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& sv = svd.singularValues();

    double norm_sq = 0.0;
    for (int i = 0; i < static_cast<int>(sv.size()); ++i)
        norm_sq += sv[i] * sv[i];
    if (norm_sq < 1e-14) return 0.0;

    double entropy = 0.0;
    for (int i = 0; i < static_cast<int>(sv.size()); ++i) {
        double p = sv[i] * sv[i] / norm_sq;
        if (p > 1e-14) entropy -= p * std::log2(p);
    }
    return entropy;
}

}}} // namespace qforge::mps::ops
