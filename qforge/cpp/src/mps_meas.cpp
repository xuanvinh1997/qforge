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
std::complex<double> single_site_expectation(
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
std::complex<double> two_site_expectation(
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
double measure_prob0(const MPS& mps, int site) {
    const std::complex<double> P0[4] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
    return single_site_expectation(mps, site, P0).real();
}

// ============================================================
// norm: sqrt(<psi|psi>)
// ============================================================
double norm(const MPS& mps) {
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
// Uses transfer matrices that stay at chi x chi size — O(N * chi^3)
// instead of expanding to the full 2^N Hilbert space.
//
// For a general (non-canonical) MPS, the squared Schmidt coefficients
// at the bipartition are the eigenvalues of T_L @ T_R, where T_L and
// T_R are the left and right transfer matrices at the bond.
// ============================================================
double entanglement_entropy(const MPS& mps, int bond) {
    const int N = mps.n_qubits();
    if (bond < 0 || bond >= N - 1)
        throw std::out_of_range("bond index out of range");

    using C = std::complex<double>;
    using CMatrix = Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // --- Left transfer matrix: sites 0..bond → T_L [chi_bond, chi_bond] ---
    const Tensor& t0 = mps.site(0);
    int chi = t0.chi_right;
    CMatrix T_L = CMatrix::Zero(chi, chi);
    for (int s = 0; s < 2; ++s)
        for (int a = 0; a < chi; ++a)
            for (int b = 0; b < chi; ++b)
                T_L(a, b) += t0.at(0, s, a) * std::conj(t0.at(0, s, b));

    for (int i = 1; i <= bond; ++i) {
        const Tensor& A = mps.site(i);
        int new_chi = A.chi_right;
        CMatrix T_new = CMatrix::Zero(new_chi, new_chi);
        for (int a = 0; a < chi; ++a)
            for (int b = 0; b < chi; ++b) {
                auto t = T_L(a, b);
                if (std::abs(t) < 1e-15) continue;
                for (int s = 0; s < 2; ++s)
                    for (int ap = 0; ap < new_chi; ++ap) {
                        auto ta = t * A.at(a, s, ap);
                        if (std::abs(ta) < 1e-15) continue;
                        for (int bp = 0; bp < new_chi; ++bp)
                            T_new(ap, bp) += ta * std::conj(A.at(b, s, bp));
                    }
            }
        T_L = std::move(T_new);
        chi = new_chi;
    }
    int chi_bond = chi;

    // --- Right transfer matrix: sites bond+1..N-1 → T_R [chi_bond, chi_bond] ---
    const Tensor& tLast = mps.site(N - 1);
    int rchi = tLast.chi_left;
    CMatrix T_R = CMatrix::Zero(rchi, rchi);
    for (int s = 0; s < 2; ++s)
        for (int a = 0; a < rchi; ++a)
            for (int b = 0; b < rchi; ++b)
                T_R(a, b) += tLast.at(a, s, 0) * std::conj(tLast.at(b, s, 0));

    for (int i = N - 2; i >= bond + 1; --i) {
        const Tensor& A = mps.site(i);
        int new_chi = A.chi_left;
        CMatrix T_new = CMatrix::Zero(new_chi, new_chi);
        for (int a = 0; a < new_chi; ++a)
            for (int b = 0; b < new_chi; ++b) {
                C val(0, 0);
                for (int s = 0; s < 2; ++s)
                    for (int cr = 0; cr < rchi; ++cr)
                        for (int cr2 = 0; cr2 < rchi; ++cr2)
                            val += A.at(a, s, cr) * T_R(cr, cr2)
                                 * std::conj(A.at(b, s, cr2));
                T_new(a, b) = val;
            }
        T_R = std::move(T_new);
        rchi = new_chi;
    }

    // Squared Schmidt coefficients = eigenvalues of T_L * T_R
    CMatrix M = T_L * T_R;
    Eigen::ComplexEigenSolver<CMatrix> eig(M, false);
    const auto& ev = eig.eigenvalues();

    double norm_sq = 0.0;
    for (int i = 0; i < static_cast<int>(ev.size()); ++i)
        norm_sq += std::max(0.0, ev[i].real());
    if (norm_sq < 1e-14) return 0.0;

    double entropy = 0.0;
    for (int i = 0; i < static_cast<int>(ev.size()); ++i) {
        double p = std::max(0.0, ev[i].real()) / norm_sq;
        if (p > 1e-14) entropy -= p * std::log2(p);
    }
    return entropy;
}

}}} // namespace qforge::mps::ops
