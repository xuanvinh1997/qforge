// -*- coding: utf-8 -*-
// mps_gates.cpp — single-qubit and two-qubit gate application to MPS
#include "qforge/mps.h"
#include <cstring>
#include <stdexcept>

namespace qforge { namespace mps { namespace ops {

// ============================================================
// apply_single_qubit_gate
// gate[4]: row-major 2x2 unitary [[g00, g01], [g10, g11]]
// In-place update of site tensor: no SVD needed.
// A_new[cl, s_out, cr] = sum_s gate[s_out, s] * A[cl, s, cr]
// ============================================================
void apply_single_qubit_gate(MPS& mps, int site,
                              const std::complex<double> gate[4]) {
    if (site < 0 || site >= mps.n_qubits())
        throw std::out_of_range("site index out of range");

    Tensor& A = mps.site(site);
    const int chi_l = A.chi_left;
    const int chi_r = A.chi_right;
    const int d = 2;

    // Temporary copy to avoid in-place aliasing
    std::vector<std::complex<double>> tmp(A.data);

    for (int cl = 0; cl < chi_l; ++cl) {
        for (int cr = 0; cr < chi_r; ++cr) {
            // Read old amplitudes for this (cl, cr) pair
            const std::complex<double> a0 = tmp[cl * d * chi_r + 0 * chi_r + cr];
            const std::complex<double> a1 = tmp[cl * d * chi_r + 1 * chi_r + cr];
            // gate = [[g00, g01], [g10, g11]] acts as: out_s = sum_s' gate[s, s'] * in_s'
            A.at(cl, 0, cr) = gate[0] * a0 + gate[1] * a1;  // g00*a0 + g01*a1
            A.at(cl, 1, cr) = gate[2] * a0 + gate[3] * a1;  // g10*a0 + g11*a1
        }
    }
}

// ============================================================
// apply_two_qubit_gate (nearest-neighbor only: sites i and i+1)
// gate[16]: row-major 4x4 unitary in computational basis
//   ordering: |00>, |01>, |10>, |11>  (qubit i is MSB)
// Steps:
//   1. Merge tensors i, i+1 into theta [chi_l, d, d, chi_r]
//   2. Apply gate: theta_out[cl, s0, s1, cr] = sum_{s0',s1'} gate[s0*2+s1, s0'*2+s1'] * theta[cl, s0', s1', cr]
//   3. SVD-split back with truncation
// Returns truncation error.
// ============================================================
double apply_two_qubit_gate(MPS& mps, int site_i,
                             const std::complex<double> gate[16],
                             int max_chi, double eps, SVDWorkspace& ws) {
    if (site_i < 0 || site_i >= mps.n_qubits() - 1)
        throw std::out_of_range("site_i must be in [0, n_qubits-2]");

    const Tensor& A = mps.site(site_i);
    const Tensor& B = mps.site(site_i + 1);
    const int chi_l = A.chi_left;
    const int chi_r = B.chi_right;
    const int chi_m = A.chi_right;  // middle (shared) bond dim; must == B.chi_left
    const int d = 2;

    if (chi_m != B.chi_left)
        throw std::runtime_error("bond dimension mismatch between sites");

    // Step 1: Merge into theta [chi_l, d, d, chi_r]
    // theta[cl, s0, s1, cr] = sum_m A[cl, s0, m] * B[m, s1, cr]
    const int theta_size = chi_l * d * d * chi_r;
    std::vector<std::complex<double>> theta(theta_size, {0.0, 0.0});
    for (int cl = 0; cl < chi_l; ++cl)
        for (int s0 = 0; s0 < d; ++s0)
            for (int m = 0; m < chi_m; ++m) {
                std::complex<double> a = A.at(cl, s0, m);
                if (a == std::complex<double>(0.0, 0.0)) continue;
                for (int s1 = 0; s1 < d; ++s1)
                    for (int cr = 0; cr < chi_r; ++cr)
                        theta[((cl * d + s0) * d + s1) * chi_r + cr] += a * B.at(m, s1, cr);
            }

    // Step 2: Apply gate
    // theta_out[cl, so0, so1, cr] = sum_{si0, si1} gate[so0*2+so1, si0*2+si1] * theta[cl, si0, si1, cr]
    std::vector<std::complex<double>> theta_out(theta_size, {0.0, 0.0});
    for (int cl = 0; cl < chi_l; ++cl)
        for (int cr = 0; cr < chi_r; ++cr)
            for (int si0 = 0; si0 < d; ++si0)
                for (int si1 = 0; si1 < d; ++si1) {
                    std::complex<double> val = theta[((cl * d + si0) * d + si1) * chi_r + cr];
                    if (val == std::complex<double>(0.0, 0.0)) continue;
                    int col = si0 * d + si1;
                    for (int so0 = 0; so0 < d; ++so0)
                        for (int so1 = 0; so1 < d; ++so1)
                            theta_out[((cl * d + so0) * d + so1) * chi_r + cr]
                                += gate[(so0 * d + so1) * 4 + col] * val;
                }

    // Step 3: SVD split — reshape theta_out to [chi_l * d, d * chi_r]
    // then split into A_new [chi_l, d, new_chi] and B_new [new_chi, d, chi_r]
    // theta is already in row-major [chi_l, d, d, chi_r]
    // reshape to [(chi_l * d), (d * chi_r)]: done implicitly since row-major

    Tensor A_new, B_new;
    int new_chi = 0;
    double trunc_err = ops::svd_split(theta_out.data(), chi_l, chi_r,
                                      A_new, B_new, new_chi, max_chi, eps, ws);

    mps.site(site_i) = std::move(A_new);
    mps.site(site_i + 1) = std::move(B_new);
    return trunc_err;
}

}}} // namespace qforge::mps::ops
