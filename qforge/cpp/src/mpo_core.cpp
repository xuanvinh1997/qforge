// -*- coding: utf-8 -*-
// mpo_core.cpp — MPO construction for standard 1D Hamiltonians
//
// MPO structure follows the standard "zipper" (finite state machine) form.
// For a chain of N sites with nearest-neighbor interactions, the MPO bond
// dimension is small (3–5), making DMRG very efficient.
//
// Pauli matrices (d=2):
//   I = [[1,0],[0,1]]
//   X = [[0,1],[1,0]]
//   Y = [[0,-i],[i,0]]
//   Z = [[1,0],[0,-1]]
//   S+ = [[0,1],[0,0]],  S- = [[0,0],[1,0]]
#include "qforge/mpo.h"
#include <complex>
#include <stdexcept>

namespace qforge { namespace mpo {

using C = std::complex<double>;

// ============================================================
// MPO constructor
// ============================================================
MPO::MPO(int n_sites, int bond_dim)
    : n_sites_(n_sites), bond_dim_(bond_dim)
{
    if (n_sites < 2)
        throw std::invalid_argument("n_sites must be >= 2");
    tensors_.resize(n_sites);
}

// ============================================================
// Helper: set a 2x2 Pauli operator into an MPO tensor slot
// MPOTensor.at(wl, wr, sb, sk) += alpha * op[sb, sk]
// ============================================================
static void add_op(MPOTensor& W, int wl, int wr,
                   const C op[4], C alpha = {1.0, 0.0}) {
    for (int sb = 0; sb < 2; ++sb)
        for (int sk = 0; sk < 2; ++sk)
            W.at(wl, wr, sb, sk) += alpha * op[sb * 2 + sk];
}

// Pauli matrices stored row-major [sb, sk]
static const C _I[4]  = {{1,0},{0,0},{0,0},{1,0}};
static const C _X[4]  = {{0,0},{1,0},{1,0},{0,0}};
static const C _Y[4]  = {{0,0},{0,-1},{0,1},{0,0}};
static const C _Z[4]  = {{1,0},{0,0},{0,0},{-1,0}};
static const C _Sp[4] = {{0,0},{1,0},{0,0},{0,0}};  // S+ = [[0,1],[0,0]]
static const C _Sm[4] = {{0,0},{0,0},{1,0},{0,0}};  // S- = [[0,0],[1,0]]

// ============================================================
// Heisenberg Hamiltonian: H = J * sum_i (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1})
//
// MPO bond dim = 5. FSM states: [I, S+, S-, Sz, I_right]
// W =  [ I    0    0    0   0  ]
//      [ S-   0    0    0   0  ]
//      [ S+   0    0    0   0  ]
//      [ Sz   0    0    0   0  ]
//      [ 0   J*S+ J*S- J*Sz I  ]
//
// Rows label w_left, columns label w_right.
// Boundary: left tensor has w_left=1 (bottom row), right tensor has w_right=1 (rightmost col).
// ============================================================
MPO heisenberg(int n_sites, double J) {
    const int W = 5;
    MPO mpo(n_sites, W);

    for (int site = 0; site < n_sites; ++site) {
        bool first = (site == 0);
        bool last  = (site == n_sites - 1);
        int wl = first ? 1 : W;
        int wr = last  ? 1 : W;
        MPOTensor& T = mpo.site(site);
        T = MPOTensor(wl, wr);

        // Indices into FSM: 0=I_left, 1=S-, 2=S+, 3=Sz, 4=I_right
        // (adjusted for boundary tensors which have wl=1 or wr=1)

        auto idx_l = [&](int i) { return first ? 0 : i; };
        auto idx_r = [&](int i) { return last  ? 0 : i; };

        if (first) {
            // Only bottom row (w_left = 1, index 0 in compressed form)
            // Output to all right indices: [S-, S+, Sz, I]
            add_op(T, idx_l(4), idx_r(1), _Sm);           // S-
            add_op(T, idx_l(4), idx_r(2), _Sp);           // S+
            add_op(T, idx_l(4), idx_r(3), _Z, {0.5*J,0}); // Sz (Z/2 for Heisenberg convention using S=1/2)
            // Actually for spin-1/2 Heisenberg: X@X + Y@Y + Z@Z = 2*(S+@S- + S-@S+) + Z@Z
            // We use: XX + YY = 2(S+S- + S-S+), and Z (no factor of 0.5)
            // Reset and use correct convention
        }
        // Re-implement with correct convention
        T.zero();

        if (first && last) {
            // Single-site MPO (degenerate): just identity
            add_op(T, 0, 0, _I);
            continue;
        }

        if (first) {
            // w_left=1, w_right=W
            // Row 0 maps to: emit operators that start interactions
            // T[0, 0=I_r] = I  (passthrough for final identity)
            add_op(T, 0, idx_r(0), _I);   // row index from left is 0 (single row)
            // Actually: first site starts chains, so we emit the left operators:
            add_op(T, 0, idx_r(1), _Sp, {J,0});   // start S+ chain (for Y: J*Y/2i -> use J*Sp*J*Sm)
            add_op(T, 0, idx_r(2), _Sm, {J,0});   // start S- chain
            add_op(T, 0, idx_r(3), _Z, {J,0});    // start Sz chain
            add_op(T, 0, idx_r(4), _I);            // identity passthrough to last col
        } else if (last) {
            // w_left=W, w_right=1
            add_op(T, idx_l(0), 0, _I);           // identity from first col
            add_op(T, idx_l(1), 0, _Sm);          // close S+ chain with S-
            add_op(T, idx_l(2), 0, _Sp);          // close S- chain with S+
            add_op(T, idx_l(3), 0, _Z);           // close Sz chain with Sz
            add_op(T, idx_l(4), 0, _I);           // final identity
        } else {
            // Bulk site: w_left=W, w_right=W
            // Identity passthrough along diagonal edges
            add_op(T, idx_l(0), idx_r(0), _I);   // top-left: identity from boundary
            add_op(T, idx_l(4), idx_r(4), _I);   // bottom-right: identity to boundary
            // Start new interaction chains from boundary row (w_l=4)
            add_op(T, idx_l(4), idx_r(1), _Sp, {J,0});
            add_op(T, idx_l(4), idx_r(2), _Sm, {J,0});
            add_op(T, idx_l(4), idx_r(3), _Z, {J,0});
            // Close existing chains
            add_op(T, idx_l(1), idx_r(0), _Sm);
            add_op(T, idx_l(2), idx_r(0), _Sp);
            add_op(T, idx_l(3), idx_r(0), _Z);
        }
    }
    return mpo;
}

// ============================================================
// Transverse-field Ising: H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
// MPO bond dim = 3. FSM states: [I_left, Sz, I_right]
//
// W = [ I      0    0  ]
//     [ -J*Sz  0    0  ]
//     [ -h*X   Sz   I  ]
// ============================================================
MPO ising(int n_sites, double J, double h) {
    const int W = 3;
    MPO mpo(n_sites, W);

    for (int site = 0; site < n_sites; ++site) {
        bool first = (site == 0);
        bool last  = (site == n_sites - 1);
        int wl = first ? 1 : W;
        int wr = last  ? 1 : W;
        MPOTensor& T = mpo.site(site);
        T = MPOTensor(wl, wr);
        T.zero();

        if (first && last) {
            add_op(T, 0, 0, _X, {-h, 0});
            continue;
        }

        if (first) {
            add_op(T, 0, 0, _Z, {-J, 0});         // start ZZ chain
            add_op(T, 0, 1, _I);                   // identity passthrough
            add_op(T, 0, 2, _X, {-h, 0});          // local X term — wait: layout
            // Correct layout: T[wl, wr, ...]:
            // First site has w_left=1 so only row 0
            // Let wr=0 be left boundary (I), wr=1 be Sz started, wr=2 be right I
            T.zero();
            add_op(T, 0, 0, _X, {-h, 0});  // -h*X into right boundary (slot 0=I_right)
            add_op(T, 0, 1, _Z, {-J, 0});  // -J*Sz starts ZZ chain at slot 1
            add_op(T, 0, 2, _I);            // identity to right-boundary accumulator
        } else if (last) {
            T.zero();
            add_op(T, 0, 0, _I);            // close left boundary
            add_op(T, 1, 0, _Z);            // close ZZ chain: Sz
            add_op(T, 2, 0, _X, {-h, 0});  // local X term
        } else {
            T.zero();
            add_op(T, 0, 0, _I);            // left boundary passthrough
            add_op(T, 1, 0, _Z);            // close ZZ chain
            add_op(T, 2, 0, _X, {-h, 0});  // local X
            add_op(T, 2, 1, _Z, {-J, 0});  // start new ZZ chain
            add_op(T, 2, 2, _I);            // right boundary passthrough
        }
    }
    return mpo;
}

// ============================================================
// XXZ: H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1})
// Same structure as Heisenberg but with different coupling for Z
// ============================================================
MPO xxz(int n_sites, double Delta, double J) {
    const int W = 5;
    MPO mpo(n_sites, W);

    for (int site = 0; site < n_sites; ++site) {
        bool first = (site == 0);
        bool last  = (site == n_sites - 1);
        int wl = first ? 1 : W;
        int wr = last  ? 1 : W;
        MPOTensor& T = mpo.site(site);
        T = MPOTensor(wl, wr);
        T.zero();

        if (first && last) {
            add_op(T, 0, 0, _I);
            continue;
        }

        auto idx_l = [&](int i) { return first ? 0 : i; };
        auto idx_r = [&](int i) { return last  ? 0 : i; };

        if (first) {
            add_op(T, 0, idx_r(1), _Sp, {J,0});
            add_op(T, 0, idx_r(2), _Sm, {J,0});
            add_op(T, 0, idx_r(3), _Z, {J*Delta,0});
            add_op(T, 0, idx_r(4), _I);
        } else if (last) {
            add_op(T, idx_l(0), 0, _I);
            add_op(T, idx_l(1), 0, _Sm);
            add_op(T, idx_l(2), 0, _Sp);
            add_op(T, idx_l(3), 0, _Z);
            add_op(T, idx_l(4), 0, _I);
        } else {
            add_op(T, idx_l(0), idx_r(0), _I);
            add_op(T, idx_l(4), idx_r(4), _I);
            add_op(T, idx_l(4), idx_r(1), _Sp, {J,0});
            add_op(T, idx_l(4), idx_r(2), _Sm, {J,0});
            add_op(T, idx_l(4), idx_r(3), _Z, {J*Delta,0});
            add_op(T, idx_l(1), idx_r(0), _Sm);
            add_op(T, idx_l(2), idx_r(0), _Sp);
            add_op(T, idx_l(3), idx_r(0), _Z);
        }
    }
    return mpo;
}

}} // namespace qforge::mpo
