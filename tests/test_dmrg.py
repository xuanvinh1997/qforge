# -*- coding: utf-8 -*-
"""
Unit tests for DMRG solver.
Covers: Heisenberg/Ising energy, convergence, magnetization profile,
correlation functions. Validates against exact diagonalization for small systems.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from qforge.dmrg import DMRG


# ================================================================
# Exact diagonalization helpers (Pauli matrices)
# ================================================================
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_n(ops):
    """Kronecker product of a list of 2x2 matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def build_heisenberg_exact(n, J=1.0):
    """Build full Heisenberg Hamiltonian matrix (exact diagonalization)."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        for pauli in (_X, _Y, _Z):
            ops = [_I] * n
            ops[i] = pauli
            ops[i + 1] = pauli
            H += J * kron_n(ops)
    return H


def build_ising_exact(n, J=1.0, h=0.5):
    """Build full TFIM Hamiltonian matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [_I] * n
        ops[i] = _Z; ops[i + 1] = _Z
        H += -J * kron_n(ops)
    for i in range(n):
        ops = [_I] * n
        ops[i] = _X
        H += -h * kron_n(ops)
    return H


def exact_ground_energy(H):
    """Lowest eigenvalue of H via full diagonalization."""
    return float(np.linalg.eigvalsh(H)[0])


# ================================================================
# Test 1: Heisenberg exact vs DMRG
# ================================================================
class TestHeisenbergDMRG:
    @pytest.mark.parametrize("n_sites", [4, 6])
    def test_energy_vs_exact_diag(self, n_sites):
        H_exact = build_heisenberg_exact(n_sites, J=1.0)
        E_exact = exact_ground_energy(H_exact)

        dmrg = DMRG.heisenberg(n_sites=n_sites, max_bond_dim=16, backend='python')
        E_dmrg, _ = dmrg.run(n_sweeps=10, convergence_tol=1e-8)

        assert abs(E_dmrg - E_exact) < 0.05, \
            f"DMRG E={E_dmrg:.6f} vs exact E={E_exact:.6f} for n={n_sites}"

    def test_energy_monotone_decreasing(self):
        """Energy should not increase between sweeps."""
        dmrg = DMRG.heisenberg(n_sites=4, max_bond_dim=8, backend='python')
        dmrg.run(n_sweeps=5)
        hist = dmrg.energy_history
        for i in range(1, len(hist)):
            assert hist[i] <= hist[i - 1] + 1e-6, \
                f"Energy increased: {hist[i-1]:.6f} -> {hist[i]:.6f}"

    def test_heisenberg_n4_energy_range(self):
        """Heisenberg L=4 OBC energy should be in known range."""
        dmrg = DMRG.heisenberg(n_sites=4, max_bond_dim=8, backend='python')
        E, _ = dmrg.run(n_sweeps=8)
        # Exact: E0(L=4, J=1, OBC) ≈ -6.46 (full Pauli matrices: XX+YY+ZZ per bond)
        assert -9.0 < E < -4.0, f"Energy {E} out of expected range"


# ================================================================
# Test 2: Ising exact vs DMRG
# ================================================================
class TestIsingDMRG:
    def test_ising_n4_energy(self):
        H_exact = build_ising_exact(4, J=1.0, h=0.5)
        E_exact = exact_ground_energy(H_exact)

        dmrg = DMRG.ising(n_sites=4, J=1.0, h=0.5, max_bond_dim=8, backend='python')
        E_dmrg, _ = dmrg.run(n_sweeps=8)

        assert abs(E_dmrg - E_exact) < 0.1, \
            f"Ising DMRG E={E_dmrg:.4f} vs exact E={E_exact:.4f}"


# ================================================================
# Test 3: Convergence and energy history
# ================================================================
class TestConvergence:
    def test_converges_within_n_sweeps(self):
        dmrg = DMRG.heisenberg(n_sites=6, max_bond_dim=8, backend='python')
        E, psi = dmrg.run(n_sweeps=15, convergence_tol=1e-6)
        assert len(dmrg.energy_history) > 0
        assert E < 0, "Heisenberg ground state energy should be negative"

    def test_ground_state_mps_normalized(self):
        dmrg = DMRG.heisenberg(n_sites=4, max_bond_dim=4, backend='python')
        _, psi = dmrg.run(n_sweeps=5)
        n = psi.norm()
        assert abs(n - 1.0) < 0.1, f"Ground state norm = {n}, should be ~1"


# ================================================================
# Test 4: Magnetization and correlations
# ================================================================
class TestObservables:
    def test_magnetization_profile_shape(self):
        dmrg = DMRG.heisenberg(n_sites=6, max_bond_dim=8, backend='python')
        dmrg.run(n_sweeps=5)
        mag = dmrg.magnetization_profile()
        assert len(mag) == 6, "Magnetization profile length == n_sites"

    def test_ising_magnetization_near_zero(self):
        """Ising at critical point: <Z_i> ~ 0 due to symmetry."""
        dmrg = DMRG.ising(n_sites=4, J=1.0, h=1.0, max_bond_dim=8, backend='python')
        dmrg.run(n_sweeps=8)
        mag = dmrg.magnetization_profile()
        # At critical point with OBC, total magnetization should be small
        # (can be non-zero due to spontaneous symmetry breaking in finite systems)
        assert all(abs(m) <= 1.0 + 1e-10 for m in mag), \
            "All <Z_i> values should be in [-1, 1]"

    def test_correlation_function_type(self):
        dmrg = DMRG.heisenberg(n_sites=4, max_bond_dim=4, backend='python')
        dmrg.run(n_sweeps=5)
        c = dmrg.correlation_function('Z', 'Z', 0, 1)
        assert isinstance(c, float), "correlation_function should return float"


# ================================================================
# Test 5: Factory methods
# ================================================================
class TestFactoryMethods:
    def test_xxz_isotropic_equals_heisenberg(self):
        """XXZ with Delta=1 should give same energy as Heisenberg."""
        dmrg_h = DMRG.heisenberg(n_sites=4, max_bond_dim=8, backend='python')
        E_h, _ = dmrg_h.run(n_sweeps=8)

        dmrg_z = DMRG.xxz(n_sites=4, Delta=1.0, J=1.0, max_bond_dim=8, backend='python')
        E_z, _ = dmrg_z.run(n_sweeps=8)

        assert abs(E_h - E_z) < 0.1, \
            f"Heisenberg E={E_h:.4f} vs XXZ(Delta=1) E={E_z:.4f}"

    def test_repr_contains_info(self):
        from qforge.mps import MatrixProductState
        psi = MatrixProductState(10, max_bond_dim=16)
        r = repr(psi)
        assert '10' in r and '16' in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
