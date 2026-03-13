# -*- coding: utf-8 -*-
"""
Unit tests for MatrixProductState (MPS) backend.
Covers: product state, gate application, roundtrip conversion,
Bell state, GHZ state, bond dimensions, entropy, measurements.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from Qforge.mps import MatrixProductState, _init_product_state, _contract_py, _svd_decompose


# ================================================================
# Helpers
# ================================================================
INV_SQRT2 = 1.0 / np.sqrt(2)


def make_mps(n, chi=32, backend='python'):
    """Create a fresh |00...0> MPS."""
    return MatrixProductState(n, max_bond_dim=chi, backend=backend)


def bell_state_mps(backend='python'):
    """Prepare Bell state |00> + |11> / sqrt(2)."""
    from Qforge import gates as Qgates
    psi = make_mps(2, chi=4, backend=backend)
    Qgates.H(psi, 0)
    Qgates.CNOT(psi, 0, 1)
    return psi


def ghz_state_mps(n, backend='python'):
    """Prepare GHZ state (|00...0> + |11...1>) / sqrt(2)."""
    from Qforge import gates as Qgates
    psi = make_mps(n, chi=4, backend=backend)
    Qgates.H(psi, 0)
    for i in range(n - 1):
        Qgates.CNOT(psi, i, i + 1)
    return psi


# ================================================================
# Test 1: Product state initialization
# ================================================================
class TestProductState:
    def test_amplitude_zero(self):
        psi = make_mps(4)
        amp = psi.amplitude
        assert abs(amp[0] - 1.0) < 1e-12, "amp[0] should be 1"
        assert np.allclose(amp[1:], 0), "all other amps should be 0"

    def test_bond_dimensions_product(self):
        psi = make_mps(5)
        assert all(d == 1 for d in psi.bond_dimensions()), \
            "product state bond dims should all be 1"

    def test_norm_product(self):
        psi = make_mps(4)
        assert abs(psi.norm() - 1.0) < 1e-10, "norm should be 1"

    def test_probabilities_product(self):
        psi = make_mps(3)
        probs = psi.probabilities()
        assert abs(probs[0] - 1.0) < 1e-12
        assert np.allclose(probs[1:], 0)


# ================================================================
# Test 2: Single-qubit gates
# ================================================================
class TestSingleQubitGates:
    def test_H_qubit0(self):
        from Qforge import gates as Qgates
        psi = make_mps(2)
        Qgates.H(psi, 0)
        amp = psi.amplitude
        assert abs(amp[0] - INV_SQRT2) < 1e-12, "|+> state: amp[00] = 1/sqrt2"
        assert abs(amp[2] - INV_SQRT2) < 1e-12, "|+> state: amp[10] = 1/sqrt2"
        assert abs(amp[1]) < 1e-12
        assert abs(amp[3]) < 1e-12

    def test_X_gate(self):
        from Qforge import gates as Qgates
        psi = make_mps(2)
        Qgates.X(psi, 0)
        amp = psi.amplitude
        assert abs(amp[2] - 1.0) < 1e-12, "X|0> = |1>, amp[10] = 1"

    def test_Z_gate(self):
        from Qforge import gates as Qgates
        psi = make_mps(2)
        Qgates.H(psi, 0)
        Qgates.Z(psi, 0)
        amp = psi.amplitude
        # H|0> = |+> = (|0>+|1>)/sqrt2
        # Z|+> = (|0>-|1>)/sqrt2 = |->
        assert abs(amp[0] - INV_SQRT2) < 1e-12
        assert abs(amp[2] + INV_SQRT2) < 1e-12

    def test_RX_half_pi(self):
        from Qforge import gates as Qgates
        psi = make_mps(1)
        Qgates.RX(psi, 0, np.pi / 2)
        amp = psi.amplitude
        assert abs(abs(amp[0]) - np.cos(np.pi / 4)) < 1e-10
        assert abs(abs(amp[1]) - np.sin(np.pi / 4)) < 1e-10


# ================================================================
# Test 3: CNOT / Bell state
# ================================================================
class TestBellState:
    def test_bell_state_amplitudes(self):
        psi = bell_state_mps()
        amp = psi.amplitude
        assert abs(amp[0] - INV_SQRT2) < 1e-10, "|00> amplitude"
        assert abs(amp[3] - INV_SQRT2) < 1e-10, "|11> amplitude"
        assert abs(amp[1]) < 1e-10
        assert abs(amp[2]) < 1e-10

    def test_bell_state_bond_dim(self):
        psi = bell_state_mps()
        assert psi.bond_dimensions() == [2], "Bell state has bond dim 2"

    def test_bell_state_entropy(self):
        psi = bell_state_mps()
        s = psi.entanglement_entropy(0)
        assert abs(s - 1.0) < 0.01, "Bell state entropy should be 1 bit"

    def test_bell_state_norm(self):
        psi = bell_state_mps()
        assert abs(psi.norm() - 1.0) < 1e-10


# ================================================================
# Test 4: GHZ state
# ================================================================
class TestGHZState:
    def test_ghz_amplitudes_n4(self):
        psi = ghz_state_mps(4)
        amp = psi.amplitude
        assert abs(amp[0] - INV_SQRT2) < 1e-10, "|0000> amplitude"
        assert abs(amp[15] - INV_SQRT2) < 1e-10, "|1111> amplitude"
        assert np.allclose(amp[1:15], 0, atol=1e-10)

    def test_ghz_bond_dims_chi2(self):
        psi = ghz_state_mps(6)
        dims = psi.bond_dimensions()
        assert all(d <= 2 for d in dims), "GHZ bond dims should be <=2"

    def test_ghz_entropy(self):
        psi = ghz_state_mps(4)
        s = psi.entanglement_entropy(1)
        assert abs(s - 1.0) < 0.05, "GHZ internal bond entropy ~ 1 bit"


# ================================================================
# Test 5: Roundtrip conversion Wavefunction <-> MPS
# ================================================================
class TestRoundtrip:
    def test_from_wavefunction_bell(self):
        from Qforge.wavefunction import Wavefunction
        # Build Bell state via Wavefunction
        states = np.array(['00', '01', '10', '11'])
        amp_bell = np.array([INV_SQRT2, 0, 0, INV_SQRT2], dtype=complex)
        wf = Wavefunction(states, amp_bell)
        psi_mps = MatrixProductState.from_wavefunction(wf, max_bond_dim=4)
        amp_out = psi_mps.amplitude
        assert np.allclose(amp_out, amp_bell, atol=1e-10), \
            "MPS roundtrip should preserve Bell state amplitudes"

    def test_to_wavefunction(self):
        psi = bell_state_mps()
        wf = psi.to_wavefunction()
        amp_bell = np.array([INV_SQRT2, 0, 0, INV_SQRT2], dtype=complex)
        assert np.allclose(wf.amplitude, amp_bell, atol=1e-10)

    def test_svd_decompose_n3(self):
        """Test that SVD decomposition of a known state is exact for chi>=2."""
        amp = np.array([INV_SQRT2, 0, 0, 0, 0, 0, 0, INV_SQRT2], dtype=complex)
        tensors = _svd_decompose(amp, 3, 4, 1e-12)
        reconstructed = _contract_py(tensors)
        assert np.allclose(reconstructed, amp, atol=1e-10)


# ================================================================
# Test 6: Measurement
# ================================================================
class TestMeasurement:
    def test_measure_prob0_zero_state(self):
        psi = make_mps(3)
        from Qforge import measurement as Qmeas
        probs = Qmeas.measure_one(psi, 0)
        assert abs(probs[0] - 1.0) < 1e-10, "P(|0>) = 1 for |0> state"
        assert abs(probs[1]) < 1e-10

    def test_measure_prob0_X_state(self):
        from Qforge import gates as Qgates, measurement as Qmeas
        psi = make_mps(2)
        Qgates.H(psi, 0)
        probs = Qmeas.measure_one(psi, 0)
        assert abs(probs[0] - 0.5) < 1e-10, "P(|0>) = 0.5 for |+> state"
        assert abs(probs[1] - 0.5) < 1e-10

    def test_pauli_expectation_Z(self):
        from Qforge import measurement as Qmeas
        psi = make_mps(2)
        val = Qmeas.pauli_expectation(psi, 0, 'Z')
        assert abs(val - 1.0) < 1e-10, "<Z> = 1 for |0>"

    def test_pauli_expectation_X_plus_state(self):
        from Qforge import gates as Qgates, measurement as Qmeas
        psi = make_mps(2)
        Qgates.H(psi, 0)
        val = Qmeas.pauli_expectation(psi, 0, 'X')
        assert abs(val - 1.0) < 1e-10, "<X> = 1 for |+>"


# ================================================================
# Test 7: SVD truncation and accuracy
# ================================================================
class TestTruncation:
    def test_ghz_exact_chi2(self):
        """GHZ state should be exact with bond dim 2."""
        psi = ghz_state_mps(6, backend='python')
        amp = psi.amplitude
        assert abs(amp[0] - INV_SQRT2) < 1e-10
        assert abs(amp[63] - INV_SQRT2) < 1e-10

    def test_truncation_reduces_chi(self):
        """Tight truncation should reduce bond dims."""
        from Qforge.wavefunction import Wavefunction
        # Random state with low entanglement
        n = 4
        amp = np.zeros(16, dtype=complex)
        amp[0] = INV_SQRT2
        amp[15] = INV_SQRT2
        states = np.array([''.join(s)
                           for s in __import__('itertools').product('01', repeat=4)])
        wf = Wavefunction(states, amp)
        psi = MatrixProductState.from_wavefunction(wf, max_bond_dim=2, eps=1e-10)
        assert max(psi.bond_dimensions()) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
