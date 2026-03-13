# -*- coding: utf-8 -*-
"""Tests for qforge.density_matrix module."""
from __future__ import annotations

import numpy as np
import pytest

from qforge.density_matrix import DensityMatrix
from qforge.circuit import Qubit
from qforge import gates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wf_probs(n_qubits, gate_sequence):
    """Run a gate sequence on a Wavefunction and return probabilities."""
    wf = Qubit(n_qubits, backend='python')
    for fn, args in gate_sequence:
        fn(wf, *args)
    return wf.probabilities()


def _dm_probs(n_qubits, gate_sequence):
    """Run a gate sequence on a DensityMatrix and return probabilities."""
    dm = DensityMatrix(n_qubits)
    for fn, args in gate_sequence:
        fn(dm, *args)
    return dm.probabilities()


# ---------------------------------------------------------------------------
# Pure state basics
# ---------------------------------------------------------------------------

class TestDensityMatrixInit:
    def test_init_state(self):
        dm = DensityMatrix(2)
        assert dm.n_qubits == 2
        assert dm.rho.shape == (4, 4)
        assert dm.rho[0, 0] == 1.0
        assert np.sum(np.abs(dm.rho)) == pytest.approx(1.0)

    def test_sv_is_none(self):
        dm = DensityMatrix(1)
        assert dm._sv is None

    def test_is_density_matrix_flag(self):
        dm = DensityMatrix(1)
        assert dm._is_density_matrix is True

    def test_state_labels(self):
        dm = DensityMatrix(2)
        assert list(dm.state) == ['00', '01', '10', '11']


class TestPureStateMatches:
    """DM probabilities must match Wavefunction probabilities for pure states."""

    def test_hadamard(self):
        seq = [(gates.H, (0,))]
        wf_p = _wf_probs(1, seq)
        dm_p = _dm_probs(1, seq)
        np.testing.assert_allclose(dm_p, wf_p, atol=1e-12)

    def test_x_gate(self):
        seq = [(gates.X, (0,))]
        wf_p = _wf_probs(1, seq)
        dm_p = _dm_probs(1, seq)
        np.testing.assert_allclose(dm_p, wf_p, atol=1e-12)

    def test_ry_gate(self):
        seq = [(gates.RY, (0, np.pi / 3))]
        wf_p = _wf_probs(1, seq)
        dm_p = _dm_probs(1, seq)
        np.testing.assert_allclose(dm_p, wf_p, atol=1e-12)

    def test_two_qubit_sequence(self):
        seq = [
            (gates.H, (0,)),
            (gates.RY, (1, 0.7)),
            (gates.Z, (0,)),
        ]
        wf_p = _wf_probs(2, seq)
        dm_p = _dm_probs(2, seq)
        np.testing.assert_allclose(dm_p, wf_p, atol=1e-12)


class TestCNOTBellState:
    def test_bell_state(self):
        """H(0) + CNOT(0,1) should produce |00> + |11> / sqrt(2)."""
        dm = DensityMatrix(2)
        gates.H(dm, 0)
        gates.CNOT(dm, 0, 1)
        probs = dm.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0, 0, 0.5], atol=1e-12)

    def test_bell_state_matches_wf(self):
        seq = [(gates.H, (0,)), (gates.CNOT, (0, 1))]
        wf_p = _wf_probs(2, seq)
        dm_p = _dm_probs(2, seq)
        np.testing.assert_allclose(dm_p, wf_p, atol=1e-12)


# ---------------------------------------------------------------------------
# Noise channel application
# ---------------------------------------------------------------------------

class TestApplyChannel:
    def test_depolarizing(self):
        """Depolarizing channel on |0> should produce mixed state."""
        from qforge.noise import Depolarizing

        dm = DensityMatrix(1)
        channel = Depolarizing(0.1)
        dm.apply_channel(channel.kraus_ops, [0])
        # Trace should still be 1
        assert dm.trace() == pytest.approx(1.0, abs=1e-12)
        # Should no longer be pure
        assert dm.purity() < 1.0 - 1e-6

    def test_bitflip_full(self):
        """BitFlip(1) on |0> -> |1>."""
        from qforge.noise import BitFlip

        dm = DensityMatrix(1)
        dm.apply_channel(BitFlip(1.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [0, 1], atol=1e-12)

    def test_bitflip_zero(self):
        """BitFlip(0) should be identity."""
        from qforge.noise import BitFlip

        dm = DensityMatrix(1)
        rho_before = dm.rho.copy()
        dm.apply_channel(BitFlip(0.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.rho, rho_before, atol=1e-12)


# ---------------------------------------------------------------------------
# Trace, purity, entropy
# ---------------------------------------------------------------------------

class TestTraceAndPurity:
    def test_trace_is_one(self):
        dm = DensityMatrix(2)
        gates.H(dm, 0)
        gates.CNOT(dm, 0, 1)
        assert dm.trace() == pytest.approx(1.0, abs=1e-12)

    def test_pure_state_purity(self):
        dm = DensityMatrix(2)
        gates.H(dm, 0)
        assert dm.purity() == pytest.approx(1.0, abs=1e-10)

    def test_is_pure(self):
        dm = DensityMatrix(1)
        assert dm.is_pure()

    def test_mixed_not_pure(self):
        from qforge.noise import Depolarizing
        dm = DensityMatrix(1)
        dm.apply_channel(Depolarizing(0.5).kraus_ops, [0])
        assert not dm.is_pure()


class TestVonNeumannEntropy:
    def test_pure_state_entropy(self):
        dm = DensityMatrix(1)
        assert dm.von_neumann_entropy() == pytest.approx(0.0, abs=1e-10)

    def test_maximally_mixed_entropy(self):
        """Maximally mixed state of 1 qubit has entropy 1 bit."""
        dm = DensityMatrix(1)
        dm.rho = np.eye(2, dtype=complex) / 2
        assert dm.von_neumann_entropy(base=2) == pytest.approx(1.0, abs=1e-10)

    def test_maximally_mixed_2q(self):
        """Maximally mixed state of 2 qubits has entropy 2 bits."""
        dm = DensityMatrix(2)
        dm.rho = np.eye(4, dtype=complex) / 4
        assert dm.von_neumann_entropy(base=2) == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------

class TestPartialTrace:
    def test_product_state(self):
        """Partial trace of |00> over qubit 1 -> |0><0|."""
        dm = DensityMatrix(2)
        reduced = dm.partial_trace([0])
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(reduced.rho, expected, atol=1e-12)

    def test_bell_state_partial_trace(self):
        """Partial trace of Bell state -> maximally mixed single qubit."""
        dm = DensityMatrix(2)
        gates.H(dm, 0)
        gates.CNOT(dm, 0, 1)
        reduced = dm.partial_trace([0])
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(reduced.rho, expected, atol=1e-10)

    def test_partial_trace_keep_second(self):
        """Partial trace keeping qubit 1 of a product state H|0> x |0>."""
        dm = DensityMatrix(2)
        gates.H(dm, 0)
        reduced = dm.partial_trace([1])
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(reduced.rho, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

class TestFidelity:
    def test_same_state(self):
        dm = DensityMatrix(1)
        gates.H(dm, 0)
        assert dm.fidelity(dm) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_states(self):
        dm0 = DensityMatrix(1)  # |0>
        dm1 = DensityMatrix(1)
        gates.X(dm1, 0)  # |1>
        assert dm0.fidelity(dm1) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

class TestFromWavefunction:
    def test_roundtrip(self):
        """DM from Wavefunction should match direct DM gate application."""
        wf = Qubit(2, backend='python')
        gates.H(wf, 0)
        gates.CNOT(wf, 0, 1)

        dm_from_wf = DensityMatrix.from_wavefunction(wf)
        dm_direct = DensityMatrix(2)
        gates.H(dm_direct, 0)
        gates.CNOT(dm_direct, 0, 1)

        np.testing.assert_allclose(dm_from_wf.rho, dm_direct.rho, atol=1e-12)

    def test_from_state_vector(self):
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        dm = DensityMatrix.from_state_vector(psi, 2)
        np.testing.assert_allclose(dm.probabilities(), [0.5, 0, 0, 0.5], atol=1e-12)
        assert dm.is_pure()


class TestAmplitudeProperty:
    def test_pure_state_amplitude(self):
        dm = DensityMatrix(1)
        gates.H(dm, 0)
        amp = dm.amplitude
        np.testing.assert_allclose(np.abs(amp) ** 2, [0.5, 0.5], atol=1e-10)

    def test_mixed_state_raises(self):
        from qforge.noise import Depolarizing
        dm = DensityMatrix(1)
        dm.apply_channel(Depolarizing(0.5).kraus_ops, [0])
        with pytest.raises(ValueError, match="mixed state"):
            _ = dm.amplitude
