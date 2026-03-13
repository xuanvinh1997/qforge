# -*- coding: utf-8 -*-
"""Tests for qforge.noise module."""
from __future__ import annotations

import numpy as np
import pytest

from qforge.noise import (
    BitFlip,
    PhaseFlip,
    Depolarizing,
    AmplitudeDamping,
    PhaseDamping,
    ThermalRelaxation,
    ReadoutError,
    KrausChannel,
    NoiseModel,
)
from qforge.density_matrix import DensityMatrix
from qforge import gates


# ---------------------------------------------------------------------------
# Trace preservation: sum_k K_k^dag K_k = I
# ---------------------------------------------------------------------------

class TestTracePreservation:
    @pytest.mark.parametrize("channel", [
        BitFlip(0.0),
        BitFlip(0.3),
        BitFlip(1.0),
        PhaseFlip(0.2),
        Depolarizing(0.1),
        Depolarizing(1.0),
        AmplitudeDamping(0.0),
        AmplitudeDamping(0.5),
        AmplitudeDamping(1.0),
        PhaseDamping(0.3),
        ThermalRelaxation(100, 80, 10),
        ThermalRelaxation(100, 200, 10),
    ])
    def test_trace_preserving(self, channel):
        assert channel.is_trace_preserving(atol=1e-10)


# ---------------------------------------------------------------------------
# BitFlip
# ---------------------------------------------------------------------------

class TestBitFlip:
    def test_identity(self):
        """BitFlip(0) should leave |0> unchanged."""
        dm = DensityMatrix(1)
        dm.apply_channel(BitFlip(0.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [1, 0], atol=1e-12)

    def test_full_flip(self):
        """BitFlip(1) should flip |0> -> |1>."""
        dm = DensityMatrix(1)
        dm.apply_channel(BitFlip(1.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [0, 1], atol=1e-12)

    def test_partial_flip(self):
        """BitFlip(0.3) on |0> should give P(0)=0.7, P(1)=0.3."""
        dm = DensityMatrix(1)
        dm.apply_channel(BitFlip(0.3).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [0.7, 0.3], atol=1e-12)


# ---------------------------------------------------------------------------
# AmplitudeDamping
# ---------------------------------------------------------------------------

class TestAmplitudeDamping:
    def test_gamma_zero(self):
        """gamma=0 should be identity."""
        dm = DensityMatrix(1)
        gates.X(dm, 0)  # |1>
        dm.apply_channel(AmplitudeDamping(0.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [0, 1], atol=1e-12)

    def test_gamma_one_decays_to_zero(self):
        """gamma=1: |1> fully decays to |0>."""
        dm = DensityMatrix(1)
        gates.X(dm, 0)  # |1>
        dm.apply_channel(AmplitudeDamping(1.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [1, 0], atol=1e-12)

    def test_ground_state_unchanged(self):
        """Amplitude damping should not affect |0>."""
        dm = DensityMatrix(1)
        dm.apply_channel(AmplitudeDamping(0.5).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [1, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Depolarizing
# ---------------------------------------------------------------------------

class TestDepolarizing:
    def test_zero_noise(self):
        dm = DensityMatrix(1)
        gates.H(dm, 0)
        rho_before = dm.rho.copy()
        dm.apply_channel(Depolarizing(0.0).kraus_ops, [0])
        np.testing.assert_allclose(dm.rho, rho_before, atol=1e-12)

    def test_full_depolarizing(self):
        """p=4/3 should map I/2 to I/2 (maximally mixed state is fixed point)."""
        dm = DensityMatrix(1)
        dm.rho = np.eye(2, dtype=complex) / 2  # Start from maximally mixed
        dm.apply_channel(Depolarizing(4 / 3).kraus_ops, [0])
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(dm.rho, expected, atol=1e-10)

    def test_on_zero_state(self):
        """Depolarizing(p) on |0>: rho = (1-p)|0><0| + p/3 * I/2.
        Diagonal should be [1 - p/2, p/2] (using Kraus formulation)."""
        p = 0.2
        dm = DensityMatrix(1)
        dm.apply_channel(Depolarizing(p).kraus_ops, [0])
        # rho_00 = 1 - p/2, rho_11 = p/2
        np.testing.assert_allclose(dm.probabilities(), [1 - p / 2, p / 2], atol=1e-12)


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

class TestNoiseModel:
    def test_add_and_get_all_qubit(self):
        model = NoiseModel()
        ch = Depolarizing(0.01)
        model.add_all_qubit_quantum_error(ch, ['H', 'X'])
        errors = model.get_errors('H', [0])
        assert len(errors) == 1
        assert errors[0] is ch

    def test_qubit_specific_error(self):
        model = NoiseModel()
        ch = BitFlip(0.1)
        model.add_quantum_error(ch, ['CNOT'], [0, 1])
        errors = model.get_errors('CNOT', [0, 1])
        assert len(errors) == 1
        errors_other = model.get_errors('CNOT', [1, 2])
        assert len(errors_other) == 0

    def test_combined_errors(self):
        model = NoiseModel()
        ch_all = Depolarizing(0.01)
        ch_specific = BitFlip(0.05)
        model.add_all_qubit_quantum_error(ch_all, ['H'])
        model.add_quantum_error(ch_specific, ['H'], [0])
        errors = model.get_errors('H', [0])
        assert len(errors) == 2

    def test_is_empty(self):
        model = NoiseModel()
        assert model.is_empty
        model.add_all_qubit_quantum_error(BitFlip(0.1), ['X'])
        assert not model.is_empty

    def test_readout_error(self):
        model = NoiseModel()
        re = ReadoutError(0.05, 0.03)
        model.add_readout_error(re, [0])
        assert model.get_readout_error(0) is re
        assert model.get_readout_error(1) is None


class TestNoiseModelIntegration:
    """Test applying a noise model during density matrix simulation."""

    def test_noisy_hadamard(self):
        """H gate followed by depolarizing noise on DM."""
        model = NoiseModel()
        p = 0.1
        model.add_all_qubit_quantum_error(Depolarizing(p), ['H'])

        dm = DensityMatrix(1)
        # Apply H
        gates.H(dm, 0)
        # Apply noise
        for ch in model.get_errors('H', [0]):
            dm.apply_channel(ch.kraus_ops, [0])

        assert dm.trace() == pytest.approx(1.0, abs=1e-12)
        # Should be slightly mixed
        assert dm.purity() < 1.0 - 1e-6

    def test_noisy_cnot_preserves_trace(self):
        """CNOT with depolarizing noise should preserve trace."""
        model = NoiseModel()
        model.add_all_qubit_quantum_error(Depolarizing(0.05), ['CNOT'])

        dm = DensityMatrix(2)
        gates.H(dm, 0)
        gates.CNOT(dm, 0, 1)
        # Apply noise on each qubit of the CNOT
        for ch in model.get_errors('CNOT', [0, 1]):
            dm.apply_channel(ch.kraus_ops, [0])
            dm.apply_channel(ch.kraus_ops, [1])

        assert dm.trace() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# PhaseFlip, PhaseDamping
# ---------------------------------------------------------------------------

class TestPhaseFlip:
    def test_preserves_diagonal(self):
        """Phase flip should not change measurement probabilities of |0>."""
        dm = DensityMatrix(1)
        dm.apply_channel(PhaseFlip(0.5).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), [1, 0], atol=1e-12)


class TestPhaseDamping:
    def test_preserves_diagonal(self):
        """Phase damping preserves populations."""
        dm = DensityMatrix(1)
        gates.H(dm, 0)  # |+>
        probs_before = dm.probabilities().copy()
        dm.apply_channel(PhaseDamping(0.5).kraus_ops, [0])
        np.testing.assert_allclose(dm.probabilities(), probs_before, atol=1e-12)


# ---------------------------------------------------------------------------
# ThermalRelaxation
# ---------------------------------------------------------------------------

class TestThermalRelaxation:
    def test_zero_gate_time(self):
        ch = ThermalRelaxation(100, 80, 0)
        ops = ch.kraus_ops
        assert len(ops) == 1
        np.testing.assert_allclose(ops[0], np.eye(2), atol=1e-12)

    def test_invalid_T2(self):
        with pytest.raises(ValueError):
            ThermalRelaxation(100, 300, 10)


# ---------------------------------------------------------------------------
# KrausChannel
# ---------------------------------------------------------------------------

class TestKrausChannel:
    def test_custom_channel(self):
        """Identity channel via KrausChannel."""
        ch = KrausChannel([np.eye(2)])
        assert ch.is_trace_preserving()
        dm = DensityMatrix(1)
        gates.H(dm, 0)
        rho_before = dm.rho.copy()
        dm.apply_channel(ch.kraus_ops, [0])
        np.testing.assert_allclose(dm.rho, rho_before, atol=1e-12)


# ---------------------------------------------------------------------------
# ReadoutError
# ---------------------------------------------------------------------------

class TestReadoutError:
    def test_confusion_matrix(self):
        re = ReadoutError(0.05, 0.03)
        # confusion_matrix[0,0] = 1 - p1_given_0 = 0.97
        assert re.confusion_matrix[0, 0] == pytest.approx(0.97)
        # confusion_matrix[0,1] = p0_given_1 = 0.05
        assert re.confusion_matrix[0, 1] == pytest.approx(0.05)

    def test_apply_to_probabilities(self):
        re = ReadoutError(0.0, 0.0)  # Perfect readout
        probs = np.array([0.7, 0.3])
        result = re.apply_to_probabilities(probs)
        np.testing.assert_allclose(result, probs, atol=1e-12)
