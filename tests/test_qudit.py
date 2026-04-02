# -*- coding: utf-8 -*-
"""Tests for qudit (d>2) gate library."""
import numpy as np
import pytest
from scipy.linalg import expm

from qforge.circuit import Qudit
from qforge.qudit_gates import (
    GELL_MANN,
    X01, X02, X12, CLOCK, ZPHASE, Hd,
    R01, R02, R12, RGM,
    CSUM, QUDIT_SWAP,
    measure_qudit, collapse_qudit, qudit_expectation,
    apply_qudit_gate,
    _givens_gate, _phase_givens_gate, _infer_dimension,
)

BACKEND = 'python'


def _qutrit(n=1):
    """Helper: create an n-qutrit circuit (d=3) with Python backend."""
    return Qudit(n, dimension=3, backend=BACKEND)


# ============================================================
# Gate unitarity
# ============================================================

class TestUnitarity:
    """All standard gates should produce unitary matrices."""

    @staticmethod
    def _is_unitary(gate, atol=1e-12):
        u = np.asarray(gate, dtype=complex)
        eye = np.eye(u.shape[0], dtype=complex)
        return np.allclose(u @ u.conj().T, eye, atol=atol)

    def test_givens_gate_unitary(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            assert self._is_unitary(_givens_gate(3, 0, 1, theta))
            assert self._is_unitary(_givens_gate(3, 0, 2, theta))
            assert self._is_unitary(_givens_gate(3, 1, 2, theta))

    def test_phase_givens_gate_unitary(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            assert self._is_unitary(_phase_givens_gate(3, 0, 1, theta))
            assert self._is_unitary(_phase_givens_gate(3, 0, 2, theta))
            assert self._is_unitary(_phase_givens_gate(3, 1, 2, theta))

    def test_clock_unitary(self):
        d = 3
        gate = np.zeros((d, d), dtype=complex)
        for k in range(d):
            gate[(k + 1) % d, k] = 1.0
        assert self._is_unitary(gate)

    def test_zphase_unitary(self):
        d = 3
        omega = np.exp(2j * np.pi / d)
        gate = np.diag([omega**k for k in range(d)])
        assert self._is_unitary(gate)

    def test_hadamard_unitary(self):
        d = 3
        omega = np.exp(2j * np.pi / d)
        gate = np.array([[omega**(j * k) for k in range(d)]
                         for j in range(d)], dtype=complex) / np.sqrt(d)
        assert self._is_unitary(gate)


# ============================================================
# Subspace swap gates
# ============================================================

class TestSubspaceSwaps:

    def test_x01(self):
        wf = _qutrit(1)
        # |0> -> |1>
        X01(wf, 0)
        expected = np.array([0, 1, 0], dtype=complex)
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_x01_roundtrip(self):
        wf = _qutrit(1)
        X01(wf, 0)
        X01(wf, 0)
        expected = np.array([1, 0, 0], dtype=complex)
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_x02(self):
        wf = _qutrit(1)
        # |0> -> |2>
        X02(wf, 0)
        expected = np.array([0, 0, 1], dtype=complex)
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_x12(self):
        wf = _qutrit(1)
        X01(wf, 0)  # |0> -> |1>
        X12(wf, 0)  # |1> -> |2>
        expected = np.array([0, 0, 1], dtype=complex)
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)


# ============================================================
# CLOCK gate
# ============================================================

class TestClock:

    def test_clock_cycle(self):
        """CLOCK^3 = identity for d=3."""
        wf = _qutrit(1)
        CLOCK(wf, 0)
        np.testing.assert_allclose(wf.amplitude, [0, 1, 0], atol=1e-12)
        CLOCK(wf, 0)
        np.testing.assert_allclose(wf.amplitude, [0, 0, 1], atol=1e-12)
        CLOCK(wf, 0)
        np.testing.assert_allclose(wf.amplitude, [1, 0, 0], atol=1e-12)


# ============================================================
# ZPHASE gate
# ============================================================

class TestZPhase:

    def test_zphase_on_superposition(self):
        wf = _qutrit(1)
        # Prepare uniform superposition
        Hd(wf, 0)
        amp_before = wf.amplitude.copy()
        ZPHASE(wf, 0)
        omega = np.exp(2j * np.pi / 3)
        expected = amp_before * np.array([1, omega, omega**2])
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)


# ============================================================
# Qudit Hadamard (DFT)
# ============================================================

class TestHadamard:

    def test_hd_uniform_superposition(self):
        """Hd|0> should give uniform superposition 1/sqrt(d) for all levels."""
        wf = _qutrit(1)
        Hd(wf, 0)
        expected_prob = 1.0 / 3.0
        probs = np.abs(wf.amplitude) ** 2
        np.testing.assert_allclose(probs, [expected_prob] * 3, atol=1e-12)

    def test_hd_fourth_power(self):
        """Hd^4 = identity (DFT property)."""
        wf = _qutrit(1)
        Hd(wf, 0)
        Hd(wf, 0)
        Hd(wf, 0)
        Hd(wf, 0)
        np.testing.assert_allclose(wf.amplitude, [1, 0, 0], atol=1e-10)


# ============================================================
# Givens rotations
# ============================================================

class TestGivens:

    def test_r01_pi_swaps(self):
        """R01(pi) should swap |0> and |1> (up to sign)."""
        wf = _qutrit(1)
        R01(wf, 0, np.pi)
        # cos(pi/2)=0, sin(pi/2)=1 => |0> -> |1>
        np.testing.assert_allclose(np.abs(wf.amplitude), [0, 1, 0], atol=1e-12)

    def test_r02_pi_swaps(self):
        wf = _qutrit(1)
        R02(wf, 0, np.pi)
        np.testing.assert_allclose(np.abs(wf.amplitude), [0, 0, 1], atol=1e-12)

    def test_r12_identity_at_zero(self):
        """R12(0) should be identity."""
        wf = _qutrit(1)
        R12(wf, 0, 0.0)
        np.testing.assert_allclose(wf.amplitude, [1, 0, 0], atol=1e-12)


# ============================================================
# RGM (Gell-Mann rotations)
# ============================================================

class TestRGM:

    @pytest.mark.parametrize("gen", range(1, 9))
    def test_rgm_matches_expm(self, gen):
        """RGM result should match scipy.linalg.expm reference."""
        angle = 0.7
        lam = GELL_MANN[gen - 1]
        expected_gate = expm(-1j * angle / 2.0 * lam)

        wf = _qutrit(1)
        RGM(wf, 0, gen, angle)

        # Apply expected_gate to |0> manually
        expected_amp = expected_gate @ np.array([1, 0, 0], dtype=complex)
        np.testing.assert_allclose(wf.amplitude, expected_amp, atol=1e-10)


# ============================================================
# CSUM (qutrit CNOT analog)
# ============================================================

class TestCSUM:

    def test_csum_control_0(self):
        """CSUM with control=|0>: target unchanged."""
        wf = _qutrit(2)  # |00>
        CSUM(wf, 0, 1)
        # |0,0> -> |0, (0+0)%3> = |0,0>
        expected = np.zeros(9, dtype=complex)
        expected[0] = 1.0  # |00>
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_csum_control_1(self):
        """CSUM with control=|1>: |1,0> -> |1,1>."""
        wf = _qutrit(2)
        X01(wf, 0)  # |10>
        CSUM(wf, 0, 1)
        # |1,0> -> |1, (0+1)%3> = |1,1>
        expected = np.zeros(9, dtype=complex)
        expected[4] = 1.0  # index 1*3+1 = 4
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_csum_control_2(self):
        """CSUM with control=|2>: |2,0> -> |2,2>."""
        wf = _qutrit(2)
        X02(wf, 0)  # |20>
        CSUM(wf, 0, 1)
        # |2,0> -> |2, (0+2)%3> = |2,2>
        expected = np.zeros(9, dtype=complex)
        expected[8] = 1.0  # index 2*3+2 = 8
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_csum_wraps(self):
        """CSUM wraps: |2,2> -> |2, (2+2)%3> = |2,1>."""
        wf = _qutrit(2)
        X02(wf, 0)  # |20>
        X02(wf, 1)  # |22>
        CSUM(wf, 0, 1)
        expected = np.zeros(9, dtype=complex)
        expected[7] = 1.0  # index 2*3+1 = 7
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)


# ============================================================
# QUDIT_SWAP
# ============================================================

class TestSwap:

    def test_swap_distinct_values(self):
        """SWAP |1,2> -> |2,1>."""
        wf = _qutrit(2)
        X01(wf, 0)  # |10>
        X02(wf, 1)  # |12>
        QUDIT_SWAP(wf, 0, 1)
        expected = np.zeros(9, dtype=complex)
        expected[2 * 3 + 1] = 1.0  # |21> -> index 7
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)

    def test_swap_same_values(self):
        """SWAP |1,1> -> |1,1> (no change)."""
        wf = _qutrit(2)
        X01(wf, 0)
        X01(wf, 1)  # |11>
        QUDIT_SWAP(wf, 0, 1)
        expected = np.zeros(9, dtype=complex)
        expected[4] = 1.0  # |11> -> index 4
        np.testing.assert_allclose(wf.amplitude, expected, atol=1e-12)


# ============================================================
# Measurement
# ============================================================

class TestMeasurement:

    def test_measure_qudit_probs_sum_to_one(self):
        wf = _qutrit(2)
        Hd(wf, 0)
        probs = measure_qudit(wf, 0)
        assert probs.shape == (3,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    def test_measure_qudit_deterministic(self):
        """Measuring |0> should give P(0)=1."""
        wf = _qutrit(1)
        probs = measure_qudit(wf, 0)
        np.testing.assert_allclose(probs, [1, 0, 0], atol=1e-12)

    def test_collapse_qudit_projects(self):
        """After collapse, only the measured outcome has nonzero probability."""
        wf = _qutrit(1)
        Hd(wf, 0)  # uniform superposition
        result = collapse_qudit(wf, 0)
        assert result in (0, 1, 2)
        probs_after = measure_qudit(wf, 0)
        np.testing.assert_allclose(probs_after[result], 1.0, atol=1e-12)

    def test_collapse_preserves_norm(self):
        wf = _qutrit(2)
        Hd(wf, 0)
        Hd(wf, 1)
        collapse_qudit(wf, 0)
        norm = np.sum(np.abs(wf.amplitude) ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


# ============================================================
# Expectation value
# ============================================================

class TestExpectation:

    def test_lambda3_on_zero(self):
        """<0|lambda_3|0> = 1 for the diagonal Gell-Mann matrix."""
        wf = _qutrit(1)
        val = qudit_expectation(wf, 0, GELL_MANN[2])  # lambda_3
        np.testing.assert_allclose(val, 1.0, atol=1e-12)

    def test_lambda3_on_one(self):
        """<1|lambda_3|1> = -1."""
        wf = _qutrit(1)
        X01(wf, 0)
        val = qudit_expectation(wf, 0, GELL_MANN[2])
        np.testing.assert_allclose(val, -1.0, atol=1e-12)

    def test_identity_expectation(self):
        """<psi|I|psi> = 1 for any normalized state."""
        wf = _qutrit(1)
        Hd(wf, 0)
        val = qudit_expectation(wf, 0, np.eye(3))
        np.testing.assert_allclose(val, 1.0, atol=1e-12)


# ============================================================
# _infer_dimension
# ============================================================

class TestInferDimension:

    @pytest.mark.parametrize("d", [2, 3, 4, 5])
    def test_infer_various_dimensions(self, d):
        n = 2
        wf = Qudit(n, dimension=d, backend=BACKEND)
        assert _infer_dimension(wf) == d

    def test_infer_single_qudit(self):
        for d in [2, 3, 4, 5, 6]:
            wf = Qudit(1, dimension=d, backend=BACKEND)
            assert _infer_dimension(wf) == d


# ============================================================
# _phase_givens_gate
# ============================================================

class TestPhaseGivens:

    def test_matches_expm(self):
        """_phase_givens_gate should match expm of the imaginary Gell-Mann generator."""
        theta = 1.3
        # lambda_2 is the imaginary off-diagonal (0,1) Gell-Mann matrix
        lam2 = GELL_MANN[1]  # [[0, -i, 0], [i, 0, 0], [0, 0, 0]]
        expected = expm(-1j * theta / 2.0 * lam2)
        actual = _phase_givens_gate(3, 0, 1, theta)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_matches_expm_02(self):
        theta = 0.8
        # lambda_5 is the imaginary off-diagonal (0,2) Gell-Mann matrix
        lam5 = GELL_MANN[4]  # [[0, 0, -i], [0, 0, 0], [i, 0, 0]]
        expected = expm(-1j * theta / 2.0 * lam5)
        actual = _phase_givens_gate(3, 0, 2, theta)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_matches_expm_12(self):
        theta = 2.1
        # lambda_7 is the imaginary off-diagonal (1,2) Gell-Mann matrix
        lam7 = GELL_MANN[6]  # [[0, 0, 0], [0, 0, -i], [0, i, 0]]
        expected = expm(-1j * theta / 2.0 * lam7)
        actual = _phase_givens_gate(3, 1, 2, theta)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
