# -*- coding: utf-8 -*-
"""Tests for standard quantum algorithms: QFT, QPE, Grover, HHL."""
import numpy as np
import pytest
from qforge.circuit import Qubit
from qforge import gates
from qforge.algo.standard.qft import qft, inverse_qft
from qforge.algo.standard.qpe import qpe
from qforge.algo.standard.grover import grover_search, grover_oracle, grover_diffusion
from qforge.algo.standard.hhl import hhl


def _amplitudes_close(a, b, atol=1e-10):
    np.testing.assert_allclose(np.abs(a - b), 0, atol=atol)


# ============================================================
# QFT tests
# ============================================================

class TestQFT:
    def test_qft_2qubit(self):
        """QFT on |00⟩ should produce uniform superposition."""
        wf = Qubit(2, backend='python')
        qft(wf)
        probs = wf.probabilities()
        # QFT|0> = uniform superposition = 1/sqrt(N) for all
        np.testing.assert_allclose(probs, [0.25] * 4, atol=1e-10)

    def test_qft_matches_numpy_fft(self):
        """QFT should match the DFT matrix (up to normalization)."""
        n = 3
        N = 2**n
        # Prepare a known state: |010⟩ = basis state 2
        wf = Qubit(n, backend='python')
        gates.X(wf, 1)  # |010⟩

        # Apply QFT
        qft(wf)
        amp_qft = wf.amplitude.copy()

        # Compare with numpy DFT
        input_vec = np.zeros(N, dtype=complex)
        input_vec[2] = 1.0
        amp_dft = np.fft.ifft(input_vec) * np.sqrt(N)

        # QFT and DFT should match up to global phase and qubit ordering
        # Check that probabilities match
        np.testing.assert_allclose(
            np.abs(amp_qft)**2, np.abs(amp_dft)**2, atol=1e-10
        )

    def test_inverse_qft_reverses(self):
        """QFT followed by inverse QFT should return to original state."""
        n = 3
        wf = Qubit(n, backend='python')
        gates.X(wf, 0)
        gates.H(wf, 1)  # Create a non-trivial state
        amp_before = wf.amplitude.copy()

        qft(wf)
        inverse_qft(wf)

        _amplitudes_close(wf.amplitude, amp_before)

    def test_qft_specific_qubits(self):
        """QFT on a subset of qubits."""
        wf = Qubit(3, backend='python')
        qft(wf, qubits=[0, 1])  # Only first 2 qubits
        # First 2 qubits should be in uniform superposition
        probs = wf.probabilities()
        # States 000, 010, 100, 110 should each have 0.25, others 0
        assert abs(probs[0b000] - 0.25) < 1e-10
        assert abs(probs[0b010] - 0.25) < 1e-10
        assert abs(probs[0b100] - 0.25) < 1e-10
        assert abs(probs[0b110] - 0.25) < 1e-10

    def test_qft_1qubit(self):
        """QFT on 1 qubit is just Hadamard."""
        wf1 = Qubit(1, backend='python')
        qft(wf1)

        wf2 = Qubit(1, backend='python')
        gates.H(wf2, 0)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)


# ============================================================
# QPE tests
# ============================================================

class TestQPE:
    def test_qpe_z_gate(self):
        """QPE should estimate the phase of Z|1⟩ = e^{iπ}|1⟩ → φ = 0.5."""
        def apply_z_power(wf, targets, power):
            for _ in range(power):
                gates.Z(wf, targets[0])

        def prep_eigenstate(wf, targets):
            gates.X(wf, targets[0])  # Prepare |1⟩

        result = qpe(apply_z_power, n_ancilla=4,
                     target_state_fn=prep_eigenstate, n_samples=1000)
        assert abs(result['phase'] - 0.5) < 0.1  # Allow some imprecision

    def test_qpe_s_gate(self):
        """QPE for S gate: S|1⟩ = e^{iπ/2}|1⟩ → φ = 0.25."""
        def apply_s_power(wf, targets, power):
            for _ in range(power):
                gates.S(wf, targets[0])

        def prep_eigenstate(wf, targets):
            gates.X(wf, targets[0])

        result = qpe(apply_s_power, n_ancilla=4,
                     target_state_fn=prep_eigenstate, n_samples=1000)
        assert abs(result['phase'] - 0.25) < 0.1

    def test_qpe_identity(self):
        """QPE for identity: I|ψ⟩ = e^{0}|ψ⟩ → φ = 0."""
        def apply_id_power(wf, targets, power):
            pass  # Identity

        result = qpe(apply_id_power, n_ancilla=3, n_samples=1000)
        assert abs(result['phase']) < 0.1


# ============================================================
# Grover tests
# ============================================================

class TestGrover:
    def test_grover_single_marked(self):
        """Grover should find a single marked state with high probability."""
        marked = 5  # |101⟩

        def oracle(wf, n):
            grover_oracle(wf, [marked], n)

        result = grover_search(oracle, n_qubits=3, n_samples=1000)
        assert result['result_int'] == marked
        assert result['probability'] > 0.7

    def test_grover_2qubit(self):
        """Grover on 2 qubits with 1 marked state — should find it in 1 iteration."""
        marked = 3  # |11⟩

        def oracle(wf, n):
            grover_oracle(wf, [marked], n)

        result = grover_search(oracle, n_qubits=2, n_iterations=1, n_samples=1000)
        assert result['result_int'] == marked
        assert result['probability'] > 0.9

    def test_grover_oracle_flips_marked(self):
        """Oracle should negate amplitudes of marked states only."""
        wf = Qubit(2, backend='python')
        gates.H(wf, 0)
        gates.H(wf, 1)
        amp_before = wf.amplitude.copy()

        grover_oracle(wf, [2], 2)  # Mark |10⟩

        # Only state 2 (|10⟩) should be negated
        for i in range(4):
            if i == 2:
                assert abs(wf.amplitude[i] - (-amp_before[i])) < 1e-10
            else:
                assert abs(wf.amplitude[i] - amp_before[i]) < 1e-10

    def test_grover_diffusion_amplifies(self):
        """Diffusion operator should amplify the marked state."""
        wf = Qubit(3, backend='python')
        # Start with uniform superposition
        for q in range(3):
            gates.H(wf, q)

        # Oracle marks |101⟩ = 5
        grover_oracle(wf, [5], 3)

        # Diffusion should amplify state 5
        grover_diffusion(wf, list(range(3)))

        probs = wf.probabilities()
        assert probs[5] > 1.0/8  # Should be amplified above uniform


# ============================================================
# HHL tests
# ============================================================

class TestHHL:
    def test_hhl_2x2_identity(self):
        """HHL with A=I, b=[1,0] should give x=[1,0]."""
        A = np.eye(2)
        b = np.array([1.0, 0.0])
        result = hhl(A, b, n_ancilla=3)
        # Solution should be proportional to b
        np.testing.assert_allclose(np.abs(result['solution']), np.abs(b), atol=0.1)

    def test_hhl_simple_system(self):
        """HHL for A=diag(1,2), b=[1,1]/sqrt(2) → x=[1, 0.5]/norm."""
        A = np.diag([1.0, 2.0])
        b = np.array([1.0, 1.0]) / np.sqrt(2)
        result = hhl(A, b, n_ancilla=4)

        # Classical solution
        x_classical = np.linalg.solve(A, b)
        x_classical /= np.linalg.norm(x_classical)

        # Check direction matches (up to global phase)
        overlap = abs(np.dot(result['solution'].conj(), x_classical))
        assert overlap > 0.9

    def test_hhl_hermitian(self):
        """HHL for a Hermitian 2x2 matrix."""
        A = np.array([[2, -1], [-1, 2]], dtype=complex)
        b = np.array([1.0, 0.0])
        result = hhl(A, b, n_ancilla=4)

        x_classical = np.linalg.solve(A, b)
        x_classical /= np.linalg.norm(x_classical)

        overlap = abs(np.dot(result['solution'].conj(), x_classical))
        assert overlap > 0.9
