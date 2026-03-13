# -*- coding: utf-8 -*-
"""Tests for Phase 1 features: Parameters, QubitUnitary, CustomGate, Multi-controlled gates."""
import numpy as np
import pytest
from qforge.circuit import Qubit
from qforge import gates
from qforge.ir import Circuit, CustomGate, register_gate, unregister_gate
from qforge.parameters import Parameter, ParameterVector


# ============================================================
# 1A: Symbolic Parameters
# ============================================================

class TestParameter:
    def test_creation(self):
        p = Parameter('theta')
        assert p.name == 'theta'
        assert p.value is None
        assert not p.is_bound()

    def test_bound(self):
        p = Parameter('theta', 0.5)
        assert p.is_bound()
        assert float(p) == 0.5

    def test_unbound_float_raises(self):
        p = Parameter('theta')
        with pytest.raises(ValueError, match="unbound"):
            float(p)

    def test_bind(self):
        p = Parameter('theta')
        pb = p.bind(1.5)
        assert pb.value == 1.5
        assert p.value is None  # original unchanged


class TestParameterVector:
    def test_creation(self):
        pv = ParameterVector('theta', 3)
        assert len(pv) == 3
        assert pv[0].name == 'theta_0'
        assert pv[2].name == 'theta_2'

    def test_bind(self):
        pv = ParameterVector('theta', 2)
        bound = pv.bind([0.5, 1.0])
        assert bound[0].value == 0.5
        assert bound[1].value == 1.0

    def test_bind_wrong_size(self):
        pv = ParameterVector('theta', 3)
        with pytest.raises(ValueError):
            pv.bind([1.0, 2.0])

    def test_iteration(self):
        pv = ParameterVector('x', 4)
        names = [p.name for p in pv]
        assert names == ['x_0', 'x_1', 'x_2', 'x_3']


# ============================================================
# 1C: QubitUnitary gate
# ============================================================

def _amplitudes_close(a, b, atol=1e-10):
    np.testing.assert_allclose(np.abs(a - b), 0, atol=atol)


class TestQubitUnitary:
    def test_1q_hadamard_matrix(self):
        """Applying the Hadamard matrix via QubitUnitary should match H gate."""
        H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        wf1 = Qubit(2, backend='python')
        gates.QubitUnitary(wf1, H_mat, [0])

        wf2 = Qubit(2, backend='python')
        gates.H(wf2, 0)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_1q_rx_matrix(self):
        """RX via unitary matrix."""
        phi = 0.7
        c, s = np.cos(phi/2), np.sin(phi/2)
        rx_mat = np.array([[c, -1j*s], [-1j*s, c]])

        wf1 = Qubit(1, backend='python')
        gates.QubitUnitary(wf1, rx_mat, [0])

        wf2 = Qubit(1, backend='python')
        gates.RX(wf2, 0, phi)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_2q_cnot_matrix(self):
        """CNOT via unitary matrix."""
        cnot_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])

        wf1 = Qubit(2, backend='python')
        gates.H(wf1, 0)
        gates.QubitUnitary(wf1, cnot_mat, [0, 1])

        wf2 = Qubit(2, backend='python')
        gates.H(wf2, 0)
        gates.CNOT(wf2, 0, 1)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_2q_swap_matrix(self):
        """SWAP via unitary matrix."""
        swap_mat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        wf1 = Qubit(3, backend='python')
        gates.X(wf1, 0)
        gates.QubitUnitary(wf1, swap_mat, [0, 2])

        wf2 = Qubit(3, backend='python')
        gates.X(wf2, 0)
        gates.SWAP(wf2, 0, 2)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_1q_identity(self):
        """Identity matrix should not change state."""
        wf = Qubit(2, backend='python')
        gates.H(wf, 0)
        amp_before = wf.amplitude.copy()
        gates.QubitUnitary(wf, np.eye(2), [0])
        _amplitudes_close(wf.amplitude, amp_before)

    def test_unsupported_3q_raises(self):
        wf = Qubit(3, backend='python')
        with pytest.raises(NotImplementedError):
            gates.QubitUnitary(wf, np.eye(8), [0, 1, 2])

    def test_circuit_unitary(self):
        """Test Circuit.unitary() builder method."""
        H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        qc = Circuit(2)
        qc.unitary(H_mat, [0]).cnot(0, 1)
        wf = qc.run(backend='python')

        wf_ref = Qubit(2, backend='python')
        gates.H(wf_ref, 0)
        gates.CNOT(wf_ref, 0, 1)

        _amplitudes_close(wf.amplitude, wf_ref.amplitude)


# ============================================================
# 1D: Custom gates
# ============================================================

class TestCustomGate:
    def setup_method(self):
        # Clean up registry before each test
        unregister_gate('TestGate')
        unregister_gate('BellPrep')

    def test_matrix_custom_gate(self):
        """Register and use a matrix-defined custom gate."""
        # Define sqrt(X) as custom gate
        sqrtx = np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2
        register_gate(CustomGate(name='TestGate', n_qubits=1, matrix=sqrtx))

        qc = Circuit(1)
        qc.add_gate(
            __import__('qforge.ir', fromlist=['GateOp']).GateOp(
                name='TestGate', qubits=(0,)))
        wf = qc.run(backend='python')

        wf_ref = Qubit(1, backend='python')
        gates.QubitUnitary(wf_ref, sqrtx, [0])

        _amplitudes_close(wf.amplitude, wf_ref.amplitude)

    def test_circuit_custom_gate(self):
        """Register and use a sub-circuit-defined custom gate."""
        sub = Circuit(2)
        sub.h(0).cnot(0, 1)
        register_gate(CustomGate(name='BellPrep', n_qubits=2, subcircuit=sub))

        qc = Circuit(2)
        qc.add_gate(
            __import__('qforge.ir', fromlist=['GateOp']).GateOp(
                name='BellPrep', qubits=(0, 1)))
        wf = qc.run(backend='python')

        wf_ref = sub.run(backend='python')
        _amplitudes_close(wf.amplitude, wf_ref.amplitude)


# ============================================================
# 1E: Multi-controlled gates
# ============================================================

class TestMultiControlled:
    def test_mcx_0_controls(self):
        """MCX with 0 controls = X."""
        wf1 = Qubit(1, backend='python')
        gates.mcx(wf1, [], 0)

        wf2 = Qubit(1, backend='python')
        gates.X(wf2, 0)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_mcx_1_control(self):
        """MCX with 1 control = CNOT."""
        wf1 = Qubit(2, backend='python')
        gates.H(wf1, 0)
        gates.mcx(wf1, [0], 1)

        wf2 = Qubit(2, backend='python')
        gates.H(wf2, 0)
        gates.CNOT(wf2, 0, 1)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_mcx_2_controls(self):
        """MCX with 2 controls = CCNOT."""
        wf1 = Qubit(3, backend='python')
        gates.X(wf1, 0)
        gates.X(wf1, 1)
        gates.mcx(wf1, [0, 1], 2)

        wf2 = Qubit(3, backend='python')
        gates.X(wf2, 0)
        gates.X(wf2, 1)
        gates.CCNOT(wf2, 0, 1, 2)

        _amplitudes_close(wf1.amplitude, wf2.amplitude)

    def test_mcx_3_controls(self):
        """MCX with 3 controls should flip target only when all 3 controls are |1>."""
        wf = Qubit(4, backend='python')
        # Set all controls to |1>
        gates.X(wf, 0)
        gates.X(wf, 1)
        gates.X(wf, 2)
        # Apply 3-controlled X on qubit 3
        gates.mcx(wf, [0, 1, 2], 3)

        # State should be |1111>
        probs = wf.probabilities()
        assert probs[0b1111] > 0.99

    def test_mcx_3_controls_not_all_one(self):
        """MCX with 3 controls should NOT flip when not all controls are |1>."""
        wf = Qubit(4, backend='python')
        gates.X(wf, 0)
        gates.X(wf, 1)
        # qubit 2 is |0> — control condition not met
        gates.mcx(wf, [0, 1, 2], 3)

        # State should remain |1100>
        probs = wf.probabilities()
        assert probs[0b1100] > 0.99

    def test_mcz(self):
        """MCZ should apply phase -1 when all controls + target are |1>."""
        wf = Qubit(3, backend='python')
        gates.X(wf, 0)
        gates.X(wf, 1)
        gates.X(wf, 2)
        amp_before = wf.amplitude.copy()
        gates.mcz(wf, [0, 1], 2)
        # The |111> amplitude should be negated
        assert abs(wf.amplitude[0b111] - (-amp_before[0b111])) < 1e-10

    def test_mcp(self):
        """MCP should apply e^(i*phi) when all controls + target are |1>."""
        import cmath
        wf = Qubit(2, backend='python')
        gates.X(wf, 0)
        gates.X(wf, 1)
        phi = 0.7
        gates.mcp(wf, [0], 1, phi)
        # State |11> should have phase e^(i*phi)
        expected_phase = cmath.exp(1j * phi)
        assert abs(wf.amplitude[0b11] - expected_phase) < 1e-10

    def test_mcx_circuit_builder(self):
        """Test Circuit.mcx() builder method."""
        qc = Circuit(4)
        qc.x(0).x(1).x(2).mcx([0, 1, 2], 3)
        wf = qc.run(backend='python')
        probs = wf.probabilities()
        assert probs[0b1111] > 0.99
