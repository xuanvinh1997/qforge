# -*- coding: utf-8 -*-
"""Tests for qforge.ir — Circuit IR and GateOp."""
import numpy as np
import pytest
from qforge.ir import Circuit, GateOp, record, _adjoint_op
from qforge.circuit import Qubit
from qforge import gates


# ============================================================
# Helpers
# ============================================================

def _amplitudes_close(a, b, atol=1e-10):
    """Assert two amplitude arrays are element-wise close."""
    np.testing.assert_allclose(np.abs(a - b), 0, atol=atol)


# ============================================================
# GateOp tests
# ============================================================

class TestGateOp:
    def test_creation(self):
        op = GateOp(name='H', qubits=(0,))
        assert op.name == 'H'
        assert op.qubits == (0,)
        assert op.params == ()
        assert op.matrix is None

    def test_parameterised(self):
        op = GateOp(name='RX', qubits=(1,), params=(0.5,))
        assert op.params == (0.5,)

    def test_equality(self):
        a = GateOp(name='H', qubits=(0,))
        b = GateOp(name='H', qubits=(0,))
        assert a == b

    def test_frozen(self):
        op = GateOp(name='H', qubits=(0,))
        with pytest.raises(AttributeError):
            op.name = 'X'


# ============================================================
# Circuit builder tests
# ============================================================

class TestCircuitBuilder:
    def test_basic_build(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        assert len(qc) == 2
        assert qc[0].name == 'H'
        assert qc[1].name == 'CNOT'

    def test_repr(self):
        qc = Circuit(3)
        qc.h(0).x(1).cnot(0, 2)
        assert 'n_qubits=3' in repr(qc)
        assert 'depth=3' in repr(qc)

    def test_all_single_qubit_gates(self):
        qc = Circuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0).xsquare(0)
        qc.rx(0, 0.1).ry(0, 0.2).rz(0, 0.3).phase(0, 0.4)
        assert len(qc) == 11

    def test_all_controlled_gates(self):
        qc = Circuit(2)
        qc.cnot(0, 1).cx(0, 1)
        qc.crx(0, 1, 0.1).cry(0, 1, 0.2).crz(0, 1, 0.3)
        qc.cphase(0, 1, 0.4).cp(0, 1, 0.5)
        assert len(qc) == 7

    def test_multi_qubit_gates(self):
        qc = Circuit(3)
        qc.ccnot(0, 1, 2).toffoli(0, 1, 2).or_gate(0, 1, 2)
        qc.swap(0, 1).iswap(0, 1).siswap(0, 1).cswap(0, 1, 2)
        assert len(qc) == 7

    def test_invalid_n_qubits(self):
        with pytest.raises(ValueError):
            Circuit(0)

    def test_iteration(self):
        qc = Circuit(2)
        qc.h(0).x(1)
        names = [op.name for op in qc]
        assert names == ['H', 'X']


# ============================================================
# Circuit execution tests — compare with direct gate calls
# ============================================================

class TestCircuitExecution:
    def test_single_h(self):
        """Circuit.run() vs direct H gate."""
        qc = Circuit(2)
        qc.h(0)
        wf_circ = qc.run(backend='python')

        wf_direct = Qubit(2, backend='python')
        gates.H(wf_direct, 0)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)

    def test_bell_state(self):
        """Bell state via Circuit matches direct construction."""
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        wf_circ = qc.run(backend='python')

        wf_direct = Qubit(2, backend='python')
        gates.H(wf_direct, 0)
        gates.CNOT(wf_direct, 0, 1)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)

    def test_parameterised_circuit(self):
        """Rotation gates with parameters."""
        qc = Circuit(2)
        qc.rx(0, 0.7).ry(1, 1.2).cnot(0, 1).rz(0, 0.3)
        wf_circ = qc.run(backend='python')

        wf_direct = Qubit(2, backend='python')
        gates.RX(wf_direct, 0, 0.7)
        gates.RY(wf_direct, 1, 1.2)
        gates.CNOT(wf_direct, 0, 1)
        gates.RZ(wf_direct, 0, 0.3)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)

    def test_three_qubit_gates(self):
        """CCNOT, SWAP, CSWAP via circuit."""
        qc = Circuit(3)
        qc.x(0).x(1).ccnot(0, 1, 2).swap(0, 2)
        wf_circ = qc.run(backend='python')

        wf_direct = Qubit(3, backend='python')
        gates.X(wf_direct, 0)
        gates.X(wf_direct, 1)
        gates.CCNOT(wf_direct, 0, 1, 2)
        gates.SWAP(wf_direct, 0, 2)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)

    def test_controlled_rotations(self):
        """CRX, CRY, CRZ, CPhase via circuit."""
        qc = Circuit(2)
        qc.h(0).crx(0, 1, 0.5).cry(0, 1, 0.3).crz(0, 1, 0.7).cphase(0, 1, 1.0)
        wf_circ = qc.run(backend='python')

        wf_direct = Qubit(2, backend='python')
        gates.H(wf_direct, 0)
        gates.CRX(wf_direct, 0, 1, 0.5)
        gates.CRY(wf_direct, 0, 1, 0.3)
        gates.CRZ(wf_direct, 0, 1, 0.7)
        gates.CPhase(wf_direct, 0, 1, 1.0)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)


# ============================================================
# Circuit.run() with params
# ============================================================

class TestCircuitParams:
    def test_bind_parameters(self):
        qc = Circuit(2)
        qc.rx(0, 0.0).ry(1, 0.0).crz(0, 1, 0.0)
        assert qc.num_parameters == 3

        bound = qc.bind_parameters(np.array([0.5, 1.0, 1.5]))
        assert bound[0].params == (0.5,)
        assert bound[1].params == (1.0,)
        assert bound[2].params == (1.5,)

    def test_run_with_params(self):
        qc = Circuit(2)
        qc.rx(0, 0.0).ry(1, 0.0)

        params = np.array([0.7, 1.2])
        wf_circ = qc.run(backend='python', params=params)

        wf_direct = Qubit(2, backend='python')
        gates.RX(wf_direct, 0, 0.7)
        gates.RY(wf_direct, 1, 1.2)

        _amplitudes_close(wf_circ.amplitude, wf_direct.amplitude)

    def test_wrong_param_count_raises(self):
        qc = Circuit(1)
        qc.rx(0, 0.0).ry(0, 0.0)
        with pytest.raises(ValueError, match="Expected 2"):
            qc.bind_parameters(np.array([1.0]))

    def test_parameter_indices(self):
        qc = Circuit(2)
        qc.h(0).rx(0, 0.1).cnot(0, 1).ry(1, 0.2)
        indices = qc.parameter_indices
        assert indices == [(1, 0), (3, 0)]


# ============================================================
# Circuit.__call__() — VQA compatibility
# ============================================================

class TestCircuitCallable:
    def test_call_on_existing_wf(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)

        wf = Qubit(2, backend='python')
        qc(wf)

        wf_direct = Qubit(2, backend='python')
        gates.H(wf_direct, 0)
        gates.CNOT(wf_direct, 0, 1)

        _amplitudes_close(wf.amplitude, wf_direct.amplitude)

    def test_call_with_params(self):
        qc = Circuit(1)
        qc.rx(0, 0.0)

        wf = Qubit(1, backend='python')
        qc(wf, params=np.array([1.5]))

        wf_direct = Qubit(1, backend='python')
        gates.RX(wf_direct, 0, 1.5)

        _amplitudes_close(wf.amplitude, wf_direct.amplitude)


# ============================================================
# Circuit transformations
# ============================================================

class TestCircuitTransformations:
    def test_adjoint_identity(self):
        """U @ U† should give identity (return to |0>)."""
        qc = Circuit(2)
        qc.h(0).cnot(0, 1).rx(1, 0.7)

        # Run U then U†
        wf = Qubit(2, backend='python')
        qc(wf)
        qc.adjoint()(wf)

        # Should be back to |00>
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        _amplitudes_close(wf.amplitude, expected, atol=1e-9)

    def test_adjoint_rotation_negates(self):
        adj = Circuit(1)
        adj.rx(0, 0.5)
        adj_circ = adj.adjoint()
        assert adj_circ[0].params == (-0.5,)

    def test_compose(self):
        c1 = Circuit(2)
        c1.h(0)
        c2 = Circuit(2)
        c2.cnot(0, 1)
        composed = c1.compose(c2)
        assert len(composed) == 2
        assert composed[0].name == 'H'
        assert composed[1].name == 'CNOT'

    def test_compose_mismatch_raises(self):
        c1 = Circuit(2)
        c2 = Circuit(3)
        with pytest.raises(ValueError, match="different qubit counts"):
            c1.compose(c2)

    def test_copy(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        cp = qc.copy()
        assert len(cp) == 2
        cp.x(0)
        assert len(cp) == 3
        assert len(qc) == 2  # original unchanged


# ============================================================
# Recording context manager
# ============================================================

class TestRecording:
    def test_record_basic(self):
        with record(2) as qc:
            wf = Qubit(2, backend='python')
            gates.H(wf, 0)
            gates.CNOT(wf, 0, 1)

        assert len(qc) == 2
        assert qc[0].name == 'H'
        assert qc[0].qubits == (0,)
        assert qc[1].name == 'CNOT'
        assert qc[1].qubits == (0, 1)

    def test_record_with_params(self):
        with record(1) as qc:
            wf = Qubit(1, backend='python')
            gates.RX(wf, 0, 0.5)
            gates.RY(wf, 0, 1.2)

        assert qc[0].params == (0.5,)
        assert qc[1].params == (1.2,)

    def test_no_recording_outside_context(self):
        """Gates called outside record() should not be recorded."""
        from qforge.ir import _RECORDING_CIRCUIT
        assert _RECORDING_CIRCUIT is None
        wf = Qubit(1, backend='python')
        gates.H(wf, 0)
        assert _RECORDING_CIRCUIT is None

    def test_recorded_circuit_replays_correctly(self):
        """Circuit recorded from gates produces same state when replayed."""
        with record(2) as qc:
            wf1 = Qubit(2, backend='python')
            gates.H(wf1, 0)
            gates.RY(wf1, 1, 0.8)
            gates.CNOT(wf1, 0, 1)

        wf2 = qc.run(backend='python')
        _amplitudes_close(wf1.amplitude, wf2.amplitude)
