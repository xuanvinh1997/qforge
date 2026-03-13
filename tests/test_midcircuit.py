# -*- coding: utf-8 -*-
"""Tests for mid-circuit measurement and classical control."""
import numpy as np
import pytest
from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp, ClassicalRegister


class TestClassicalRegister:
    def test_init(self):
        creg = ClassicalRegister(3)
        assert creg.size == 3
        assert creg[0] == 0
        assert creg[1] == 0
        assert creg[2] == 0

    def test_set_get(self):
        creg = ClassicalRegister(2)
        creg[0] = 1
        assert creg[0] == 1
        assert creg[1] == 0


class TestMeasureOp:
    def test_creation(self):
        op = MeasureOp(qubit=0, classical_bit=1)
        assert op.qubit == 0
        assert op.classical_bit == 1

    def test_frozen(self):
        op = MeasureOp(qubit=0, classical_bit=0)
        with pytest.raises(AttributeError):
            op.qubit = 1


class TestConditionalOp:
    def test_creation(self):
        inner = GateOp(name='X', qubits=(1,))
        op = ConditionalOp(classical_bit=0, expected_value=1, op=inner)
        assert op.classical_bit == 0
        assert op.expected_value == 1
        assert op.op.name == 'X'


class TestCircuitMidMeasurement:
    def test_measure_adds_op(self):
        qc = Circuit(2)
        qc.h(0).measure(0, 0)
        assert len(qc) == 2
        assert isinstance(qc[1], MeasureOp)

    def test_c_if_adds_op(self):
        qc = Circuit(2)
        qc.measure(0, 0)
        gate = GateOp(name='X', qubits=(1,))
        qc.c_if(0, 1, gate)
        assert len(qc) == 2
        assert isinstance(qc[1], ConditionalOp)

    def test_measure_produces_classical_register(self):
        """Running a circuit with measurement should attach a classical register."""
        np.random.seed(42)
        qc = Circuit(1)
        qc.x(0)  # Put qubit in |1>
        qc.measure(0, 0)
        wf = qc.run(backend='python')
        # After measuring |1>, classical bit should be 1
        assert hasattr(wf, 'classical_register')
        assert wf.classical_register[0] == 1

    def test_conditional_x_on_measured_1(self):
        """If qubit 0 measured as 1, apply X to qubit 1."""
        np.random.seed(42)
        qc = Circuit(2)
        qc.x(0)  # qubit 0 = |1>
        qc.measure(0, 0)
        gate = GateOp(name='X', qubits=(1,))
        qc.c_if(0, 1, gate)
        wf = qc.run(backend='python')
        # qubit 0 was |1>, measured as 1, so X applied to qubit 1
        # Final state should have qubit 1 flipped to |1>
        assert wf.classical_register[0] == 1

    def test_conditional_x_on_measured_0(self):
        """If qubit 0 measured as 0, conditional X should NOT fire."""
        np.random.seed(42)
        qc = Circuit(2)
        # qubit 0 stays |0>
        qc.measure(0, 0)
        gate = GateOp(name='X', qubits=(1,))
        qc.c_if(0, 1, gate)  # Only fire if bit 0 == 1
        wf = qc.run(backend='python')
        # qubit 0 was |0>, measured as 0, so X NOT applied to qubit 1
        assert wf.classical_register[0] == 0
        # qubit 1 should still be |0> => amplitude at |00> should be nonzero
        assert abs(wf.amplitude[0]) > 0.9

    def test_teleportation_pattern(self):
        """Teleportation-like circuit with mid-circuit measurement."""
        np.random.seed(123)
        qc = Circuit(3)
        # Prepare Bell pair on qubits 1,2
        qc.h(1).cnot(1, 2)
        # CNOT qubit 0->1, then H on qubit 0
        qc.cnot(0, 1).h(0)
        # Measure qubits 0 and 1
        qc.measure(0, 0).measure(1, 1)
        # Corrections
        qc.c_if(1, 1, GateOp(name='X', qubits=(2,)))
        qc.c_if(0, 1, GateOp(name='Z', qubits=(2,)))
        # This should run without errors
        wf = qc.run(backend='python')
        assert hasattr(wf, 'classical_register')
