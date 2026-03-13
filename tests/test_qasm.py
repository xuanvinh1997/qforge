# -*- coding: utf-8 -*-
"""Tests for OpenQASM import/export."""
import math
import numpy as np
import pytest
from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp
from qforge.qasm.exporter import circuit_to_qasm2, circuit_to_qasm3
from qforge.qasm.importer import qasm2_to_circuit, qasm3_to_circuit


class TestQASM2Export:
    def test_basic_circuit(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        qasm = circuit_to_qasm2(qc)
        assert 'OPENQASM 2.0' in qasm
        assert 'h q[0]' in qasm
        assert 'cx q[0], q[1]' in qasm

    def test_parameterised_gates(self):
        qc = Circuit(1)
        qc.rx(0, 0.5)
        qasm = circuit_to_qasm2(qc)
        assert 'rx(' in qasm

    def test_measurement(self):
        qc = Circuit(1)
        qc.h(0)
        qc.measure(0, 0)
        qasm = circuit_to_qasm2(qc)
        assert 'measure q[0] -> c[0]' in qasm

    def test_qreg_creg(self):
        qc = Circuit(3)
        qasm = circuit_to_qasm2(qc)
        assert 'qreg q[3]' in qasm
        assert 'creg c[3]' in qasm


class TestQASM3Export:
    def test_basic_circuit(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        qasm = circuit_to_qasm3(qc)
        assert 'OPENQASM 3.0' in qasm
        assert 'qubit[2] q' in qasm
        assert 'h q[0]' in qasm

    def test_measurement(self):
        qc = Circuit(1)
        qc.measure(0, 0)
        qasm = circuit_to_qasm3(qc)
        assert 'c[0] = measure q[0]' in qasm


class TestQASM2Import:
    def test_basic_import(self):
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
"""
        qc = qasm2_to_circuit(qasm)
        assert qc.n_qubits == 2
        assert len(qc) == 2
        assert qc[0].name == 'H'
        assert qc[1].name == 'CNOT'

    def test_parameterised_import(self):
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rx(0.5) q[0];
ry(1.2) q[0];
"""
        qc = qasm2_to_circuit(qasm)
        assert len(qc) == 2
        assert qc[0].name == 'RX'
        assert abs(qc[0].params[0] - 0.5) < 1e-10
        assert abs(qc[1].params[0] - 1.2) < 1e-10

    def test_measurement_import(self):
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
"""
        qc = qasm2_to_circuit(qasm)
        assert len(qc) == 1
        assert isinstance(qc[0], MeasureOp)

    def test_no_qreg_raises(self):
        with pytest.raises(ValueError, match="No qreg"):
            qasm2_to_circuit("OPENQASM 2.0;\n")


class TestQASM3Import:
    def test_basic_import(self):
        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
"""
        qc = qasm3_to_circuit(qasm)
        assert qc.n_qubits == 2
        assert len(qc) == 2

    def test_measurement_import(self):
        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""
        qc = qasm3_to_circuit(qasm)
        assert len(qc) == 1
        assert isinstance(qc[0], MeasureOp)

    def test_conditional_import(self):
        qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
c[0] = measure q[0];
if (c[0] == 1) {
  x q[1];
}
"""
        qc = qasm3_to_circuit(qasm)
        assert any(isinstance(op, ConditionalOp) for op in qc.ops)


class TestRoundTrip:
    def test_qasm2_roundtrip(self):
        """Export -> import -> same ops."""
        qc = Circuit(2)
        qc.h(0).cnot(0, 1).rx(0, 0.5)
        qasm = circuit_to_qasm2(qc)
        qc2 = qasm2_to_circuit(qasm)
        assert qc2.n_qubits == 2
        assert len(qc2) == len(qc)
        assert qc2[0].name == 'H'
        assert qc2[1].name == 'CNOT'
        assert qc2[2].name == 'RX'

    def test_qasm3_roundtrip(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        qasm = circuit_to_qasm3(qc)
        qc2 = qasm3_to_circuit(qasm)
        assert len(qc2) == len(qc)

    def test_roundtrip_produces_same_state(self):
        """Exported+imported circuit should produce the same state."""
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        wf1 = qc.run(backend='python')

        qasm = circuit_to_qasm2(qc)
        qc2 = qasm2_to_circuit(qasm)
        wf2 = qc2.run(backend='python')

        np.testing.assert_allclose(np.abs(wf1.amplitude - wf2.amplitude), 0, atol=1e-10)
