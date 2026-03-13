# -*- coding: utf-8 -*-
"""Tests for circuit serialization."""
import numpy as np
import pytest
from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp
from qforge.serialization import (
    circuit_to_json, circuit_from_json,
    circuit_to_binary, circuit_from_binary,
)


class TestJSONSerialization:
    def test_basic_roundtrip(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert qc2.n_qubits == 2
        assert len(qc2) == 2
        assert qc2[0].name == 'H'
        assert qc2[1].name == 'CNOT'

    def test_params_roundtrip(self):
        qc = Circuit(1)
        qc.rx(0, 0.5).ry(0, 1.2)
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert abs(qc2[0].params[0] - 0.5) < 1e-10
        assert abs(qc2[1].params[0] - 1.2) < 1e-10

    def test_measurement_roundtrip(self):
        qc = Circuit(2)
        qc.h(0).measure(0, 0)
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert isinstance(qc2[1], MeasureOp)
        assert qc2[1].qubit == 0

    def test_conditional_roundtrip(self):
        qc = Circuit(2)
        qc.measure(0, 0)
        qc.c_if(0, 1, GateOp(name='X', qubits=(1,)))
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert isinstance(qc2[1], ConditionalOp)
        assert qc2[1].op.name == 'X'

    def test_matrix_roundtrip(self):
        qc = Circuit(1)
        mat = np.array([[0, 1], [1, 0]], dtype=complex)
        qc.unitary(mat, [0])
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert qc2[0].matrix is not None
        np.testing.assert_allclose(qc2[0].matrix, mat, atol=1e-10)

    def test_produces_same_state(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1).rx(1, 0.7)
        wf1 = qc.run(backend='python')

        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        wf2 = qc2.run(backend='python')

        np.testing.assert_allclose(np.abs(wf1.amplitude - wf2.amplitude), 0, atol=1e-10)


class TestBinarySerialization:
    def test_basic_roundtrip(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        data = circuit_to_binary(qc)
        qc2 = circuit_from_binary(data)
        assert qc2.n_qubits == 2
        assert len(qc2) == 2
        assert qc2[0].name == 'H'
        assert qc2[1].name == 'CNOT'

    def test_params_roundtrip(self):
        qc = Circuit(1)
        qc.rx(0, 0.5).rz(0, 1.7)
        data = circuit_to_binary(qc)
        qc2 = circuit_from_binary(data)
        assert abs(qc2[0].params[0] - 0.5) < 1e-10
        assert abs(qc2[1].params[0] - 1.7) < 1e-10

    def test_measurement_roundtrip(self):
        qc = Circuit(2)
        qc.h(0).measure(0, 0)
        data = circuit_to_binary(qc)
        qc2 = circuit_from_binary(data)
        assert isinstance(qc2[1], MeasureOp)

    def test_conditional_roundtrip(self):
        qc = Circuit(2)
        qc.measure(0, 0)
        qc.c_if(0, 1, GateOp(name='X', qubits=(1,)))
        data = circuit_to_binary(qc)
        qc2 = circuit_from_binary(data)
        assert isinstance(qc2[1], ConditionalOp)

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="bad magic"):
            circuit_from_binary(b'XXXX' + b'\x00' * 20)

    def test_binary_smaller_than_json(self):
        qc = Circuit(3)
        qc.h(0).cnot(0, 1).rx(2, 0.5).crz(0, 1, 1.2)
        json_data = circuit_to_json(qc).encode('utf-8')
        bin_data = circuit_to_binary(qc)
        assert len(bin_data) < len(json_data)

    def test_produces_same_state(self):
        qc = Circuit(2)
        qc.h(0).cnot(0, 1).ry(1, 0.3)
        wf1 = qc.run(backend='python')

        data = circuit_to_binary(qc)
        qc2 = circuit_from_binary(data)
        wf2 = qc2.run(backend='python')

        np.testing.assert_allclose(np.abs(wf1.amplitude - wf2.amplitude), 0, atol=1e-10)
