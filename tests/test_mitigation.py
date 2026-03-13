# -*- coding: utf-8 -*-
"""Tests for error mitigation modules."""
import numpy as np
import pytest
from qforge.ir import Circuit, GateOp
from qforge.mitigation.zne import fold_circuit, zero_noise_extrapolation, _extrapolate_linear
from qforge.mitigation.pec import PEC, DepolarizingNoiseModel
from qforge.mitigation.readout import correct_readout


class TestFoldCircuit:
    def test_fold_factor_1_unchanged(self):
        qc = Circuit(1)
        qc.h(0).rx(0, 0.5)
        folded = fold_circuit(qc, 1)
        assert len(folded) == len(qc)

    def test_fold_factor_2(self):
        qc = Circuit(1)
        qc.h(0)
        folded = fold_circuit(qc, 2)
        # H + H_adj H = 3 ops
        assert len(folded) == 3

    def test_fold_factor_3(self):
        qc = Circuit(1)
        qc.h(0)
        folded = fold_circuit(qc, 3)
        # H + H_adj H + H_adj H = 5 ops
        assert len(folded) == 5

    def test_fold_invalid(self):
        qc = Circuit(1)
        with pytest.raises(ValueError):
            fold_circuit(qc, 0)

    def test_folded_circuit_same_result(self):
        """Folded circuit should produce the same final state (ideal)."""
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        wf1 = qc.run(backend='python')
        folded = fold_circuit(qc, 2)
        wf2 = folded.run(backend='python')
        np.testing.assert_allclose(np.abs(wf1.amplitude - wf2.amplitude), 0, atol=1e-9)


class TestZNE:
    def test_linear_extrapolation(self):
        """Linear extrapolation to zero noise."""
        sf = np.array([1.0, 2.0, 3.0])
        # Simulate noisy expectations: E(lambda) = 0.5 + 0.1*lambda
        exps = 0.5 + 0.1 * sf
        mitigated = _extrapolate_linear(sf, exps)
        np.testing.assert_allclose(mitigated, 0.5, atol=0.01)

    def test_zne_ideal_circuit(self):
        """ZNE on a noiseless circuit should return approximately the ideal value."""
        qc = Circuit(1)
        qc.rx(0, 0.7)

        from qforge.algo.hamiltonian import Hamiltonian
        H = Hamiltonian([1.0], [[('Z', 0)]])

        def executor(circuit):
            wf = circuit.run(backend='python')
            return H.expectation(wf)

        result = zero_noise_extrapolation(
            circuit_fn=lambda: qc,
            executor=executor,
            scale_factors=[1, 2, 3],
            extrapolator='linear',
        )
        ideal = np.cos(0.7)
        np.testing.assert_allclose(result, ideal, atol=0.1)


class TestPEC:
    def test_depolarizing_model(self):
        model = DepolarizingNoiseModel(p=0.01)
        op = GateOp(name='H', qubits=(0,))
        decomp = model.quasi_probabilities(op)
        assert len(decomp) == 4
        # Weights should sum to > 1 (one-norm)
        total = sum(abs(w) for w, _ in decomp)
        assert total > 1.0

    def test_zero_noise_model(self):
        model = DepolarizingNoiseModel(p=0.0)
        op = GateOp(name='H', qubits=(0,))
        decomp = model.quasi_probabilities(op)
        assert len(decomp) == 1
        assert decomp[0][0] == 1.0

    def test_one_norm(self):
        model = DepolarizingNoiseModel(p=0.01)
        assert model.one_norm > 1.0

    def test_pec_runs(self):
        """PEC should run without errors."""
        qc = Circuit(1)
        qc.h(0)

        def executor(circuit):
            wf = circuit.run(backend='python')
            return float(np.abs(wf.amplitude[0]) ** 2 - np.abs(wf.amplitude[1]) ** 2)

        pec = PEC(DepolarizingNoiseModel(p=0.01))
        result = pec.mitigate(qc, executor, n_samples=50)
        # Result should be close to 0 for H|0>
        assert abs(result) < 1.0  # Just check it doesn't crash


class TestReadoutCorrection:
    def test_perfect_calibration(self):
        """Perfect calibration matrix = identity should leave counts unchanged."""
        cal = np.eye(4)
        counts = {'00': 500, '01': 300, '10': 150, '11': 50}
        corrected = correct_readout(counts, cal)
        for bs, c in counts.items():
            assert abs(corrected.get(bs, 0) - c) < 1

    def test_noisy_calibration(self):
        """Noisy calibration should shift counts toward corrected values."""
        # Simple 1-qubit case: 10% flip rate
        cal = np.array([[0.9, 0.1], [0.1, 0.9]])
        counts = {'0': 450, '1': 550}  # Noisy measurement of |1>
        corrected = correct_readout(counts, cal)
        # Corrected should show more |1>
        assert corrected.get('1', 0) > corrected.get('0', 0)
