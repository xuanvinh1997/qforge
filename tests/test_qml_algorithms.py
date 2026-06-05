# -*- coding: utf-8 -*-
"""Tests for QML algorithms and paper-backed circuit invariants."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from qforge.algo import (
    DataReuploadingClassifier,
    QCNN,
    QSVM,
    VQC,
    hardware_efficient_ansatz,
    strongly_entangling_ansatz,
)
from qforge.benchmarks import BenchConfig
from qforge.benchmarks.suites.qml import QMLBenchmarkSuite
from qforge.circuit import Qubit


def _single_qubit_matrix(gate: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if gate == "RY":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if gate == "RZ":
        return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    raise ValueError(gate)


def _apply_single_ref(state: np.ndarray, n_qubits: int, qubit: int, matrix: np.ndarray) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(matrix, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def _apply_cnot_ref(state: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    out = state.copy()
    for idx, amp in enumerate(state):
        if ((idx >> (n_qubits - control - 1)) & 1) == 1:
            flipped = idx ^ (1 << (n_qubits - target - 1))
            out[flipped] = amp
    return out


def _hardware_efficient_ref(n_qubits: int, params: np.ndarray, n_layers: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    idx = 0
    for layer in range(n_layers + 1):
        for q in range(n_qubits):
            state = _apply_single_ref(state, n_qubits, q, _single_qubit_matrix("RY", params[idx]))
            idx += 1
        if layer < n_layers:
            for q in range(n_qubits - 1):
                state = _apply_cnot_ref(state, n_qubits, q, q + 1)
    return state


def _strongly_entangling_ref(n_qubits: int, params: np.ndarray, n_layers: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            state = _apply_single_ref(state, n_qubits, q, _single_qubit_matrix("RZ", params[idx]))
            idx += 1
            state = _apply_single_ref(state, n_qubits, q, _single_qubit_matrix("RY", params[idx]))
            idx += 1
            state = _apply_single_ref(state, n_qubits, q, _single_qubit_matrix("RZ", params[idx]))
            idx += 1
        stride = (layer % (n_qubits - 1)) + 1
        for q in range(n_qubits):
            target = (q + stride) % n_qubits
            if target != q:
                state = _apply_cnot_ref(state, n_qubits, q, target)
    return state


def test_hardware_efficient_ansatz_matches_numpy_reference():
    """Hardware-efficient ansatz runs the real RY/CNOT ladder."""
    n_qubits, n_layers = 3, 2
    params = np.linspace(-0.7, 0.9, n_qubits * (n_layers + 1))

    wf = Qubit(n_qubits, backend="python")
    hardware_efficient_ansatz(wf, params, n_layers=n_layers)

    np.testing.assert_allclose(
        wf.amplitude,
        _hardware_efficient_ref(n_qubits, params, n_layers),
        atol=1e-12,
    )


def test_strongly_entangling_ansatz_matches_numpy_reference():
    """Strongly entangling layers use RZ/RY/RZ blocks plus stride CNOT rings."""
    n_qubits, n_layers = 3, 2
    params = np.linspace(-0.4, 0.8, n_layers * n_qubits * 3)

    wf = Qubit(n_qubits, backend="python")
    strongly_entangling_ansatz(wf, params, n_layers=n_layers)

    np.testing.assert_allclose(
        wf.amplitude,
        _strongly_entangling_ref(n_qubits, params, n_layers),
        atol=1e-12,
    )


def test_qsvm_havlicek_fidelity_kernel_is_symmetric_psd():
    """Havlicek-style fidelity kernels have unit diagonal and are PSD."""
    X = np.array([
        [0.1, -0.2],
        [0.4, 0.3],
        [-0.5, 0.7],
    ])
    qsvm = QSVM(n_qubits=2, n_layers=1, backend="python")

    K = qsvm.kernel_matrix(X)

    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), np.ones(len(X)), atol=1e-12)
    assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-10


def test_binary_qml_readouts_are_not_degenerate():
    """One-qubit binary readout must not duplicate the same class logit."""
    x1 = np.array([0.37, -0.25])
    x4 = np.array([0.37, -0.25, 0.0, 0.0])

    dr = DataReuploadingClassifier(n_qubits=1, n_layers=2, n_classes=2, backend="python")
    dr_params = np.linspace(-0.5, 0.4, dr.n_params)
    dr_logits = dr._circuit(x1, dr_params)
    assert dr_logits[0] == pytest.approx(-dr_logits[1])
    assert not np.allclose(dr._probabilities(x1, dr_params), [0.5, 0.5])

    qcnn = QCNN(n_qubits=4, n_classes=2, backend="python")
    qcnn_params = np.linspace(-0.35, 0.45, qcnn.n_params)
    qcnn_logits = qcnn._circuit(x4, qcnn_params)
    assert qcnn_logits[0] == pytest.approx(-qcnn_logits[1])
    assert not np.allclose(qcnn._probabilities(x4, qcnn_params), [0.5, 0.5])

    vqc = VQC(n_qubits=1, n_layers=1, n_classes=2, backend="python")
    vqc_params = np.array([0.2, -0.1])
    vqc_logits = vqc._circuit(x1, vqc_params)
    assert vqc_logits[0] == pytest.approx(-vqc_logits[1])


def test_qml_models_run_tiny_training_loops():
    X = np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [-0.9, 0.8, 0.0, 0.0],
        [0.8, -0.7, 0.0, 0.0],
        [0.9, 0.9, 0.0, 0.0],
    ])
    y = np.array([0, 1, 1, 0])

    for model in (
        VQC(n_qubits=4, n_layers=1, n_classes=2, backend="python"),
        DataReuploadingClassifier(n_qubits=1, n_layers=2, n_classes=2, backend="python"),
        QCNN(n_qubits=4, n_classes=2, backend="python"),
    ):
        params0 = np.linspace(-0.2, 0.2, model.n_params)
        params, history = model.fit(X, y, params=params0, steps=2)
        assert params.shape == (model.n_params,)
        assert len(history) == 2
        assert np.all(np.isfinite(history))
        assert model.predict_proba(X, params).shape == (len(X), 2)


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="PennyLane not installed")
def test_ansatz_and_qsvm_kernel_match_pennylane():
    """Cross-library check against PennyLane statevectors."""
    import pennylane as qml

    n_qubits, n_layers = 3, 2
    params = np.linspace(-0.7, 0.9, n_qubits * (n_layers + 1))

    wf = Qubit(n_qubits, backend="python")
    hardware_efficient_ansatz(wf, params, n_layers=n_layers)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def hea_state(p):
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(n_qubits):
                qml.RY(p[idx], wires=q)
                idx += 1
            if layer < n_layers:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.state()

    np.testing.assert_allclose(wf.amplitude, np.asarray(hea_state(params)), atol=1e-12)

    X = np.array([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.7]])
    qforge_kernel = QSVM(n_qubits=2, n_layers=1, backend="python").kernel_matrix(X)

    dev_kernel = qml.device("default.qubit", wires=2)

    @qml.qnode(dev_kernel)
    def feature_state(x):
        feats = np.zeros(2)
        feats[:len(x)] = x[:2]
        for q in range(2):
            qml.Hadamard(wires=q)
        for q in range(2):
            qml.PhaseShift(2.0 * feats[q], wires=q)
        qml.ControlledPhaseShift(
            2.0 * (np.pi - feats[0]) * (np.pi - feats[1]),
            wires=[0, 1],
        )
        return qml.state()

    states = [np.asarray(feature_state(x)) for x in X]
    pl_kernel = np.array([[abs(np.vdot(a, b)) ** 2 for b in states] for a in states])
    np.testing.assert_allclose(qforge_kernel, pl_kernel, atol=1e-12)


def test_qml_benchmark_suite_smoke():
    config = BenchConfig(n_warmup=0, n_runs=1, steps=1, max_qubits=4)
    suite = QMLBenchmarkSuite(config)

    results = suite.run()

    assert "ansatz_2q" in results
    assert "kernel_2q" in results
    assert "train_vqc_2q" in results
    assert results["ansatz_2q"]["qforge_ms"] >= 0.0
    assert results["kernel_2q"]["kernel_trace"] == pytest.approx(4.0)
    assert results["kernel_2q"]["min_eig"] > -1e-10
