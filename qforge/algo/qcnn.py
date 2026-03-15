# -*- coding: utf-8 -*-
# author: vinhpx
"""QCNN — Quantum Convolutional Neural Network.

Hierarchical circuit architecture inspired by classical CNNs:
convolution layers (two-qubit unitaries) followed by pooling layers
(measure-and-conditioned rotations that halve the active qubits).

Reference: Cong, Choi, Lukin — Nature Physics 15, 1273 (2019).

Example::

    from qforge.algo import QCNN
    import numpy as np

    X = np.random.randn(40, 4)
    y = (np.sum(X, axis=1) > 0).astype(int)

    qcnn = QCNN(n_qubits=4, n_classes=2)
    params, history = qcnn.fit(X[:30], y[:30], steps=60)
    preds = qcnn.predict(X[30:])
"""
from __future__ import annotations
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import RX, RY, RZ, CNOT, H
from qforge.measurement import pauli_expectation
from qforge.algo.gradient import parameter_shift
from qforge.algo.optimizers import Adam


class QCNN:
    """Quantum Convolutional Neural Network.

    Architecture (for 8 qubits):
        Layer 1 — Conv: two-qubit unitaries on pairs (0,1),(2,3),(4,5),(6,7)
        Layer 1 — Pool: entangle pairs → measure-out half → 4 active qubits
        Layer 2 — Conv: two-qubit unitaries on pairs of remaining qubits
        Layer 2 — Pool: → 2 active qubits
        Readout: Pauli-Z on final active qubits

    The number of qubits must be a power of 2.

    Args:
        n_qubits:  Number of qubits (must be power of 2, ≥ 2).
        n_classes: Number of output classes.
        backend:   qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_classes: int = 2,
        backend: str = 'auto',
    ):
        if n_qubits < 2 or (n_qubits & (n_qubits - 1)) != 0:
            raise ValueError(f"n_qubits must be a power of 2, got {n_qubits}")
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.backend = backend

        # Count layers: log2(n_qubits) pooling stages
        self.n_pool_layers = int(np.log2(n_qubits))
        # Parameters per conv pair: 6 (RY+RZ on each qubit + CNOT + RY+RZ)
        # Parameters per pool pair: 2 (RY on pooled-out qubit conditioned)
        self.n_params = self._count_params()
        self.params = None

    def _count_params(self) -> int:
        total = 0
        active = self.n_qubits
        for _ in range(self.n_pool_layers):
            n_pairs = active // 2
            total += n_pairs * 6  # conv: 6 params per pair
            total += n_pairs * 2  # pool: 2 params per pair
            active //= 2
        return total

    # ------------------------------------------------------------------
    # Circuit building blocks
    # ------------------------------------------------------------------

    def _conv_block(self, wf, q0: int, q1: int, params6: np.ndarray) -> None:
        """Two-qubit convolutional unitary with 6 parameters."""
        RY(wf, q0, params6[0])
        RZ(wf, q0, params6[1])
        RY(wf, q1, params6[2])
        RZ(wf, q1, params6[3])
        CNOT(wf, q0, q1)
        RY(wf, q1, params6[4])
        RZ(wf, q1, params6[5])

    def _pool_block(self, wf, q_measure: int, q_keep: int, params2: np.ndarray) -> None:
        """Pooling: conditional rotation to transfer info, then discard q_measure."""
        CNOT(wf, q_measure, q_keep)
        RY(wf, q_keep, params2[0])
        RZ(wf, q_keep, params2[1])

    def _encode(self, wf, x: np.ndarray) -> None:
        """Encode features via RX rotations."""
        for q in range(min(len(x), self.n_qubits)):
            RX(wf, q, x[q])

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Run full QCNN circuit, return Pauli-Z expectations on final qubits."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        self._encode(wf, x)

        active_qubits = list(range(self.n_qubits))
        idx = 0

        for _ in range(self.n_pool_layers):
            n_pairs = len(active_qubits) // 2
            # Convolution
            for p in range(n_pairs):
                q0 = active_qubits[2 * p]
                q1 = active_qubits[2 * p + 1]
                self._conv_block(wf, q0, q1, params[idx:idx + 6])
                idx += 6
            # Pooling
            kept = []
            for p in range(n_pairs):
                q_measure = active_qubits[2 * p]
                q_keep = active_qubits[2 * p + 1]
                self._pool_block(wf, q_measure, q_keep, params[idx:idx + 2])
                idx += 2
                kept.append(q_keep)
            active_qubits = kept

        # Readout
        expectations = np.array([
            pauli_expectation(wf, active_qubits[q % len(active_qubits)], 'Z')
            for q in range(self.n_classes)
        ])
        return expectations

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def _probabilities(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return self._softmax(self._circuit(x, params))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        loss = 0.0
        for i in range(len(X)):
            probs = self._probabilities(X[i], params)
            loss -= np.log(np.clip(probs[y[i]], 1e-12, 1.0))
        return loss / len(X)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray | None = None,
        optimizer=None,
        steps: int = 100,
        batch_size: int | None = None,
        callback=None,
    ) -> tuple[np.ndarray, list[float]]:
        """Train the QCNN classifier.

        Args:
            X:          Training features, shape ``(n_samples, n_features)``.
            y:          Integer labels, shape ``(n_samples,)``.
            params:     Initial parameters. Random if ``None``.
            optimizer:  Optimizer with ``step(params, grad)`` interface.
            steps:      Number of training steps.
            batch_size: Mini-batch size. ``None`` uses full dataset.
            callback:   Optional ``callable(step, params, loss)``.

        Returns:
            ``(optimal_params, loss_history)``
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if optimizer is None:
            optimizer = Adam(lr=0.05)
        if params is None:
            rng = np.random.default_rng(42)
            params = rng.uniform(-np.pi, np.pi, self.n_params)
        else:
            params = np.asarray(params, dtype=float).copy()

        history: list[float] = []
        n = len(X)

        for step in range(steps):
            if batch_size is not None and batch_size < n:
                idx = np.random.choice(n, batch_size, replace=False)
                X_b, y_b = X[idx], y[idx]
            else:
                X_b, y_b = X, y

            cost_fn = lambda p: self._loss(p, X_b, y_b)
            loss = cost_fn(params)
            history.append(loss)
            grad = parameter_shift(cost_fn, params)
            params = optimizer.step(params, grad)
            if callback is not None:
                callback(step, params, loss)

        self.params = params
        return params, history

    def predict_proba(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Predict class probabilities."""
        if params is None:
            params = self.params
        X = np.asarray(X, dtype=float)
        return np.array([self._probabilities(x, params) for x in X])

    def predict(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X, params), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray, params: np.ndarray | None = None) -> float:
        """Classification accuracy."""
        return np.mean(self.predict(X, params) == np.asarray(y, dtype=int))
