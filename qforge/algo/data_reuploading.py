# -*- coding: utf-8 -*-
# author: vinhpx
"""Data Re-uploading Classifier.

A single-qubit (or few-qubit) classifier that re-encodes classical data at
every layer, interleaved with trainable rotations.  Universal approximation
on a single qubit was proven by Pérez-Salinas et al. (2020).

Reference: Pérez-Salinas et al. — Quantum 4, 226 (2020).

Example::

    from qforge.algo import DataReuploadingClassifier
    import numpy as np

    X = np.random.randn(60, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5).astype(int)

    clf = DataReuploadingClassifier(n_qubits=1, n_layers=6, n_classes=2)
    params, history = clf.fit(X[:50], y[:50], steps=80)
    preds = clf.predict(X[50:])
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import RX, RY, RZ, CNOT
from qforge.measurement import pauli_expectation
from qforge.algo.gradient import parameter_shift
from qforge.algo.optimizers import Adam


class DataReuploadingClassifier:
    """Data Re-uploading Classifier.

    Each layer applies:
        1. Data encoding: ``RX(w_x * x_0 + b_x)``, ``RY(w_y * x_1 + b_y)``, ... per qubit
        2. Trainable rotation: ``RZ(θ)`` per qubit
        3. Entanglement (if n_qubits > 1): CNOT ladder

    The key insight is that re-encoding data at every layer allows the
    circuit to learn non-linear decision boundaries even with few qubits.

    Args:
        n_qubits:  Number of qubits.
        n_layers:  Number of re-uploading layers.
        n_classes: Number of output classes.
        backend:   qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 6,
        n_classes: int = 2,
        backend: str = 'auto',
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.backend = backend
        # Per layer per qubit: 3 weights (for feature scaling) + 3 biases + 1 RZ
        # Simplified: per qubit per layer = 3 params (RX_weight, RY_weight, RZ_angle)
        # Plus n_features scaling weights per layer
        # We use: per layer per qubit: 4 params (w_x, w_y, b, rz_angle)
        self.n_params_per_qubit_per_layer = 4
        self.n_params = n_layers * n_qubits * self.n_params_per_qubit_per_layer
        self.params = None

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Run circuit with data re-uploading, return expectations."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        x = np.asarray(x, dtype=float)
        ppl = self.n_params_per_qubit_per_layer
        idx = 0

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                p = params[idx:idx + ppl]
                # Data re-uploading: encode features with trainable weights
                feat_x = x[q % len(x)] if len(x) > 0 else 0.0
                feat_y = x[(q + 1) % len(x)] if len(x) > 1 else 0.0
                RX(wf, q, p[0] * feat_x + p[2])
                RY(wf, q, p[1] * feat_y + p[2])
                RZ(wf, q, p[3])
                idx += ppl
            # Entanglement
            if self.n_qubits > 1:
                for q in range(self.n_qubits - 1):
                    CNOT(wf, q, q + 1)

        # Readout
        expectations = np.array([
            pauli_expectation(wf, q % self.n_qubits, 'Z')
            for q in range(self.n_classes)
        ])
        return expectations

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def _probabilities(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        return self._softmax(self._circuit(x, params))

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
        optimizer: object = None,
        steps: int = 100,
        batch_size: int | None = None,
        callback: Callable | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        """Train the classifier.

        Args:
            X:          Training features, shape ``(n_samples, n_features)``.
            y:          Integer labels, shape ``(n_samples,)``.
            params:     Initial parameters. Random if ``None``.
            optimizer:  Optimizer with ``step(params, grad)`` interface.
            steps:      Number of training steps.
            batch_size: Mini-batch size.
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
