# -*- coding: utf-8 -*-
# author: vinhpx
"""VQC — Variational Quantum Classifier.

Parameterized quantum circuit trained end-to-end for classification tasks.
Supports binary and multi-class classification with softmax readout.

Example::

    from qforge.algo import VQC
    import numpy as np

    X = np.random.randn(60, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int)

    vqc = VQC(n_qubits=2, n_layers=3, n_classes=2)
    params, history = vqc.fit(X[:50], y[:50], steps=80)
    preds = vqc.predict(X[50:])
"""
from __future__ import annotations
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import H, RX, RY, RZ, CNOT
from qforge.measurement import pauli_expectation
from qforge.algo.gradient import parameter_shift
from qforge.algo.optimizers import Adam


class VQC:
    """Variational Quantum Classifier.

    Architecture:
        1. Feature encoding: ``RX(x_i)`` per qubit (re-scaled to [0, 2π]).
        2. Variational layers: ``RY/RZ`` rotations + CNOT entanglement.
        3. Readout: Pauli-Z expectations on the first ``n_classes`` qubits,
           passed through softmax to produce class probabilities.

    Args:
        n_qubits:  Number of qubits (≥ max(n_features, n_classes)).
        n_layers:  Number of variational layers.
        n_classes: Number of output classes.
        backend:   qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        n_classes: int = 2,
        backend: str = 'auto',
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.backend = backend
        # Parameters: n_layers * n_qubits * 2 (RY + RZ per qubit per layer)
        self.n_params = n_layers * n_qubits * 2
        self.params = None

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------

    def _encode(self, wf, x: np.ndarray) -> None:
        """Encode features into the circuit via RX rotations."""
        for q in range(min(len(x), self.n_qubits)):
            RX(wf, q, x[q])

    def _variational_layer(self, wf, params_slice: np.ndarray) -> None:
        """One variational layer: RY + RZ rotations + CNOT ladder."""
        n = self.n_qubits
        for q in range(n):
            RY(wf, q, params_slice[2 * q])
            RZ(wf, q, params_slice[2 * q + 1])
        for q in range(n - 1):
            CNOT(wf, q, q + 1)

    def _circuit(self, x: np.ndarray, params: np.ndarray):
        """Run full circuit and return class probabilities."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        # Encode
        self._encode(wf, x)
        # Variational layers
        layer_size = self.n_qubits * 2
        for l in range(self.n_layers):
            self._variational_layer(wf, params[l * layer_size:(l + 1) * layer_size])
        # Readout: Pauli-Z expectations → softmax
        expectations = np.array([
            pauli_expectation(wf, q, 'Z') for q in range(self.n_classes)
        ])
        return expectations

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def _probabilities(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Class probabilities for a single sample."""
        return self._softmax(self._circuit(x, params))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _cross_entropy_loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Average cross-entropy loss over the dataset."""
        loss = 0.0
        for i in range(len(X)):
            probs = self._probabilities(X[i], params)
            loss -= np.log(np.clip(probs[y[i]], 1e-12, 1.0))
        return loss / len(X)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
        """Train the classifier.

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
            # Mini-batch
            if batch_size is not None and batch_size < n:
                idx = np.random.choice(n, batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]
            else:
                X_batch, y_batch = X, y

            cost_fn = lambda p: self._cross_entropy_loss(p, X_batch, y_batch)
            loss = cost_fn(params)
            history.append(loss)
            grad = parameter_shift(cost_fn, params)
            params = optimizer.step(params, grad)
            if callback is not None:
                callback(step, params, loss)

        self.params = params
        return params, history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features, shape ``(n_samples, n_features)``.

        Returns:
            Probabilities, shape ``(n_samples, n_classes)``.
        """
        if params is None:
            params = self.params
        X = np.asarray(X, dtype=float)
        return np.array([self._probabilities(x, params) for x in X])

    def predict(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features, shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels, shape ``(n_samples,)``.
        """
        return np.argmax(self.predict_proba(X, params), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray, params: np.ndarray | None = None) -> float:
        """Classification accuracy."""
        return np.mean(self.predict(X, params) == np.asarray(y, dtype=int))
