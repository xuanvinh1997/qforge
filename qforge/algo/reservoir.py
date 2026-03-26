# -*- coding: utf-8 -*-
# author: vinhpx
"""Quantum Reservoir Computing.

A fixed (random) quantum circuit acts as a non-linear feature map.
Only the classical readout layer is trained, making this approach
extremely fast to train compared to variational methods.

Reference: Fujii & Nakajima — Physical Review Applied 8, 024030 (2017).

Example::

    from qforge.algo import QuantumReservoir
    import numpy as np

    X = np.random.randn(80, 3)
    y = (np.sin(X[:, 0]) + X[:, 1] > 0).astype(int)

    qr = QuantumReservoir(n_qubits=4, n_layers=3)
    qr.fit(X[:60], y[:60])
    preds = qr.predict(X[60:])
    print("Accuracy:", qr.score(X[60:], y[60:]))
"""
from __future__ import annotations
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import RX, RY, RZ, CNOT, H
from qforge.measurement import pauli_expectation


class QuantumReservoir:
    """Quantum Reservoir Computing for classification and regression.

    The reservoir circuit is a random (but fixed) parameterized circuit.
    Features are encoded, passed through the reservoir, and Pauli
    expectations are extracted as classical features for a linear model.

    Args:
        n_qubits:    Number of qubits in the reservoir.
        n_layers:    Number of random entangling layers.
        n_readout:   Number of Pauli observables per qubit (1-3 for Z, XZ, XYZ).
        task:        ``'classification'`` or ``'regression'``.
        regularization: Ridge regression regularization strength.
        random_state: Seed for the fixed reservoir circuit.
        backend:     qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        n_readout: int = 3,
        task: str = 'classification',
        regularization: float = 1e-4,
        random_state: int = 42,
        backend: str = 'auto',
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_readout = min(n_readout, 3)
        self.task = task
        self.regularization = regularization
        self.backend = backend

        # Generate fixed random reservoir parameters
        rng = np.random.default_rng(random_state)
        # Per layer: n_qubits * 3 rotation params
        self._reservoir_params = rng.uniform(
            0, 2 * np.pi, size=(n_layers, n_qubits, 3)
        )
        # Readout weights (trained)
        self._weights = None
        self._bias = None
        self._n_classes = None

    def _reservoir_circuit(self, x: np.ndarray) -> np.ndarray:
        """Run the fixed reservoir circuit and extract features."""
        wf = Qubit(self.n_qubits, backend=self.backend)

        # Encode input data
        for q in range(self.n_qubits):
            feat = x[q % len(x)] if len(x) > 0 else 0.0
            RX(wf, q, feat)

        # Fixed random reservoir layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                p = self._reservoir_params[layer, q]
                RX(wf, q, p[0])
                RY(wf, q, p[1])
                RZ(wf, q, p[2])
            # Entanglement
            for q in range(self.n_qubits - 1):
                CNOT(wf, q, q + 1)
            if self.n_qubits > 2:
                CNOT(wf, self.n_qubits - 1, 0)

        # Extract Pauli expectations as features
        paulis = ['Z', 'X', 'Y'][:self.n_readout]
        features = []
        for q in range(self.n_qubits):
            for p in paulis:
                features.append(pauli_expectation(wf, q, p))
        return np.array(features)

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract quantum reservoir features for all samples."""
        return np.array([self._reservoir_circuit(x) for x in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumReservoir':
        """Train the readout layer using ridge regression.

        Args:
            X: Training features, shape ``(n_samples, n_features)``.
            y: Labels — integers for classification, floats for regression.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Extract quantum features
        Q = self._extract_features(X)
        # Add bias column
        Q_aug = np.column_stack([Q, np.ones(len(Q))])

        if self.task == 'classification':
            self._n_classes = len(np.unique(y))
            # One-hot encode targets
            Y = np.zeros((len(y), self._n_classes))
            for i, label in enumerate(y):
                Y[i, int(label)] = 1.0
        else:
            Y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Ridge regression: W = (Q^T Q + λI)^{-1} Q^T Y
        lam = self.regularization
        A = Q_aug.T @ Q_aug + lam * np.eye(Q_aug.shape[1])
        self._weights = np.linalg.solve(A, Q_aug.T @ Y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        X = np.asarray(X, dtype=float)
        Q = self._extract_features(X)
        Q_aug = np.column_stack([Q, np.ones(len(Q))])
        raw = Q_aug @ self._weights
        # Softmax
        e = np.exp(raw - np.max(raw, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.task == 'classification':
            return np.argmax(self.predict_proba(X), axis=1)
        X = np.asarray(X, dtype=float)
        Q = self._extract_features(X)
        Q_aug = np.column_stack([Q, np.ones(len(Q))])
        return (Q_aug @ self._weights).ravel()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy (classification) or R² (regression)."""
        y = np.asarray(y)
        preds = self.predict(X)
        if self.task == 'classification':
            return np.mean(preds == y.astype(int))
        # R² score
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)
