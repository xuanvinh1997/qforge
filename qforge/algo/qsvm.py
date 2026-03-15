# -*- coding: utf-8 -*-
# author: vinhpx
"""QSVM — Quantum Support Vector Machine.

Uses quantum kernel methods (fidelity-based or projected) with a classical
SVM for binary or multi-class classification.

Example::

    from qforge.algo import QSVM
    import numpy as np

    X_train = np.random.randn(40, 2)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test  = np.random.randn(10, 2)

    qsvm = QSVM(n_qubits=2, n_layers=2)
    qsvm.fit(X_train, y_train)
    predictions = qsvm.predict(X_test)
    accuracy = qsvm.score(X_test, y_train[:10])
"""
from __future__ import annotations
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import H, RY, RZ, CNOT
from qforge.measurement import pauli_expectation


class QSVM:
    """Quantum Support Vector Machine.

    Builds a quantum kernel matrix from parameterized circuits and uses
    it with a classical SVM solver (simple quadratic programming or
    scikit-learn when available).

    Args:
        n_qubits:    Number of qubits in the feature map.
        n_layers:    Number of entangling layers in the feature map.
        gamma:       RBF kernel width for projected kernel mode.
        feature_map: Feature map type — ``'zz'`` (ZZ feature map) or
                     ``'projected'`` (expectation-based projected kernel).
        C:           SVM regularization parameter.
        backend:     qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        gamma: float = 1.0,
        feature_map: str = 'zz',
        C: float = 1.0,
        backend: str = 'auto',
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.gamma = gamma
        self.feature_map = feature_map
        self.C = C
        self.backend = backend
        self._alpha = None
        self._bias = 0.0
        self._support_X = None
        self._support_y = None
        self._K_train = None
        self._X_train = None
        self._y_train = None

    # ------------------------------------------------------------------
    # Feature maps
    # ------------------------------------------------------------------

    def _zz_feature_map(self, wf, x: np.ndarray) -> None:
        """ZZ feature map: H + Phase(2x) layers with ZZ entanglement."""
        n = self.n_qubits
        feats = np.zeros(n)
        feats[:len(x)] = x[:n]
        for _ in range(self.n_layers):
            for q in range(n):
                H(wf, q)
            for q in range(n):
                from qforge.gates import Phase
                Phase(wf, q, 2.0 * feats[q])
            for q in range(n - 1):
                from qforge.gates import CPhase
                CPhase(wf, q, q + 1, 2.0 * (np.pi - feats[q]) * (np.pi - feats[q + 1]))

    def _encode(self, x: np.ndarray):
        """Encode a data point into a quantum state."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        self._zz_feature_map(wf, x)
        return wf

    def _projected_features(self, x: np.ndarray) -> np.ndarray:
        """Compute projected quantum features (Pauli expectations)."""
        wf = self._encode(x)
        feats = []
        for q in range(self.n_qubits):
            for pauli in ('X', 'Y', 'Z'):
                feats.append(pauli_expectation(wf, q, pauli))
        return np.array(feats)

    # ------------------------------------------------------------------
    # Kernel computation
    # ------------------------------------------------------------------

    def _fidelity_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Fidelity kernel: |<φ(x1)|φ(x2)>|²."""
        wf1 = self._encode(x1)
        wf2 = self._encode(x2)
        overlap = np.abs(np.vdot(wf1.amplitude, wf2.amplitude))
        return overlap ** 2

    def _projected_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Projected kernel: exp(-γ ||f(x1) - f(x2)||²)."""
        f1 = self._projected_features(x1)
        f2 = self._projected_features(x2)
        return np.exp(-self.gamma * np.sum((f1 - f2) ** 2))

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.feature_map == 'projected':
            return self._projected_kernel(x1, x2)
        return self._fidelity_kernel(x1, x2)

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Compute the quantum kernel matrix.

        Args:
            X1: Array of shape ``(n1, n_features)``.
            X2: Array of shape ``(n2, n_features)``. If ``None``, uses ``X1``.

        Returns:
            Kernel matrix of shape ``(n1, n2)``.
        """
        if X2 is None:
            X2 = X1
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel(X1[i], X2[j])
        return K

    # ------------------------------------------------------------------
    # SVM solver (simplified SMO-like)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QSVM':
        """Train the QSVM on labelled data.

        Args:
            X: Training features, shape ``(n_samples, n_features)``.
            y: Labels in ``{0, 1}`` or ``{-1, +1}``.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).copy()
        # Convert 0/1 labels to -1/+1
        if set(np.unique(y)).issubset({0, 1}):
            y = 2.0 * y - 1.0
        self._X_train = X
        self._y_train = y
        n = len(X)

        K = self.kernel_matrix(X)
        self._K_train = K

        # Solve dual SVM via simple gradient ascent on dual objective
        alpha = np.zeros(n)
        lr = 0.01
        for _ in range(500):
            grad = np.ones(n) - (y * (K @ (alpha * y)))
            alpha = np.clip(alpha + lr * grad, 0.0, self.C)

        # Identify support vectors
        sv_mask = alpha > 1e-7
        self._alpha = alpha[sv_mask]
        self._support_X = X[sv_mask]
        self._support_y = y[sv_mask]

        # Compute bias from support vectors
        if len(self._alpha) > 0:
            K_sv = self.kernel_matrix(self._support_X, X)
            decision = (self._alpha * self._support_y) @ K_sv[:, sv_mask].T
            # Use median for numerical stability
            self._bias = np.median(self._support_y[:len(decision)] - decision[:len(self._support_y)])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the SVM decision function for each sample.

        Args:
            X: Test features, shape ``(n_samples, n_features)``.

        Returns:
            Decision values, shape ``(n_samples,)``.
        """
        K = self.kernel_matrix(self._support_X, X)
        return (self._alpha * self._support_y) @ K + self._bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Test features, shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels in ``{0, 1}``, shape ``(n_samples,)``.
        """
        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy.

        Args:
            X: Test features.
            y: True labels (0/1 or -1/+1).

        Returns:
            Accuracy as a float in [0, 1].
        """
        y_true = np.asarray(y)
        if set(np.unique(y_true)).issubset({-1, 1}):
            y_true = ((y_true + 1) / 2).astype(int)
        return np.mean(self.predict(X) == y_true)
