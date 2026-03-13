# -*- coding: utf-8 -*-
# author: vinhpx
"""Classical optimizers for variational quantum algorithms."""
from __future__ import annotations
import numpy as np


class GradientDescent:
    """Vanilla gradient-descent optimizer.

    Args:
        lr: Learning rate (default 0.1).
    """

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return ``params − lr * grad``."""
        return np.asarray(params, dtype=float) - self.lr * np.asarray(grad, dtype=float)


class Adam:
    """Adam optimizer (Kingma & Ba, 2014).

    Args:
        lr:    Learning rate (default 0.01).
        beta1: First-moment decay (default 0.9).
        beta2: Second-moment decay (default 0.999).
        eps:   Numerical stability constant (default 1e-8).
    """

    def __init__(self, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset moment estimates (call before reusing on a new problem)."""
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return bias-corrected Adam update."""
        params = np.asarray(params, dtype=float)
        grad = np.asarray(grad, dtype=float)
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
