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


class SPSA:
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA estimates gradients using only two function evaluations per
    iteration regardless of parameter dimension, using random perturbations.

    Args:
        lr:           Learning rate (default 0.1).
        perturbation: Perturbation magnitude (default 0.1).
    """

    def __init__(self, lr: float = 0.1, perturbation: float = 0.1):
        self.lr = lr
        self.perturbation = perturbation

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return updated parameters using a pre-computed gradient.

        This method maintains API compatibility with GradientDescent/Adam.

        Args:
            params: Current parameter array.
            grad:   Pre-computed gradient array.

        Returns:
            Updated parameter array.
        """
        return np.asarray(params, dtype=float) - self.lr * np.asarray(grad, dtype=float)

    def estimate_and_step(self, params: np.ndarray, cost_fn) -> np.ndarray:
        """Estimate gradient via SPSA and take a step.

        Uses two cost function evaluations to estimate the full gradient.

        Args:
            params:  Current parameter array.
            cost_fn: Callable ``params -> float``.

        Returns:
            Updated parameter array.
        """
        params = np.asarray(params, dtype=float)
        delta = np.random.choice([-1, 1], size=len(params)).astype(float)
        perturbation = self.perturbation * delta

        f_plus = cost_fn(params + perturbation)
        f_minus = cost_fn(params - perturbation)

        grad_estimate = (f_plus - f_minus) / (2.0 * perturbation)
        return params - self.lr * grad_estimate


class LBFGS:
    """L-BFGS-B optimizer (wrapper around scipy.optimize.minimize).

    Unlike gradient-based step optimizers, L-BFGS manages its own
    iteration loop.  Use the :meth:`minimize` method.
    """

    def __init__(self):
        pass

    def minimize(
        self,
        cost_fn,
        params: np.ndarray,
        maxiter: int = 100,
    ) -> tuple[np.ndarray, float]:
        """Minimize a cost function using L-BFGS-B.

        Args:
            cost_fn: Callable ``params -> float``.
            params:  Initial parameter array.
            maxiter: Maximum number of iterations.

        Returns:
            Tuple of (optimized_params, final_cost).
        """
        from scipy.optimize import minimize as sp_minimize

        params = np.asarray(params, dtype=float)
        result = sp_minimize(
            cost_fn,
            params,
            method='L-BFGS-B',
            options={'maxiter': maxiter},
        )
        return result.x, float(result.fun)
