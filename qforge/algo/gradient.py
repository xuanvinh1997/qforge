# -*- coding: utf-8 -*-
# author: vinhpx
"""Parameter-shift gradient rule for variational quantum circuits."""
from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def parameter_shift(
    cost_fn,
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the gradient via the parameter-shift rule.

    For each parameter θᵢ::

        ∂E/∂θᵢ = (E(θ⁺ᵢ) − E(θ⁻ᵢ)) / (2 sin(shift))

    A shift of π/2 gives exact gradients for RX, RY, and RZ gates.

    Args:
        cost_fn: ``callable(params) → float`` — evaluates the cost at ``params``.
        params:  1-D parameter array.
        shift:   Shift amount (default π/2).

    Returns:
        Gradient array with the same shape as ``params``.
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    denom = 2.0 * np.sin(shift)
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += shift
        p_minus = params.copy(); p_minus[i] -= shift
        grad[i] = (cost_fn(p_plus) - cost_fn(p_minus)) / denom
    return grad


def parallel_parameter_shift(
    cost_fn,
    params: np.ndarray,
    shift: float = np.pi / 2,
    max_workers: int | None = None,
) -> np.ndarray:
    """Compute the parameter-shift gradient with concurrent circuit evaluations.

    Evaluates all ``(θ⁺ᵢ, θ⁻ᵢ)`` pairs in parallel using a thread pool.
    The GIL is released during NumPy operations and C-extension calls (CPU
    OpenMP kernels, CUDA runtime), so multiple evaluations overlap in
    practice — particularly effective for the CPU backend and for circuits
    where gate kernels dominate wall time.

    For the CUDA backend on a single GPU, threads share the same device and
    serialize on SM access; speedup there is modest. For best CUDA throughput
    use the batched-circuit API instead (Phase 4B).

    Args:
        cost_fn:     ``callable(params) → float``.  Must be thread-safe —
                     each call must construct its own independent state
                     (e.g. via ``Qubit(n)``) rather than mutating shared state.
        params:      1-D parameter array.
        shift:       Shift amount (default π/2).
        max_workers: Thread-pool size.  ``None`` lets Python choose
                     (typically ``min(32, cpu_count + 4)``).

    Returns:
        Gradient array with the same shape as ``params``.
    """
    params = np.asarray(params, dtype=float)
    denom = 2.0 * np.sin(shift)
    N = len(params)

    def _eval_pair(i: int) -> tuple[float, float]:
        p_plus  = params.copy(); p_plus[i]  += shift
        p_minus = params.copy(); p_minus[i] -= shift
        return cost_fn(p_plus), cost_fn(p_minus)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pairs = list(pool.map(_eval_pair, range(N)))

    grad = np.empty(N, dtype=float)
    for i, (fp, fm) in enumerate(pairs):
        grad[i] = (fp - fm) / denom
    return grad
