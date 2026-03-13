# -*- coding: utf-8 -*-
# author: vinhpx
"""Parameter-shift Hessian computation for variational quantum circuits."""
from __future__ import annotations

import numpy as np


def parameter_shift_hessian(
    cost_fn,
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the Hessian matrix via the double parameter-shift rule.

    For each pair (theta_i, theta_j):
        d2E/d_i d_j = (E(++)-E(+-)-E(-+)+E(--)) / (4*sin^2(s))

    Args:
        cost_fn: callable(params) -> float.
        params:  1-D parameter array.
        shift:   Shift amount (default pi/2).

    Returns:
        Hessian matrix of shape (n_params, n_params).
    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    hessian = np.zeros((n, n))
    denom = 4.0 * np.sin(shift) ** 2

    for i in range(n):
        for j in range(i, n):
            pp = params.copy(); pp[i] += shift; pp[j] += shift
            pm = params.copy(); pm[i] += shift; pm[j] -= shift
            mp = params.copy(); mp[i] -= shift; mp[j] += shift
            mm = params.copy(); mm[i] -= shift; mm[j] -= shift

            h_ij = (cost_fn(pp) - cost_fn(pm) - cost_fn(mp) + cost_fn(mm)) / denom
            hessian[i, j] = h_ij
            hessian[j, i] = h_ij

    return hessian
