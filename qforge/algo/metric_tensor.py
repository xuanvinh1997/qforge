# -*- coding: utf-8 -*-
# author: vinhpx
"""Fubini-Study metric tensor for variational quantum circuits."""
from __future__ import annotations

import numpy as np
from qforge.circuit import Qubit
from qforge.ir import Circuit


def fubini_study_metric(
    circuit: Circuit,
    params: np.ndarray,
    backend: str = 'python',
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the Fubini-Study metric tensor.

    Uses the parameter-shift rule to estimate:
        g_ij = Re(<d_i psi|d_j psi>) - Re(<d_i psi|psi>) * Re(<psi|d_j psi>)

    Args:
        circuit: Parameterised circuit.
        params:  Parameter values.
        backend: Simulation backend.
        shift:   Parameter shift (default pi/2).

    Returns:
        Metric tensor of shape (n_params, n_params).
    """
    params = np.asarray(params, dtype=float)
    n_params = len(params)

    amp_base = circuit.run(backend=backend, params=params).amplitude.copy()

    amps_plus = []
    amps_minus = []
    for i in range(n_params):
        p_plus = params.copy(); p_plus[i] += shift
        amps_plus.append(circuit.run(backend=backend, params=p_plus).amplitude.copy())
        p_minus = params.copy(); p_minus[i] -= shift
        amps_minus.append(circuit.run(backend=backend, params=p_minus).amplitude.copy())

    metric = np.zeros((n_params, n_params))
    denom = 2.0 * np.sin(shift)

    for i in range(n_params):
        d_i = (amps_plus[i] - amps_minus[i]) / denom
        for j in range(i, n_params):
            d_j = (amps_plus[j] - amps_minus[j]) / denom
            overlap_ij = np.dot(d_i.conj(), d_j)
            overlap_i0 = np.dot(d_i.conj(), amp_base)
            overlap_0j = np.dot(amp_base.conj(), d_j)
            g_ij = overlap_ij.real - overlap_i0.real * overlap_0j.real
            metric[i, j] = g_ij
            metric[j, i] = g_ij

    return metric
