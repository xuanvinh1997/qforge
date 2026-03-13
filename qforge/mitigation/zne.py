# -*- coding: utf-8 -*-
# author: vinhpx
"""Zero Noise Extrapolation (ZNE) for error mitigation."""
from __future__ import annotations

import numpy as np
from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp


def fold_circuit(circuit: Circuit, scale_factor: int) -> Circuit:
    """Fold a circuit to amplify noise by a given integer scale factor.

    For each gate U in the circuit, replace it with U (U† U)^(k-1)
    where k = scale_factor.  This preserves the ideal unitary while
    increasing the effective noise.

    Args:
        circuit:      Input circuit.
        scale_factor: Integer >= 1.  scale_factor=1 returns the original
                      circuit unchanged; scale_factor=2 inserts one U† U
                      after each gate; etc.

    Returns:
        A new Circuit with folded gates.
    """
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")
    if scale_factor == 1:
        return circuit.copy()

    folded = Circuit(circuit.n_qubits)
    for op in circuit.ops:
        if not isinstance(op, GateOp):
            folded.ops.append(op)
            continue
        # Original gate
        folded.ops.append(op)
        # Insert (U† U) pairs (scale_factor - 1) times
        adj_op = _adjoint_gate(op)
        for _ in range(scale_factor - 1):
            folded.ops.append(adj_op)
            folded.ops.append(op)
    return folded


def _adjoint_gate(op: GateOp) -> GateOp:
    """Return the adjoint of a GateOp for folding purposes."""
    from qforge.ir import _adjoint_op
    return _adjoint_op(op)


def _extrapolate_linear(scale_factors: np.ndarray,
                        expectations: np.ndarray) -> float:
    """Linear extrapolation to zero noise (scale_factor=0)."""
    coeffs = np.polyfit(scale_factors, expectations, deg=1)
    return float(np.polyval(coeffs, 0.0))


def _extrapolate_polynomial(scale_factors: np.ndarray,
                            expectations: np.ndarray) -> float:
    """Polynomial extrapolation to zero noise."""
    deg = min(len(scale_factors) - 1, 3)
    coeffs = np.polyfit(scale_factors, expectations, deg=deg)
    return float(np.polyval(coeffs, 0.0))


def _extrapolate_exponential(scale_factors: np.ndarray,
                             expectations: np.ndarray) -> float:
    """Exponential extrapolation: E(λ) = a * exp(b * λ) + c.

    Uses a simple approach: fit log(E - c_est) linearly,
    with c_est estimated from the trend.
    Falls back to linear if the fit fails.
    """
    try:
        # Try: E(λ) ≈ a * exp(b * λ) + c
        # Estimate c as the asymptotic value (use the largest scale_factor value)
        # Simple approach: fit a * exp(b * λ) by taking log
        shifted = expectations - expectations[-1] + 1e-10
        if np.all(shifted > 0):
            log_vals = np.log(shifted)
            coeffs = np.polyfit(scale_factors, log_vals, deg=1)
            a = np.exp(coeffs[1])
            b = coeffs[0]
            c = expectations[-1] - 1e-10
            return float(a * np.exp(b * 0.0) + c)
        else:
            return _extrapolate_linear(scale_factors, expectations)
    except (ValueError, np.linalg.LinAlgError):
        return _extrapolate_linear(scale_factors, expectations)


_EXTRAPOLATORS = {
    'linear': _extrapolate_linear,
    'polynomial': _extrapolate_polynomial,
    'exponential': _extrapolate_exponential,
}


def zero_noise_extrapolation(
    circuit_fn,
    executor,
    scale_factors: list[int],
    extrapolator: str = 'linear',
) -> float:
    """Perform Zero Noise Extrapolation (ZNE).

    Args:
        circuit_fn:     Callable that returns a Circuit.
        executor:       Callable ``(circuit) -> float`` that runs the circuit
                        and returns an expectation value.
        scale_factors:  List of noise scaling factors, e.g. [1, 2, 3].
        extrapolator:   Extrapolation method: 'linear', 'polynomial',
                        or 'exponential'.

    Returns:
        Mitigated expectation value extrapolated to zero noise.
    """
    if extrapolator not in _EXTRAPOLATORS:
        raise ValueError(
            f"Unknown extrapolator {extrapolator!r}. "
            f"Choose from {list(_EXTRAPOLATORS.keys())}"
        )

    circuit = circuit_fn()
    expectations = []
    for sf in scale_factors:
        folded = fold_circuit(circuit, sf)
        exp_val = executor(folded)
        expectations.append(exp_val)

    sf_arr = np.asarray(scale_factors, dtype=float)
    exp_arr = np.asarray(expectations, dtype=float)
    return _EXTRAPOLATORS[extrapolator](sf_arr, exp_arr)
