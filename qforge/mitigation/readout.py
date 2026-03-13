# -*- coding: utf-8 -*-
# author: vinhpx
"""Readout error calibration and correction."""
from __future__ import annotations

import numpy as np


def calibrate_readout(
    executor,
    n_qubits: int,
    n_shots: int = 1000,
) -> np.ndarray:
    """Build a readout confusion matrix via calibration circuits.

    Runs calibration circuits for the all-|0> and all-|1> states,
    then for each individual qubit flipped to |1>.

    The executor should accept a "calibration state" (a list of 0/1
    specifying which qubits to prepare as |1>) and return a dict of
    counts: ``{bitstring: count}``.

    Args:
        executor:  Callable ``(prep_state: list[int]) -> dict[str, int]``
                   that prepares the given state, measures, and returns counts.
        n_qubits:  Number of qubits.
        n_shots:   Number of measurement shots per calibration circuit.

    Returns:
        Confusion matrix M of shape (2^n, 2^n) where M[i][j] = P(measure i | true state j).
    """
    n_states = 2 ** n_qubits
    confusion = np.zeros((n_states, n_states), dtype=float)

    # Run calibration for each computational basis state
    for j in range(n_states):
        # Prepare state |j> (binary representation)
        prep_state = [(j >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        counts = executor(prep_state)

        total = sum(counts.values())
        if total == 0:
            continue

        for bitstring, count in counts.items():
            # Convert bitstring to integer index
            if isinstance(bitstring, str):
                i = int(bitstring, 2)
            else:
                i = int(bitstring)
            confusion[i, j] = count / total

    return confusion


def correct_readout(
    counts: dict,
    calibration_matrix: np.ndarray,
) -> dict:
    """Apply readout error correction to measurement counts.

    Uses the pseudo-inverse of the confusion matrix for numerical stability.

    Args:
        counts:             Dict of ``{bitstring: count}`` from a measurement.
        calibration_matrix: Confusion matrix from :func:`calibrate_readout`.

    Returns:
        Corrected counts dict with the same bitstring keys.
    """
    n_states = calibration_matrix.shape[0]
    n_qubits = int(np.log2(n_states))

    # Build the raw probability vector
    total_counts = sum(counts.values())
    if total_counts == 0:
        return dict(counts)

    raw_probs = np.zeros(n_states, dtype=float)
    for bitstring, count in counts.items():
        if isinstance(bitstring, str):
            idx = int(bitstring, 2)
        else:
            idx = int(bitstring)
        raw_probs[idx] = count / total_counts

    # Apply pseudo-inverse of confusion matrix
    cal_inv = np.linalg.pinv(calibration_matrix)
    corrected_probs = cal_inv @ raw_probs

    # Clip negative values and renormalize
    corrected_probs = np.maximum(corrected_probs, 0.0)
    total_prob = corrected_probs.sum()
    if total_prob > 0:
        corrected_probs /= total_prob

    # Convert back to counts
    corrected_counts = {}
    for idx in range(n_states):
        if corrected_probs[idx] > 1e-10:
            bitstring = format(idx, f'0{n_qubits}b')
            corrected_counts[bitstring] = corrected_probs[idx] * total_counts

    return corrected_counts
