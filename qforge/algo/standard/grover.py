# -*- coding: utf-8 -*-
"""Grover's search algorithm."""
from __future__ import annotations

import math
from typing import Callable
import numpy as np
from qforge import gates
from qforge.circuit import Qubit
from qforge.measurement import measure_all


def grover_oracle(wavefunction: object, marked_states: list[int], n_qubits: int) -> None:
    """Apply the Grover oracle: flip the sign of marked states.

    Args:
        wavefunction: Target state.
        marked_states: List of integer indices of marked states (e.g., [5] for |101⟩).
        n_qubits:      Number of qubits.
    """
    amp = wavefunction.amplitude.copy()
    for s in marked_states:
        amp[s] *= -1
    wavefunction.amplitude = amp


def grover_diffusion(wavefunction: object, qubits: list[int]) -> None:
    """Apply the Grover diffusion operator: 2|ψ⟩⟨ψ| - I where |ψ⟩ = H⊗n|0⟩.

    This is: H⊗n · (2|0⟩⟨0| - I) · H⊗n

    Args:
        wavefunction: Target state.
        qubits:       Qubit indices to apply diffusion on.
    """
    n = len(qubits)

    # Apply H to all qubits
    for q in qubits:
        gates.H(wavefunction, q)

    # Apply 2|0><0| - I  (flip all except |0>)
    # This is equivalent to: X⊗n · MCZ · X⊗n · global_phase(-1)
    for q in qubits:
        gates.X(wavefunction, q)

    # Multi-controlled Z on last qubit with all others as controls
    if n == 1:
        gates.Z(wavefunction, qubits[0])
    elif n == 2:
        gates.CPhase(wavefunction, qubits[0], qubits[1], np.pi)
    else:
        gates.mcz(wavefunction, qubits[:-1], qubits[-1])

    for q in qubits:
        gates.X(wavefunction, q)

    # Apply H to all qubits
    for q in qubits:
        gates.H(wavefunction, q)


def grover_search(oracle_fn: Callable, n_qubits: int, n_iterations: int | None = None,
                   backend: str = 'auto', n_samples: int = 1000) -> dict:
    """Run Grover's search algorithm.

    Args:
        oracle_fn:     Callable ``(wf, n_qubits)`` that applies the oracle
                       (sign-flip on marked states).
        n_qubits:      Number of search qubits.
        n_iterations:  Number of Grover iterations. If None, uses optimal
                       ⌊π/4 · √N⌋ where N = 2^n_qubits.
        backend:       Simulation backend.
        n_samples:     Number of measurement samples.

    Returns:
        Dict with keys:
        - ``'result'``:  Most likely bitstring.
        - ``'result_int'``: Integer value of the result.
        - ``'probability'``: Probability of the result.
        - ``'samples'``: All measurement results.
        - ``'counts'``: Measurement counts.

    Example::

        # Search for |101⟩ = 5 in a 3-qubit space
        def oracle(wf, n):
            grover_oracle(wf, [5], n)

        result = grover_search(oracle, n_qubits=3)
        print(result['result'])       # '101'
        print(result['result_int'])   # 5
    """
    N = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = max(1, int(math.pi / 4 * math.sqrt(N)))

    qubits = list(range(n_qubits))
    wf = Qubit(n_qubits, backend=backend)

    # Initial superposition
    for q in qubits:
        gates.H(wf, q)

    # Grover iterations
    for _ in range(n_iterations):
        oracle_fn(wf, n_qubits)
        grover_diffusion(wf, qubits)

    # Measure
    states, counts = measure_all(wf, n_samples)

    best_idx = np.argmax(counts)
    best_state = states[best_idx]
    best_prob = counts[best_idx] / n_samples

    return {
        'result': best_state,
        'result_int': int(best_state, 2),
        'probability': best_prob,
        'samples': states,
        'counts': counts,
    }
