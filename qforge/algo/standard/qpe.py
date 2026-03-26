# -*- coding: utf-8 -*-
"""Quantum Phase Estimation (QPE)."""
from __future__ import annotations

import numpy as np
from qforge import gates
from qforge.circuit import Qubit
from qforge.algo.standard.qft import inverse_qft
from qforge.measurement import measure_all


def qpe(unitary_fn, n_ancilla: int, target_state_fn=None,
         n_target: int = 1, backend: str = 'auto',
         n_samples: int = 1000) -> dict:
    """Quantum Phase Estimation.

    Estimates the eigenphase of a unitary operator. Given U|ψ⟩ = e^{2πiφ}|ψ⟩,
    QPE estimates φ using `n_ancilla` precision qubits.

    Args:
        unitary_fn:       Callable ``(wf, target_qubits, power)`` that applies U^power
                          to the target qubits of the wavefunction.
        n_ancilla:        Number of ancilla (precision) qubits.
        target_state_fn:  Optional callable ``(wf, target_qubits)`` to prepare the
                          eigenstate on the target register. If None, target stays |0⟩.
        n_target:         Number of target qubits (default 1).
        backend:          Simulation backend.
        n_samples:        Number of measurement samples.

    Returns:
        Dict with keys:
        - ``'phase'``: Estimated phase φ ∈ [0, 1).
        - ``'phase_binary'``: Binary string of the ancilla measurement.
        - ``'samples'``: Raw measurement results.
        - ``'counts'``: Measurement counts.

    Example::

        # Estimate the phase of a Z gate: Z|1⟩ = e^{iπ}|1⟩ → φ = 0.5
        def apply_z_power(wf, targets, power):
            for _ in range(power):
                gates.Z(wf, targets[0])

        def prep_eigenstate(wf, targets):
            gates.X(wf, targets[0])  # Prepare |1⟩

        result = qpe(apply_z_power, n_ancilla=3, target_state_fn=prep_eigenstate)
        print(result['phase'])  # ≈ 0.5
    """
    total_qubits = n_ancilla + n_target
    wf = Qubit(total_qubits, backend=backend)

    ancilla_qubits = list(range(n_ancilla))
    target_qubits = list(range(n_ancilla, total_qubits))

    # Prepare eigenstate on target register
    if target_state_fn is not None:
        target_state_fn(wf, target_qubits)

    # Hadamard on all ancilla qubits
    for q in ancilla_qubits:
        gates.H(wf, q)

    # Controlled-U^(2^k) for each ancilla qubit k
    for k in range(n_ancilla):
        power = 2 ** (n_ancilla - 1 - k)
        # Apply controlled U^power: control = ancilla[k], targets = target_qubits
        # We need controlled-unitary. Since we can't generically control arbitrary
        # unitaries, the user provides a function that accepts power.
        # We implement the controlled version by:
        # 1. Check control qubit via measurement-free approach
        _controlled_unitary_power(wf, ancilla_qubits[k], target_qubits,
                                   unitary_fn, power)

    # Inverse QFT on ancilla register
    inverse_qft(wf, ancilla_qubits)

    # Measure ancilla qubits
    states, counts = measure_all(wf, n_samples)

    # Extract best result
    best_idx = np.argmax(counts)
    best_state = states[best_idx]

    # Extract ancilla bits from measurement result
    ancilla_bits = best_state[:n_ancilla]
    phase = int(ancilla_bits, 2) / (2 ** n_ancilla)

    return {
        'phase': phase,
        'phase_binary': ancilla_bits,
        'samples': states,
        'counts': counts,
    }


def _controlled_unitary_power(wf, control: int, targets: list[int],
                               unitary_fn, power: int) -> None:
    """Apply controlled-U^power using amplitude manipulation.

    This works for any unitary but requires statevector access.
    The approach: save state, apply U^power conditionally on control=|1>.
    """
    from qforge._utils import _nq
    nq = _nq(wf)
    amp = wf.amplitude.copy()
    ctrl_bit = nq - control - 1

    # Zero out amplitudes where control=1, apply U^power, then combine
    # Save original amplitudes
    dim = len(amp)

    # Create state with only control=0 amplitudes
    amp_ctrl0 = amp.copy()
    amp_ctrl1 = amp.copy()
    for i in range(dim):
        if (i >> ctrl_bit) & 1:
            amp_ctrl0[i] = 0
        else:
            amp_ctrl1[i] = 0

    # Apply U^power to the ctrl=1 subspace
    wf.amplitude = amp_ctrl1
    unitary_fn(wf, targets, power)
    amp_ctrl1_evolved = wf.amplitude.copy()

    # Combine: final = ctrl0_unchanged + ctrl1_evolved
    wf.amplitude = amp_ctrl0 + amp_ctrl1_evolved
