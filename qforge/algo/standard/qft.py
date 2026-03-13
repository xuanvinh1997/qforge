# -*- coding: utf-8 -*-
"""Quantum Fourier Transform."""
from __future__ import annotations

import numpy as np
from qforge import gates


def qft(wavefunction, qubits: list[int] | None = None) -> None:
    """Apply the Quantum Fourier Transform to the given qubits.

    Args:
        wavefunction: Target state (Wavefunction or MPS).
        qubits:       Qubit indices to apply QFT on. If None, applies to all qubits.
    """
    from qforge._utils import _nq
    nq = _nq(wavefunction)
    if qubits is None:
        qubits = list(range(nq))
    n = len(qubits)

    for i in range(n):
        gates.H(wavefunction, qubits[i])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            gates.CPhase(wavefunction, qubits[j], qubits[i], angle)

    # Swap qubits to reverse order (QFT convention)
    for i in range(n // 2):
        gates.SWAP(wavefunction, qubits[i], qubits[n - 1 - i])


def inverse_qft(wavefunction, qubits: list[int] | None = None) -> None:
    """Apply the inverse Quantum Fourier Transform.

    Args:
        wavefunction: Target state.
        qubits:       Qubit indices. If None, applies to all qubits.
    """
    from qforge._utils import _nq
    nq = _nq(wavefunction)
    if qubits is None:
        qubits = list(range(nq))
    n = len(qubits)

    # Reverse the swaps
    for i in range(n // 2):
        gates.SWAP(wavefunction, qubits[i], qubits[n - 1 - i])

    # Reverse the gate sequence with negated phases
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -np.pi / (2 ** (j - i))
            gates.CPhase(wavefunction, qubits[j], qubits[i], angle)
        gates.H(wavefunction, qubits[i])
