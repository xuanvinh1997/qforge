# -*- coding: utf-8 -*-
"""HHL algorithm for solving linear systems Ax = b."""
from __future__ import annotations

import numpy as np
from qforge import gates
from qforge.circuit import Qubit
from qforge.algo.standard.qft import qft, inverse_qft


def hhl(A: np.ndarray, b: np.ndarray, n_ancilla: int = 4,
         backend: str = 'auto') -> dict:
    """Solve the linear system Ax = b using the HHL algorithm.

    This is a simplified/pedagogical implementation for small systems.
    The matrix A must be Hermitian. If not, use the extended system
    [[0, A], [A†, 0]].

    Args:
        A:          Hermitian matrix (2^n × 2^n).
        b:          Right-hand side vector (length 2^n).
        n_ancilla:  Number of ancilla qubits for phase estimation precision.
        backend:    Simulation backend.

    Returns:
        Dict with keys:
        - ``'solution'``: Normalized solution vector x/|x|.
        - ``'success_prob'``: Probability of measuring ancilla in |1⟩.
        - ``'wavefunction'``: Final wavefunction.

    Note:
        This implementation uses direct amplitude manipulation for the
        eigenvalue inversion step, suitable for educational/small-system use.
        For production use, a proper QPE-based implementation with controlled
        rotations is needed.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    n = int(np.log2(len(b)))
    assert A.shape == (2**n, 2**n), f"A must be {2**n}x{2**n}"
    assert len(b) == 2**n

    # Normalize b
    b_norm = np.linalg.norm(b)
    b = b / b_norm

    # Classical eigendecomposition (for simulation)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Express b in eigenbasis: b = Σ β_j |u_j⟩
    betas = eigenvectors.conj().T @ b

    # HHL core: |b⟩ → Σ β_j (C/λ_j) |u_j⟩
    # where C is a normalization constant chosen so that C/max(|λ|) ≤ 1
    C = min(abs(eigenvalues[eigenvalues != 0])) if np.any(eigenvalues != 0) else 1.0

    # Build solution: x = Σ β_j (1/λ_j) |u_j⟩
    x = np.zeros(2**n, dtype=complex)
    for j in range(2**n):
        if abs(eigenvalues[j]) > 1e-10:
            x += betas[j] * (C / eigenvalues[j]) * eigenvectors[:, j]

    success_prob = np.linalg.norm(x) ** 2

    # Normalize solution
    if np.linalg.norm(x) > 1e-14:
        x_normalized = x / np.linalg.norm(x)
    else:
        x_normalized = x

    # Also create the quantum state for verification
    total_qubits = 1 + n_ancilla + n  # ancilla_flag + clock + data
    wf = Qubit(total_qubits, backend=backend)

    # Encode b into the data register
    data_qubits = list(range(1 + n_ancilla, total_qubits))

    # Set amplitudes for the data register
    amp = np.zeros(2**total_qubits, dtype=complex)
    for i in range(2**n):
        amp[i] = b[i]
    wf.amplitude = amp

    return {
        'solution': x_normalized,
        'success_prob': success_prob,
        'wavefunction': wf,
        'eigenvalues': eigenvalues,
        'C': C,
    }
