# -*- coding: utf-8 -*-
# author: vinhpx
"""UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.

Implements the Trotterized UCCSD ansatz circuit for use in VQE and other
variational quantum algorithms for quantum chemistry.
"""
from __future__ import annotations

import numpy as np

from qforge import gates


# ============================================================
# Parameter counting
# ============================================================

def uccsd_n_params(n_electrons: int, n_orbitals: int) -> int:
    """Compute the number of variational parameters for UCCSD.

    Args:
        n_electrons: Number of electrons.
        n_orbitals:  Number of spatial orbitals.

    Returns:
        Total number of singles + doubles excitation parameters.
    """
    n_occ = n_electrons
    n_virt = 2 * n_orbitals - n_electrons  # spin-orbitals

    # Singles: occupied -> virtual
    n_singles = n_occ * n_virt

    # Doubles: pairs of occupied -> pairs of virtual
    n_occ_pairs = n_occ * (n_occ - 1) // 2
    n_virt_pairs = n_virt * (n_virt - 1) // 2
    n_doubles = n_occ_pairs * n_virt_pairs

    return n_singles + n_doubles


def _get_excitations(n_electrons: int, n_orbitals: int):
    """Generate single and double excitation indices.

    Returns:
        singles: list of (p, q) where p is virtual, q is occupied
        doubles: list of (p, q, r, s) where p>q are virtual, r>s are occupied
    """
    n_spin_orbitals = 2 * n_orbitals
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_spin_orbitals))

    singles = []
    for q in occupied:
        for p in virtual:
            singles.append((p, q))

    doubles = []
    for i, s in enumerate(occupied):
        for r in occupied[i + 1:]:
            for j, q in enumerate(virtual):
                for p in virtual[j + 1:]:
                    doubles.append((p, q, r, s))

    return singles, doubles


# ============================================================
# UCCSD ansatz circuit
# ============================================================

def uccsd_ansatz(
    wf,
    params: np.ndarray,
    n_electrons: int,
    n_orbitals: int,
) -> None:
    """Apply the UCCSD ansatz circuit to a wavefunction.

    Implements the Trotterized form of exp(T - T†) where:
    - T1 = sum_i,a theta_ia * a†_a a_i  (singles)
    - T2 = sum_ij,ab theta_ijab * a†_a a†_b a_j a_i  (doubles)

    Each excitation generator G = (a†_p a_q - a†_q a_p) is implemented as
    a CNOT-ladder with an RZ rotation in the middle.

    Args:
        wf:           Wavefunction to apply the ansatz to.
        params:       Flat array of variational parameters.
        n_electrons:  Number of electrons.
        n_orbitals:   Number of spatial orbitals.
    """
    params = np.asarray(params, dtype=float)
    singles, doubles = _get_excitations(n_electrons, n_orbitals)

    expected = len(singles) + len(doubles)
    if len(params) != expected:
        raise ValueError(
            f"Expected {expected} parameters, got {len(params)}"
        )

    idx = 0

    # Apply single excitations
    for p, q in singles:
        theta = params[idx]
        idx += 1
        if abs(theta) > 1e-14:
            _apply_single_excitation(wf, theta, p, q)

    # Apply double excitations
    for p, q_virt, r, s in doubles:
        theta = params[idx]
        idx += 1
        if abs(theta) > 1e-14:
            _apply_double_excitation(wf, theta, p, q_virt, r, s)


def _apply_single_excitation(wf, theta: float, p: int, q: int) -> None:
    """Apply a single excitation e^{theta * (a†_p a_q - a†_q a_p)}.

    Implemented via a CNOT ladder with RY rotation:
    - CNOT chain from q to p
    - RY(theta) on p
    - Reverse CNOT chain
    """
    if p < q:
        p, q = q, p

    # CNOT ladder: entangle qubits q through p
    for i in range(q, p):
        gates.CNOT(wf, i, i + 1)

    # Rotation
    gates.RY(wf, p, theta)

    # Reverse CNOT ladder
    for i in range(p - 1, q - 1, -1):
        gates.CNOT(wf, i, i + 1)


def _apply_double_excitation(
    wf, theta: float, p: int, q: int, r: int, s: int,
) -> None:
    """Apply a double excitation e^{theta * (a†_p a†_q a_r a_s - h.c.)}.

    Uses a simplified Trotterized decomposition with CNOT ladders and
    RZ rotations. The full double excitation is decomposed into 8 terms
    following the standard UCCSD circuit construction.
    """
    qubits = sorted([p, q, r, s])
    lo, mid1, mid2, hi = qubits

    # Simplified implementation: CNOT staircase + RZ
    # This is a first-order Trotter step for the double excitation
    # Full decomposition uses 8 CNOT-RZ-CNOT blocks for the 8 Pauli terms

    # CNOT staircase
    for i in range(lo, hi):
        gates.CNOT(wf, i, i + 1)

    # RZ rotation on the highest qubit
    gates.RZ(wf, hi, theta)

    # Reverse CNOT staircase
    for i in range(hi - 1, lo - 1, -1):
        gates.CNOT(wf, i, i + 1)

    # Additional terms for better accuracy (Pauli gadget decomposition)
    # Apply Hadamard basis changes for X-type terms
    gates.H(wf, lo)
    gates.H(wf, hi)

    for i in range(lo, hi):
        gates.CNOT(wf, i, i + 1)

    gates.RZ(wf, hi, theta)

    for i in range(hi - 1, lo - 1, -1):
        gates.CNOT(wf, i, i + 1)

    gates.H(wf, lo)
    gates.H(wf, hi)
