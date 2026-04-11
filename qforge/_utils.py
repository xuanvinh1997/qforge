# -*- coding: utf-8 -*-
# author: vinhpx
"""Internal shared utilities for Qforge."""
from __future__ import annotations


def _validate_qubit(n: int, qubit_num: int) -> None:
    if n >= qubit_num or n < 0:
        raise ValueError("Index is out of range")


def _validate_ctrl_target(control: int, target: int, qubit_num: int) -> None:
    if control >= qubit_num or control < 0 or target >= qubit_num or target < 0:
        raise ValueError("Index is out of range")
    if control == target:
        raise ValueError("Control qubit and target qubit must be distinct")


def _nq(wavefunction) -> int:
    """Get qubit count from wavefunction."""
    n = getattr(wavefunction, '_n_qubits', 0)
    if n:
        return n
    return len(wavefunction.state[0])


def _is_mps(wavefunction) -> bool:
    """Check if wavefunction uses MPS backend (C++ or Python tensors)."""
    return (hasattr(wavefunction, '_mps') or hasattr(wavefunction, '_tensors')) \
           and wavefunction._sv is None


def _is_dm(wavefunction) -> bool:
    """Check if wavefunction is a DensityMatrix."""
    return getattr(wavefunction, '_is_density_matrix', False)


# Visual circuit gate-type constants
_VIS_SINGLE = 1         # Single-qubit gate (H, X, Y, Z, S, T, XS)
_VIS_ROTATION = 1.5     # Parameterized single-qubit (RX, RY, RZ)
_VIS_WIRE = 2           # Vertical wire between qubits
_VIS_CTRL_ROT = 2.5     # Controlled rotation (CRX, CRY, CRZ)
_VIS_CTRL = 3           # Two-qubit controlled (CX, SWAP)
_VIS_CTRL_PHASE = 3.5   # Controlled phase (CP)
_VIS_DOUBLE_CTRL = 4    # Double-controlled (CCX)
_VIS_CTRL_SWAP = 5      # Controlled swap (CSWAP)
