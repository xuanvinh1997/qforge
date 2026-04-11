# -*- coding: utf-8 -*-
# author: vinhpx
"""Vectorized NumPy gate kernels for the small-qubit fast path.

These kernels operate in-place on a flat ``(2**n,)`` amplitude array using
pure NumPy slicing. No allocation per gate, no Python-to-C++ FFI — ideal for
circuits with few qubits where the C++ dispatch overhead dominates the
actual compute cost.

Qubit indexing follows the rest of Qforge: qubit 0 is the most-significant
bit of the basis state label.
"""
from __future__ import annotations
import numpy as np


def apply_single_inplace(amp: np.ndarray, n_qubits: int, target: int,
                         m00, m01, m10, m11) -> None:
    """Apply a 2x2 unitary [[m00, m01], [m10, m11]] to ``target`` qubit, in place.

    Uses a reshape-based view so updates write directly into ``amp``.
    """
    stride = 1 << (n_qubits - target - 1)
    block = stride << 1
    view = amp.reshape(-1, block)
    a0 = view[:, :stride].copy()
    a1 = view[:, stride:].copy()
    view[:, :stride] = m00 * a0 + m01 * a1
    view[:, stride:] = m10 * a0 + m11 * a1


def apply_controlled_inplace(amp: np.ndarray, n_qubits: int,
                             control: int, target: int,
                             m00, m01, m10, m11) -> None:
    """Apply a controlled 2x2 unitary on ``target`` when ``control`` is |1>."""
    dim = amp.size
    cmask = 1 << (n_qubits - control - 1)
    tmask = 1 << (n_qubits - target - 1)
    idx = np.arange(dim)
    sel = ((idx & cmask) != 0) & ((idx & tmask) == 0)
    i0 = np.nonzero(sel)[0]
    i1 = i0 | tmask
    a0 = amp[i0].copy()
    a1 = amp[i1].copy()
    amp[i0] = m00 * a0 + m01 * a1
    amp[i1] = m10 * a0 + m11 * a1


def apply_cnot_inplace(amp: np.ndarray, n_qubits: int,
                       control: int, target: int) -> None:
    """Specialization of CNOT — pure index permutation, no multiplies."""
    dim = amp.size
    cmask = 1 << (n_qubits - control - 1)
    tmask = 1 << (n_qubits - target - 1)
    idx = np.arange(dim)
    sel = ((idx & cmask) != 0) & ((idx & tmask) == 0)
    i0 = np.nonzero(sel)[0]
    i1 = i0 | tmask
    tmp = amp[i0].copy()
    amp[i0] = amp[i1]
    amp[i1] = tmp


def apply_swap_inplace(amp: np.ndarray, n_qubits: int,
                       target_1: int, target_2: int) -> None:
    """SWAP two qubits — pure permutation."""
    if target_1 == target_2:
        return
    dim = amp.size
    m1 = 1 << (n_qubits - target_1 - 1)
    m2 = 1 << (n_qubits - target_2 - 1)
    idx = np.arange(dim)
    bit1 = (idx & m1) != 0
    bit2 = (idx & m2) != 0
    sel = bit1 & ~bit2  # differ with qubit1=1, qubit2=0
    i0 = np.nonzero(sel)[0]
    i1 = (i0 ^ m1) | m2  # flip both bits
    tmp = amp[i0].copy()
    amp[i0] = amp[i1]
    amp[i1] = tmp


def apply_ccnot_inplace(amp: np.ndarray, n_qubits: int,
                        c1: int, c2: int, target: int) -> None:
    """Toffoli — flip target when both controls are |1>."""
    dim = amp.size
    m1 = 1 << (n_qubits - c1 - 1)
    m2 = 1 << (n_qubits - c2 - 1)
    mt = 1 << (n_qubits - target - 1)
    idx = np.arange(dim)
    sel = ((idx & m1) != 0) & ((idx & m2) != 0) & ((idx & mt) == 0)
    i0 = np.nonzero(sel)[0]
    i1 = i0 | mt
    tmp = amp[i0].copy()
    amp[i0] = amp[i1]
    amp[i1] = tmp
