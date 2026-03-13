# -*- coding: utf-8 -*-
# author: vinhpx
"""Pre-built parameterized ansatz circuits."""
from __future__ import annotations
import numpy as np
from qforge.gates import RY, RZ, CNOT


def hardware_efficient_ansatz(wf, params: np.ndarray, n_layers: int = 1) -> None:
    """Hardware-efficient ansatz: alternating RY blocks and CNOT ladders.

    Structure: ``(n_layers + 1)`` RY blocks separated by ``n_layers``
    nearest-neighbour CNOT ladders.

    Number of parameters: ``n_qubits * (n_layers + 1)``

    Args:
        wf:       Wavefunction (modified in-place).
        params:   1-D rotation-angle array.
        n_layers: Number of entangling layers.
    """
    n_qubits = len(wf.state[0])
    expected = n_qubits * (n_layers + 1)
    if len(params) != expected:
        raise ValueError(
            f"hardware_efficient_ansatz: expected {expected} params, got {len(params)}"
        )
    idx = 0
    for layer in range(n_layers + 1):
        for q in range(n_qubits):
            RY(wf, q, params[idx]); idx += 1
        if layer < n_layers:
            for q in range(n_qubits - 1):
                CNOT(wf, q, q + 1)


def strongly_entangling_ansatz(wf, params: np.ndarray, n_layers: int = 1) -> None:
    """Strongly entangling ansatz: RZ/RY/RZ rotations + long-range CNOT ring.

    Each layer applies RZ/RY/RZ to every qubit, then a CNOT ring with a
    stride that increments each layer to cover different qubit pairs.

    Number of parameters: ``n_layers * n_qubits * 3``

    Args:
        wf:       Wavefunction (modified in-place).
        params:   1-D rotation-angle array.
        n_layers: Number of entangling layers.
    """
    n_qubits = len(wf.state[0])
    expected = n_layers * n_qubits * 3
    if len(params) != expected:
        raise ValueError(
            f"strongly_entangling_ansatz: expected {expected} params, got {len(params)}"
        )
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            RZ(wf, q, params[idx]); idx += 1
            RY(wf, q, params[idx]); idx += 1
            RZ(wf, q, params[idx]); idx += 1
        stride = (layer % (n_qubits - 1)) + 1 if n_qubits > 1 else 0
        for q in range(n_qubits):
            target = (q + stride) % n_qubits
            if target != q:
                CNOT(wf, q, target)
