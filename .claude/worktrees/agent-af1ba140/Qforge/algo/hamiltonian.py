# -*- coding: utf-8 -*-
# author: vinhpx
"""Hamiltonian representation and Pauli-string expectation values."""
from __future__ import annotations
import numpy as np


def _pauli_string_expectation_from_amp(
    amp: np.ndarray, n_qubits: int, ops: list[tuple[str, int]]
) -> float:
    """Compute <ψ|P₀⊗P₁⊗…|ψ> from a pre-fetched amplitude array.

    Separating the array fetch from the computation lets callers amortize a
    single ``sync_to_host`` across many Pauli-string evaluations.

    Args:
        amp:      Complex amplitude array of length 2**n_qubits.
        n_qubits: Number of qubits.
        ops:      ``[(pauli_type, qubit_index), …]``.
                  An empty list represents the identity → returns 1.0.

    Returns:
        Real expectation value.
    """
    if not ops:
        return 1.0

    dim = len(amp)
    i_arr = np.arange(dim, dtype=np.int64)
    j_arr = i_arr.copy()
    phase_arr = np.ones(dim, dtype=complex)

    for pauli, q in ops:
        bits = (i_arr >> (n_qubits - q - 1)) & 1
        mask = np.int64(1 << (n_qubits - q - 1))
        if pauli == 'Z':
            phase_arr[bits == 1] *= -1.0
        elif pauli == 'X':
            j_arr ^= mask
        elif pauli == 'Y':
            j_arr ^= mask
            phase_arr[bits == 0] *= -1j
            phase_arr[bits == 1] *= +1j
        # 'I' → no change

    return float(np.real(np.sum(np.conj(amp) * phase_arr * amp[j_arr])))


def _pauli_string_expectation(wf, ops: list[tuple[str, int]]) -> float:
    """Compute <ψ|P₀⊗P₁⊗…|ψ> analytically from the amplitude vector.

    Args:
        wf:  Wavefunction with ``.amplitude`` (complex ndarray) and ``.state``.
        ops: ``[(pauli_type, qubit_index), …]``.
             An empty list represents the identity → returns 1.0.

    Returns:
        Real expectation value.
    """
    return _pauli_string_expectation_from_amp(
        wf.amplitude, len(wf.state[0]), ops
    )


class Hamiltonian:
    """Hamiltonian as a weighted sum of Pauli strings.

    Args:
        coeffs: List of real coefficients.
        terms:  List of Pauli-string specs; each spec is a list of
                ``(pauli_type, qubit_index)`` pairs.
                An empty list ``[]`` represents the identity.

    Example::

        H = Hamiltonian(
            coeffs=[-1.0523, 0.3979, -0.3979, -0.0112, 0.1809],
            terms=[
                [],
                [('Z', 0)],
                [('Z', 1)],
                [('Z', 0), ('Z', 1)],
                [('X', 0), ('X', 1)],
            ],
        )
        energy = H.expectation(wf)
    """

    def __init__(self, coeffs: list[float], terms: list[list[tuple[str, int]]]):
        if len(coeffs) != len(terms):
            raise ValueError("coeffs and terms must have the same length")
        self.coeffs = list(coeffs)
        self.terms = list(terms)

    def expectation(self, wf) -> float:
        """Compute <ψ|H|ψ>.

        Fetches the amplitude array once (triggering a single GPU→host sync
        for CUDA/Metal backends) and evaluates all Pauli-string terms on the
        cached array.
        """
        amp = wf.amplitude
        n_qubits = len(wf.state[0])
        return sum(
            c * _pauli_string_expectation_from_amp(amp, n_qubits, t)
            for c, t in zip(self.coeffs, self.terms)
        )

    def __repr__(self) -> str:
        parts = []
        for c, ops in zip(self.coeffs, self.terms):
            op_str = " ".join(f"{p}{q}" for p, q in ops) if ops else "I"
            parts.append(f"{c:.4g}*({op_str})")
        return "Hamiltonian(" + " + ".join(parts) + ")"
