# -*- coding: utf-8 -*-
# author: vinhpx
"""Clifford/Stabilizer simulator using the Aaronson-Gottesman CHP algorithm.

Efficiently simulates Clifford circuits (H, S, CNOT, X, Y, Z, measurements)
in O(n^2) time per gate, O(n^2) space.
"""
from __future__ import annotations

import numpy as np


class StabilizerState:
    """Stabilizer tableau representation of an n-qubit state.

    The tableau tracks 2n generators (n stabilizers + n destabilizers)
    using the binary symplectic representation.

    Args:
        n_qubits: Number of qubits. Initializes to |0...0>.
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n = n_qubits
        # Tableau: 2n rows x (2n + 1) columns
        # [0..n-1] = X bits, [n..2n-1] = Z bits, [2n] = phase (r)
        # Rows [0..n-1] = destabilizers, [n..2n-1] = stabilizers
        self.tableau = np.zeros((2 * n_qubits, 2 * n_qubits + 1), dtype=np.int8)
        for i in range(n_qubits):
            self.tableau[i, i] = 1                        # destabilizer: X_i
            self.tableau[n_qubits + i, n_qubits + i] = 1  # stabilizer: Z_i

    def _rowmult(self, h: int, i: int) -> None:
        """Multiply row h by row i (h <- h * i)."""
        n = self.n
        phase = self._row_mult_phase(h, i)
        self.tableau[h, 2 * n] = (self.tableau[h, 2 * n] + self.tableau[i, 2 * n] + phase) % 2
        for j in range(2 * n):
            self.tableau[h, j] ^= self.tableau[i, j]

    def _row_mult_phase(self, h: int, i: int) -> int:
        """Compute the phase contribution when multiplying rows h and i."""
        n = self.n
        total = 0
        for j in range(n):
            x_h = int(self.tableau[h, j])
            z_h = int(self.tableau[h, n + j])
            x_i = int(self.tableau[i, j])
            z_i = int(self.tableau[i, n + j])
            if x_i == 1 and z_i == 1:
                total += x_h - z_h
            elif x_i == 1 and z_i == 0:
                total += z_h * (2 * x_h - 1)
            elif x_i == 0 and z_i == 1:
                total += x_h * (1 - 2 * z_h)
        return ((total % 4) + 4) % 4 // 2

    # ---- Clifford gates ----

    def h(self, qubit: int) -> 'StabilizerState':
        """Apply Hadamard gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= (
                self.tableau[i, qubit] & self.tableau[i, n + qubit]
            )
            self.tableau[i, qubit], self.tableau[i, n + qubit] = (
                self.tableau[i, n + qubit], self.tableau[i, qubit]
            )
        return self

    def s(self, qubit: int) -> 'StabilizerState':
        """Apply S (phase) gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= (
                self.tableau[i, qubit] & self.tableau[i, n + qubit]
            )
            self.tableau[i, n + qubit] ^= self.tableau[i, qubit]
        return self

    def cnot(self, control: int, target: int) -> 'StabilizerState':
        """Apply CNOT gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= (
                self.tableau[i, control]
                & self.tableau[i, n + target]
                & (self.tableau[i, target] ^ self.tableau[i, n + control] ^ 1)
            )
            self.tableau[i, target] ^= self.tableau[i, control]
            self.tableau[i, n + control] ^= self.tableau[i, n + target]
        return self

    def x(self, qubit: int) -> 'StabilizerState':
        """Apply Pauli X gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= self.tableau[i, n + qubit]
        return self

    def y(self, qubit: int) -> 'StabilizerState':
        """Apply Pauli Y gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= (
                self.tableau[i, qubit] ^ self.tableau[i, n + qubit]
            )
        return self

    def z(self, qubit: int) -> 'StabilizerState':
        """Apply Pauli Z gate."""
        n = self.n
        for i in range(2 * n):
            self.tableau[i, 2 * n] ^= self.tableau[i, qubit]
        return self

    # ---- Measurement ----

    def measure(self, qubit: int) -> int:
        """Measure a qubit in the computational basis.

        Returns 0 or 1 and updates the tableau.
        """
        n = self.n
        p = None
        for i in range(n, 2 * n):
            if self.tableau[i, qubit] == 1:
                p = i
                break

        if p is not None:
            # Random outcome
            for i in range(2 * n):
                if i != p and self.tableau[i, qubit] == 1:
                    self._rowmult(i, p)
            self.tableau[p - n] = self.tableau[p].copy()
            self.tableau[p] = 0
            self.tableau[p, n + qubit] = 1
            outcome = int(np.random.randint(2))
            self.tableau[p, 2 * n] = outcome
            return outcome
        else:
            # Deterministic outcome
            scratch = np.zeros(2 * n + 1, dtype=np.int8)
            for i in range(n):
                if self.tableau[i, qubit] == 1:
                    phase = 0
                    for j in range(n):
                        x_s = int(scratch[j])
                        z_s = int(scratch[n + j])
                        x_i = int(self.tableau[i + n, j])
                        z_i = int(self.tableau[i + n, n + j])
                        if x_i == 1 and z_i == 1:
                            phase += x_s - z_s
                        elif x_i == 1 and z_i == 0:
                            phase += z_s * (2 * x_s - 1)
                        elif x_i == 0 and z_i == 1:
                            phase += x_s * (1 - 2 * z_s)
                    scratch[2 * n] = (scratch[2 * n] + self.tableau[i + n, 2 * n] + ((phase % 4 + 4) % 4 // 2)) % 2
                    for j in range(2 * n):
                        scratch[j] ^= self.tableau[i + n, j]
            return int(scratch[2 * n])

    def probabilities(self, n_samples: int = 1000) -> dict[str, float]:
        """Estimate measurement probabilities by sampling.

        Args:
            n_samples: Number of measurement samples.

        Returns:
            Dict mapping bitstrings to estimated probabilities.
        """
        counts: dict[str, int] = {}
        for _ in range(n_samples):
            # Create a fresh copy for each sample
            s_copy = StabilizerState(self.n)
            s_copy.tableau = self.tableau.copy()
            bits = []
            for q in range(self.n):
                bits.append(str(s_copy.measure(q)))
            bitstring = ''.join(bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return {k: v / n_samples for k, v in counts.items()}

    def __repr__(self) -> str:
        return f"StabilizerState(n_qubits={self.n})"
