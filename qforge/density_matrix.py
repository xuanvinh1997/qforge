# -*- coding: utf-8 -*-
# author: vinhpx
"""Density matrix simulator for qforge.

Provides the ``DensityMatrix`` class for simulating quantum circuits with
mixed states and noise channels via the Kraus operator formalism.
"""
from __future__ import annotations

import itertools
import numpy as np
from numpy.linalg import eigvalsh


class DensityMatrix:
    """Density matrix representation of a quantum state.

    Stores rho as a 2^n x 2^n complex numpy array. Compatible with the
    existing gate dispatch in ``qforge.gates`` by exposing ``._sv = None``,
    ``.state``, ``.visual``, and ``._is_density_matrix = True``.

    Args:
        n_qubits: Number of qubits.
    """

    _is_density_matrix: bool = True

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        dim = 2 ** n_qubits
        # Initialise to |0...0><0...0|
        self.rho = np.zeros((dim, dim), dtype=complex)
        self.rho[0, 0] = 1.0

        # Compatibility attributes for gate dispatch
        self._sv = None
        self.state = np.array(
            ["".join(seq) for seq in itertools.product("01", repeat=n_qubits)]
        )
        self.visual: list = []

    # ------------------------------------------------------------------
    # Gate / channel application
    # ------------------------------------------------------------------

    def apply_gate(self, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply a unitary gate: rho -> U rho U†.

        Args:
            matrix: Row-major flat array (length 4 for 1q, 16 for 2q)
                    or 2D matrix.
            qubits: List of qubit indices the gate acts on.
        """
        matrix = np.asarray(matrix, dtype=complex)
        if len(qubits) == 1:
            U = self._build_full_unitary_1q(matrix, qubits[0])
        elif len(qubits) == 2:
            U = self._build_full_unitary_2q(matrix, qubits[0], qubits[1])
        else:
            raise NotImplementedError("Only 1- and 2-qubit gates are supported")
        self.rho = U @ self.rho @ U.conj().T

    def apply_channel(self, kraus_ops: list[np.ndarray], qubits: list[int]) -> None:
        """Apply a quantum channel: rho -> sum_k  K_k rho K_k†.

        Each Kraus operator is a 2x2 (single-qubit) or 4x4 (two-qubit) matrix.

        Args:
            kraus_ops: List of Kraus operator matrices.
            qubits:    Qubit indices the channel acts on.
        """
        new_rho = np.zeros_like(self.rho)
        for K in kraus_ops:
            K = np.asarray(K, dtype=complex)
            if len(qubits) == 1:
                K_full = self._build_full_unitary_1q(K, qubits[0])
            elif len(qubits) == 2:
                K_full = self._build_full_unitary_2q(K, qubits[0], qubits[1])
            else:
                raise NotImplementedError("Only 1- and 2-qubit channels supported")
            new_rho += K_full @ self.rho @ K_full.conj().T
        self.rho = new_rho

    # ------------------------------------------------------------------
    # Internal: build full-space operators via Kronecker products
    # ------------------------------------------------------------------

    def _build_full_unitary_1q(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Embed a single-qubit gate into the full 2^n space."""
        gate = gate.reshape(2, 2)
        n = self.n_qubits
        # I^{qubit} ⊗ gate ⊗ I^{n - qubit - 1}
        ops = []
        for i in range(n):
            ops.append(gate if i == qubit else np.eye(2, dtype=complex))
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _build_full_unitary_2q(
        self, gate: np.ndarray, q0: int, q1: int
    ) -> np.ndarray:
        """Embed a two-qubit gate (acting on q0, q1) into the full 2^n space.

        The gate matrix is in the computational basis ordered by (q0, q1):
        |00>, |01>, |10>, |11> where q0 is the more-significant bit.
        """
        gate = gate.reshape(4, 4)
        n = self.n_qubits
        dim = 2 ** n

        # Build via index manipulation
        full = np.eye(dim, dtype=complex)
        bit0 = n - q0 - 1
        bit1 = n - q1 - 1

        result = np.zeros((dim, dim), dtype=complex)
        for idx in range(dim):
            # Decompose idx into (q0_bit, q1_bit, rest)
            b0 = (idx >> bit0) & 1
            b1 = (idx >> bit1) & 1
            col_in = b0 * 2 + b1  # input sub-index

            for out_sub in range(4):
                coeff = gate[out_sub, col_in]
                if coeff == 0:
                    continue
                ob0 = (out_sub >> 1) & 1
                ob1 = out_sub & 1
                # Construct output index: replace bits at q0, q1
                out_idx = idx
                # Clear bits
                out_idx = out_idx & ~(1 << bit0) & ~(1 << bit1)
                # Set new bits
                out_idx = out_idx | (ob0 << bit0) | (ob1 << bit1)
                result[out_idx, idx] += coeff

        return result

    # ------------------------------------------------------------------
    # Properties and measurements
    # ------------------------------------------------------------------

    @property
    def amplitude(self) -> np.ndarray:
        """Extract state-vector amplitudes (only valid for pure states).

        Raises:
            ValueError: If the state is mixed.
        """
        if not self.is_pure():
            raise ValueError(
                "Cannot extract amplitudes from a mixed state density matrix"
            )
        # For a pure state rho = |psi><psi|, the eigenvector with eigenvalue 1
        # gives |psi> (up to global phase).
        eigenvalues, eigenvectors = np.linalg.eigh(self.rho)
        # Pick eigenvector corresponding to eigenvalue ~1
        idx = np.argmax(eigenvalues.real)
        psi = eigenvectors[:, idx]
        # Fix global phase so that the first non-zero element is real-positive
        for v in psi:
            if abs(v) > 1e-10:
                psi = psi * np.exp(-1j * np.angle(v))
                break
        return psi

    def probabilities(self) -> np.ndarray:
        """Return measurement probabilities (diagonal of rho)."""
        return np.real(np.diag(self.rho))

    def trace(self) -> float:
        """Return Tr(rho)."""
        return float(np.real(np.trace(self.rho)))

    def purity(self) -> float:
        """Return Tr(rho^2)."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def fidelity(self, other: DensityMatrix) -> float:
        """Compute fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2.

        Uses the simplified formula via eigendecomposition.
        """
        # sqrt(rho)
        eigvals, eigvecs = np.linalg.eigh(self.rho)
        eigvals = np.maximum(eigvals, 0)
        sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

        # M = sqrt(rho) @ sigma @ sqrt(rho)
        M = sqrt_rho @ other.rho @ sqrt_rho
        eigvals_M = np.linalg.eigvalsh(M)
        eigvals_M = np.maximum(eigvals_M, 0)
        return float(np.real(np.sum(np.sqrt(eigvals_M)) ** 2))

    def partial_trace(self, keep_qubits: list[int]) -> DensityMatrix:
        """Compute the reduced density matrix by tracing out all qubits
        not in ``keep_qubits``.

        Args:
            keep_qubits: List of qubit indices to keep.

        Returns:
            A new ``DensityMatrix`` for the subsystem.
        """
        n = self.n_qubits
        keep = sorted(keep_qubits)
        trace_out = sorted(set(range(n)) - set(keep))

        n_keep = len(keep)
        dim_keep = 2 ** n_keep
        result = np.zeros((dim_keep, dim_keep), dtype=complex)

        dim = 2 ** n
        for i in range(dim):
            for j in range(dim):
                # Check if traced-out qubits have same bits in i and j
                match = True
                for q in trace_out:
                    bit_pos = n - q - 1
                    if ((i >> bit_pos) & 1) != ((j >> bit_pos) & 1):
                        match = False
                        break
                if not match:
                    continue

                # Extract kept-qubit bits
                ki = 0
                kj = 0
                for idx_k, q in enumerate(keep):
                    bit_pos = n - q - 1
                    bi = (i >> bit_pos) & 1
                    bj = (j >> bit_pos) & 1
                    out_bit = n_keep - idx_k - 1
                    ki |= (bi << out_bit)
                    kj |= (bj << out_bit)

                result[ki, kj] += self.rho[i, j]

        dm = DensityMatrix(n_keep)
        dm.rho = result
        return dm

    def von_neumann_entropy(self, base: int = 2) -> float:
        """Compute the von Neumann entropy S = -Tr(rho log rho).

        Args:
            base: Logarithm base (default 2, giving entropy in bits).
        """
        eigvals = eigvalsh(self.rho)
        # Filter out zero/negative eigenvalues (numerical noise)
        eigvals = eigvals[eigvals > 1e-15]
        if base == 2:
            return float(-np.sum(eigvals * np.log2(eigvals)))
        elif base == np.e or base == 'e':
            return float(-np.sum(eigvals * np.log(eigvals)))
        else:
            return float(-np.sum(eigvals * np.log(eigvals) / np.log(base)))

    def is_pure(self) -> bool:
        """Check whether the state is pure (Tr(rho^2) ≈ 1)."""
        return abs(self.purity() - 1.0) < 1e-8

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_wavefunction(cls, wf) -> DensityMatrix:
        """Create a density matrix from a Wavefunction: rho = |psi><psi|.

        Args:
            wf: A ``Wavefunction`` instance with ``.amplitude`` and ``.state``.
        """
        n_qubits = len(wf.state[0])
        dm = cls(n_qubits)
        psi = np.asarray(wf.amplitude, dtype=complex)
        dm.rho = np.outer(psi, psi.conj())
        return dm

    @classmethod
    def from_state_vector(cls, amplitudes: np.ndarray, n_qubits: int) -> DensityMatrix:
        """Create a density matrix from a state vector.

        Args:
            amplitudes: Complex amplitude vector of length 2^n_qubits.
            n_qubits:   Number of qubits.
        """
        amplitudes = np.asarray(amplitudes, dtype=complex)
        if len(amplitudes) != 2 ** n_qubits:
            raise ValueError(
                f"Expected amplitude vector of length {2**n_qubits}, "
                f"got {len(amplitudes)}"
            )
        dm = cls(n_qubits)
        dm.rho = np.outer(amplitudes, amplitudes.conj())
        return dm

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(n_qubits={self.n_qubits}, "
            f"purity={self.purity():.6f}, trace={self.trace():.6f})"
        )
