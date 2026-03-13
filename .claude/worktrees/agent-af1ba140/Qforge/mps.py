# -*- coding: utf-8 -*-
# author: vinhpx
"""
Matrix Product State (MPS) simulation backend for qforge.

Provides an alternative to the Wavefunction/StateVector backend for
low-entanglement circuits with 50+ qubits. The MatrixProductState class
mirrors the Wavefunction interface so that Qgates functions can be
called against it with the same API.

Usage::

    from Qforge.mps import MatrixProductState
    from Qforge import gates

    psi = MatrixProductState(n_qubits=50, max_bond_dim=64)
    gates.H(psi, 0)
    for i in range(49):
        gates.CNOT(psi, i, i + 1)
    print(psi.bond_dimensions())       # [2, 2, ..., 2]
    print(psi.entanglement_entropy(0)) # ~ 1 bit
"""
from __future__ import annotations
import numpy as np
import itertools
from typing import List, Optional

try:
    from Qforge._qforge_mps import MPS as _MPS_core
    _HAS_MPS_CPP = True
except ImportError:
    _HAS_MPS_CPP = False


class MatrixProductState:
    """Quantum state represented as a Matrix Product State.

    Args:
        n_qubits:     Number of qubits (supports 2–1000+).
        max_bond_dim: Maximum bond dimension chi. Controls accuracy vs memory.
        eps:          SVD truncation threshold (relative to largest singular value).
        backend:      'auto', 'cpu' (C++), or 'python' (numpy fallback).

    The class exposes the same attributes that gates/measurement use:
        .state       — placeholder array (length n_qubits, needed for _nq())
        .visual      — circuit log list
        ._sv         — always None (signals gates to use _mps path instead)
        ._mps        — C++ MPS object (or None if Python backend)
        ._tensors    — Python tensor list (if Python backend)
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond_dim: int = 32,
        eps: float = 1e-10,
        backend: str = 'auto',
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.max_bond_dim = max_bond_dim
        self.eps = eps
        self.visual: list = []

        # Compatibility with gates._nq() which does len(wavefunction.state[0])
        self.state = np.array(['0' * n_qubits])

        self._mps: Optional[_MPS_core] = None
        self._tensors: Optional[list] = None

        if backend == 'auto':
            backend = 'cpu' if _HAS_MPS_CPP else 'python'

        if backend == 'cpu' and _HAS_MPS_CPP:
            self._mps = _MPS_core(n_qubits, max_bond_dim)
        else:
            self._tensors = _init_product_state(n_qubits)

    # ----------------------------------------------------------------
    # Wavefunction-compatibility interface
    # ----------------------------------------------------------------

    @property
    def _sv(self):
        """Always None — signals gates to use the _mps dispatch branch."""
        return None

    @property
    def amplitude(self) -> np.ndarray:
        """Full state vector amplitude (exponential cost — use sparingly)."""
        if self._mps is not None:
            return self._mps.to_statevector()
        return _contract_py(self._tensors)

    @amplitude.setter
    def amplitude(self, arr: np.ndarray) -> None:
        """Set from full amplitude array via SVD decomposition."""
        arr = np.ascontiguousarray(arr, dtype=complex)
        if self._mps is not None:
            self._mps.from_statevector(arr, self.eps)
        else:
            self._tensors = _svd_decompose(arr, self.n_qubits,
                                           self.max_bond_dim, self.eps)

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitude) ** 2

    # ----------------------------------------------------------------
    # MPS-specific properties
    # ----------------------------------------------------------------

    def bond_dimensions(self) -> List[int]:
        """Bond dimensions at each of the N-1 bonds."""
        if self._mps is not None:
            return list(self._mps.bond_dimensions())
        return [int(t.shape[2]) for t in self._tensors[:-1]]

    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entropy at bond (bond, bond+1) in bits."""
        if self._mps is not None:
            return float(self._mps.entanglement_entropy(bond))
        return _entropy_py(self._tensors, bond)

    def max_entanglement(self) -> float:
        """Maximum entanglement entropy across all bonds."""
        if self._mps is not None:
            return float(self._mps.max_entanglement_entropy())
        n = self.n_qubits
        return max(_entropy_py(self._tensors, b) for b in range(n - 1))

    def norm(self) -> float:
        """Norm of the MPS state (should be 1.0 for normalized states)."""
        if self._mps is not None:
            return float(self._mps.norm())
        return float(np.linalg.norm(self.amplitude))

    # ----------------------------------------------------------------
    # Conversion helpers
    # ----------------------------------------------------------------

    def to_wavefunction(self):
        """Contract MPS → Wavefunction object. Exponential memory cost."""
        from Qforge.wavefunction import Wavefunction
        states = np.array(
            [''.join(s) for s in itertools.product('01', repeat=self.n_qubits)]
        )
        return Wavefunction(states, self.amplitude)

    @classmethod
    def from_wavefunction(cls, wf, max_bond_dim: int = 32,
                          eps: float = 1e-10) -> 'MatrixProductState':
        """Create MPS from Wavefunction via sequential SVD decomposition."""
        n = len(wf.state[0])
        obj = cls(n, max_bond_dim=max_bond_dim, eps=eps)
        amp = np.ascontiguousarray(wf.amplitude, dtype=complex)
        if obj._mps is not None:
            obj._mps.from_statevector(amp, eps)
        else:
            obj._tensors = _svd_decompose(amp, n, max_bond_dim, eps)
        return obj

    def __repr__(self) -> str:
        chi = (self._mps.max_current_chi() if self._mps is not None
               else max(t.shape[2] for t in self._tensors))
        backend = 'C++' if self._mps is not None else 'Python'
        return (f"MatrixProductState(n_qubits={self.n_qubits}, "
                f"max_bond_dim={self.max_bond_dim}, "
                f"current_chi={chi}, backend={backend})")


# ================================================================
# Python fallback: list of rank-3 numpy tensors [chi_l, d, chi_r]
# ================================================================

def _init_product_state(n: int) -> list:
    """Initialize |00...0> as list of tensors, each shape [1, 2, 1]."""
    tensors = []
    for _ in range(n):
        t = np.zeros((1, 2, 1), dtype=complex)
        t[0, 0, 0] = 1.0  # |0>
        tensors.append(t)
    return tensors


def _contract_py(tensors: list) -> np.ndarray:
    """Contract tensor list to full amplitude vector (Python fallback)."""
    # Start: [d0, chi_1]
    result = tensors[0][0, :, :]
    for A in tensors[1:]:
        # result: [states, chi_prev]  A: [chi_prev, d, chi_next]
        new_dim = result.shape[0] * A.shape[1]
        tmp = np.tensordot(result, A, axes=([1], [0]))  # [states, d, chi_next]
        result = tmp.reshape(new_dim, A.shape[2])
    return result[:, 0]  # chi_final = 1


def _svd_decompose(amp: np.ndarray, n: int,
                   max_chi: int, eps: float) -> list:
    """Decompose amplitude vector into MPS via sequential SVD."""
    tensors = []
    psi = amp.reshape(1, -1)
    chi_l = 1
    for i in range(n - 1):
        d = 2
        rows = chi_l * d
        cols = psi.shape[1] // d
        M = psi.reshape(rows, cols)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        threshold = eps * S[0] if S[0] > 0 else eps
        keep = max(1, min(max_chi, int(np.sum(S > threshold))))
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep, :]
        tensors.append(U.reshape(chi_l, d, keep))
        chi_l = keep
        psi = (np.diag(S) @ Vt)
    tensors.append(psi.reshape(chi_l, 2, 1))
    return tensors


def _entropy_py(tensors: list, bond: int) -> float:
    """Von Neumann entropy at bond via contraction and SVD (Python fallback)."""
    n = len(tensors)
    # Left-contract to [2^(bond+1), chi]
    result = tensors[0][0, :, :]
    chi = result.shape[1]
    left_states = 2
    for i in range(1, bond + 1):
        A = tensors[i]
        new_chi = A.shape[2]
        tmp = np.tensordot(result, A, axes=([1], [0]))  # [states, d, new_chi]
        result = tmp.reshape(left_states * 2, new_chi)
        left_states *= 2
        chi = new_chi

    # Right-contract to [chi, 2^(N-bond-1)]
    right = tensors[-1][:, :, 0]  # [chi_last, d]
    right_states = 2
    for i in range(n - 2, bond, -1):
        A = tensors[i]
        new_chi = A.shape[0]
        # right: [chi, states], A: [new_chi, d, chi]
        tmp = np.tensordot(A, right, axes=([2], [0]))  # [new_chi, d, states]
        right = tmp.reshape(new_chi, right_states * 2)
        right_states *= 2

    # theta = result @ right: [left_states, right_states]
    theta = result @ right
    sv = np.linalg.svd(theta, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-14:
        return 0.0
    sv2 /= total
    sv2 = sv2[sv2 > 1e-14]
    return float(-np.sum(sv2 * np.log2(sv2)))
