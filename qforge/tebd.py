# -*- coding: utf-8 -*-
# author: vinhpx
"""
TEBD (Time-Evolving Block Decimation) for MPS time evolution.

Implements Trotter-Suzuki decomposed time evolution for nearest-neighbor
Hamiltonians. Operates on MatrixProductState objects in-place.

Usage::

    from qforge.mps import MatrixProductState
    from qforge.tebd import TEBD

    psi = MatrixProductState(n_qubits=20, max_bond_dim=64)
    # Transverse-field Ising: H = -J*ZZ - h*X
    tebd = TEBD.ising(psi, J=1.0, h=0.5, dt=0.01)
    tebd.evolve(total_time=1.0)
    print(tebd.energy())
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Union
from scipy.linalg import expm

from qforge.mps import MatrixProductState

try:
    from qforge._qforge_mps import MPS as _MPS_core
    _HAS_MPS_CPP = True
except ImportError:
    _HAS_MPS_CPP = False

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class TEBD:
    """Time-evolving block decimation for MPS.

    Args:
        psi:              MatrixProductState to evolve (modified in-place).
        hamiltonian_bonds: List of N-1 two-site Hamiltonians h_{i,i+1},
                          each a 4x4 complex matrix.
        dt:               Time step.
        order:            Trotter order (1, 2, or 4).
        imaginary:        If True, evolve in imaginary time (exp(-dt*H))
                          for ground-state finding.
    """

    def __init__(
        self,
        psi: MatrixProductState,
        hamiltonian_bonds: List[np.ndarray],
        dt: float = 0.01,
        order: int = 2,
        imaginary: bool = False,
    ) -> None:
        self.psi = psi
        self.dt = dt
        self.order = order
        self.imaginary = imaginary
        n = psi.n_qubits

        if len(hamiltonian_bonds) != n - 1:
            raise ValueError(
                f"Need {n - 1} bond Hamiltonians, got {len(hamiltonian_bonds)}")
        self._h_bonds = [np.asarray(h, dtype=complex).reshape(4, 4)
                         for h in hamiltonian_bonds]

        # Precompute time-evolution gates
        self._gates = {}
        self._precompute_gates(dt)

    def _precompute_gates(self, dt: float) -> None:
        """Compute exp(-i*dt*h) for each bond (or exp(-dt*h) for imaginary time)."""
        factor = -dt if self.imaginary else -1j * dt

        if self.order == 1:
            # First-order Trotter: e^{-i dt H_even} * e^{-i dt H_odd}
            self._gates['full'] = [
                expm(factor * h) for h in self._h_bonds
            ]
        elif self.order == 2:
            # Second-order: e^{-i dt/2 H_even} * e^{-i dt H_odd} * e^{-i dt/2 H_even}
            self._gates['half'] = [
                expm(0.5 * factor * h) for h in self._h_bonds
            ]
            self._gates['full'] = [
                expm(factor * h) for h in self._h_bonds
            ]
        elif self.order == 4:
            # Fourth-order Yoshida decomposition
            # p = 1/(2 - 2^{1/3})
            p = 1.0 / (2.0 - 2.0**(1.0/3.0))
            steps = [p / 2, p, (1 - 2 * p) / 2, 1 - 2 * p, (1 - 2 * p) / 2, p, p / 2]
            self._gates['yoshida'] = []
            for s in steps:
                self._gates['yoshida'].append([
                    expm(s * factor * h) for h in self._h_bonds
                ])
        else:
            raise ValueError(f"Trotter order must be 1, 2, or 4, got {order}")

    def step(self, n_steps: int = 1) -> None:
        """Perform n_steps time-evolution steps."""
        for _ in range(n_steps):
            if self.order == 1:
                self._step_first_order()
            elif self.order == 2:
                self._step_second_order()
            elif self.order == 4:
                self._step_fourth_order()

            if self.imaginary:
                self._normalize()

    def evolve(self, total_time: float) -> None:
        """Evolve for total_time (may adjust last step)."""
        n_steps = int(round(total_time / self.dt))
        self.step(n_steps)

    def _apply_even_odd(self, gates: List[np.ndarray], even: bool) -> None:
        """Apply gates to even or odd bonds."""
        n = self.psi.n_qubits
        start = 0 if even else 1
        for bond in range(start, n - 1, 2):
            gate = gates[bond]
            self._apply_two_site_gate(bond, gate)

    def _apply_two_site_gate(self, bond: int, gate: np.ndarray) -> None:
        """Apply a 4x4 gate to bond (bond, bond+1)."""
        psi = self.psi
        gate_flat = gate.reshape(16)

        if psi._mps is not None:
            psi._mps.apply_two_qubit_gate(
                bond, gate_flat, psi.max_bond_dim, psi.eps)
        elif psi._tensors is not None:
            self._apply_gate_py(bond, gate)

    def _apply_gate_py(self, bond: int, gate: np.ndarray) -> None:
        """Python fallback: apply gate via SVD split."""
        tensors = self.psi._tensors
        A = tensors[bond]       # [chi_l, d, chi_m]
        B = tensors[bond + 1]   # [chi_m, d, chi_r]
        chi_l = A.shape[0]
        chi_r = B.shape[2]
        d = 2

        # Contract: theta[cl, s0, s1, cr] = A[cl, s0, m] * B[m, s1, cr]
        theta = np.einsum('asm,mtc->astc', A, B)
        # Apply gate: theta'[cl, s0', s1', cr] = gate[s0',s1',s0,s1] * theta[cl,s0,s1,cr]
        theta = np.einsum('ijkl,akle->aije',
                          gate.reshape(d, d, d, d), theta)

        # SVD split
        M = theta.reshape(chi_l * d, d * chi_r)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        max_chi = self.psi.max_bond_dim
        eps = self.psi.eps
        threshold = eps * S[0] if S[0] > 0 else eps
        keep = max(1, min(max_chi, int(np.sum(S > threshold))))
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep, :]

        tensors[bond] = U.reshape(chi_l, d, keep)
        tensors[bond + 1] = (np.diag(S) @ Vt).reshape(keep, d, chi_r)

    def _step_first_order(self) -> None:
        gates = self._gates['full']
        self._apply_even_odd(gates, even=True)
        self._apply_even_odd(gates, even=False)

    def _step_second_order(self) -> None:
        half = self._gates['half']
        full = self._gates['full']
        self._apply_even_odd(half, even=True)
        self._apply_even_odd(full, even=False)
        self._apply_even_odd(half, even=True)

    def _step_fourth_order(self) -> None:
        layers = self._gates['yoshida']
        # Yoshida: 7 layers alternating even/odd
        parities = [True, False, True, False, True, False, True]
        for gates, even in zip(layers, parities):
            self._apply_even_odd(gates, even=even)

    def _normalize(self) -> None:
        """Normalize MPS (needed after imaginary time evolution)."""
        psi = self.psi
        if psi._mps is not None:
            nrm = psi.norm()
            if nrm > 1e-14:
                # Scale amplitude by 1/norm via contraction
                amp = psi._mps.to_statevector()
                amp /= nrm
                psi._mps.from_statevector(amp, psi.eps)
        elif psi._tensors is not None:
            from qforge.mps import _contract_py
            amp = _contract_py(psi._tensors)
            nrm = np.linalg.norm(amp)
            if nrm > 1e-14:
                # Scale last tensor
                psi._tensors[-1] /= nrm

    def energy(self) -> float:
        """Compute <psi|H|psi> as sum of bond energies."""
        e = 0.0
        psi = self.psi
        for bond, h in enumerate(self._h_bonds):
            e += self._bond_energy(bond, h)
        return float(e)

    def _bond_energy(self, bond: int, h: np.ndarray) -> complex:
        """Compute <psi|h_{bond,bond+1}|psi>."""
        psi = self.psi
        d = 2
        h4 = h.reshape(d, d, d, d)

        if psi._mps is not None:
            # Use two-site expectation via operator decomposition
            # For general h, decompose as sum of tensor products
            # h = sum_k alpha_k * A_k x B_k via SVD
            h_mat = h.reshape(d * d, d * d)
            U, S, Vt = np.linalg.svd(h_mat)
            result = 0.0
            for k in range(len(S)):
                if S[k] < 1e-14:
                    break
                op_a = (np.sqrt(S[k]) * U[:, k]).reshape(d, d)
                op_b = (np.sqrt(S[k]) * Vt[k, :]).reshape(d, d)
                result += psi._mps.two_site_expectation(
                    bond, bond + 1,
                    op_a.flatten(), op_b.flatten()
                ).real
            return result
        else:
            tensors = psi._tensors
            A = tensors[bond]
            B = tensors[bond + 1]
            n = psi.n_qubits

            # Build left contraction
            L = np.ones((1, 1), dtype=complex)
            for i in range(bond):
                T = tensors[i]
                L = np.einsum('ab,asc,bsd->cd', L, np.conj(T), T)

            # Contract with h
            val = np.einsum('ab,asc,cte,stpq,bpf,fqg,gh->',
                            L, np.conj(A), np.conj(B),
                            h4, A, B,
                            np.ones((B.shape[2], B.shape[2]) if bond + 2 >= n
                                    else (1, 1), dtype=complex),
                            optimize=True)

            # Build right contraction
            R = np.ones((1, 1), dtype=complex) if bond + 2 >= n else None
            if R is None:
                R = np.ones((1, 1), dtype=complex)
                for i in range(n - 1, bond + 1, -1):
                    T = tensors[i]
                    R = np.einsum('ab,csa,csb->cd',
                                  R, np.conj(T).transpose(2, 1, 0),
                                  T.transpose(2, 1, 0))
                    R = R.T

                val = np.einsum('ab,asc,cte,stpq,bpf,fqg,gh->',
                                L, np.conj(A), np.conj(B),
                                h4, A, B, R,
                                optimize=True)

            return val

    # ----------------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def ising(cls, psi: MatrixProductState, J: float = 1.0, h: float = 0.5,
              dt: float = 0.01, **kwargs) -> 'TEBD':
        """Transverse-field Ising: H = -J * sum ZZ - h * sum X."""
        n = psi.n_qubits
        bonds = []
        for i in range(n - 1):
            h_bond = -J * np.kron(_Z, _Z)
            # Add on-site terms split equally between bonds
            if i == 0:
                h_bond += -h * np.kron(_X, _I) - 0.5 * h * np.kron(_I, _X)
            elif i == n - 2:
                h_bond += -0.5 * h * np.kron(_X, _I) - h * np.kron(_I, _X)
            else:
                h_bond += -0.5 * h * np.kron(_X, _I) - 0.5 * h * np.kron(_I, _X)
        # Fix: split on-site terms evenly across bonds touching each site
            bonds.append(h_bond)
        return cls(psi, bonds, dt=dt, **kwargs)

    @classmethod
    def heisenberg(cls, psi: MatrixProductState, J: float = 1.0,
                   dt: float = 0.01, **kwargs) -> 'TEBD':
        """Heisenberg: H = J * sum (XX + YY + ZZ)."""
        n = psi.n_qubits
        h_bond = J * (np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z))
        bonds = [h_bond.copy() for _ in range(n - 1)]
        return cls(psi, bonds, dt=dt, **kwargs)

    @classmethod
    def xxz(cls, psi: MatrixProductState, J: float = 1.0, Delta: float = 1.0,
            dt: float = 0.01, **kwargs) -> 'TEBD':
        """XXZ: H = J * sum (XX + YY + Delta*ZZ)."""
        n = psi.n_qubits
        h_bond = J * (np.kron(_X, _X) + np.kron(_Y, _Y) + Delta * np.kron(_Z, _Z))
        bonds = [h_bond.copy() for _ in range(n - 1)]
        return cls(psi, bonds, dt=dt, **kwargs)
