# -*- coding: utf-8 -*-
# author: vinhpx
"""
iTEBD (infinite Time-Evolving Block Decimation) for infinite 1D systems.

Implements imaginary/real time evolution on translation-invariant infinite
MPS in Vidal canonical form (Gamma-Lambda representation).

Usage::

    from qforge.itebd import iTEBD

    # Find ground state of infinite transverse-field Ising chain
    sim = iTEBD.ising(J=1.0, h=0.5, chi=64)
    sim.evolve_imaginary(dt=0.1, n_steps=100)
    sim.evolve_imaginary(dt=0.01, n_steps=100)   # refine
    sim.evolve_imaginary(dt=0.001, n_steps=100)   # converge
    print(f"Energy per site: {sim.energy():.8f}")
    print(f"Correlation length: {sim.correlation_length():.4f}")
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from typing import Optional

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class iTEBD:
    """Infinite TEBD simulator with two-site unit cell (Vidal canonical form).

    State representation:
        |psi> = ... lambda_A * Gamma_A * lambda_B * Gamma_B * lambda_A * ...

    Args:
        d:   Physical dimension (default 2 for qubits).
        chi: Maximum bond dimension.
        eps: SVD truncation threshold.
    """

    def __init__(self, d: int = 2, chi: int = 64, eps: float = 1e-10) -> None:
        self.d = d
        self.chi = chi
        self.eps = eps

        # Initialize Gamma tensors [chi_l, d, chi_r] as identity-like
        # and Lambda vectors as [1.0]
        self.Gamma_A = np.zeros((1, d, 1), dtype=complex)
        self.Gamma_A[0, 0, 0] = 1.0  # |0> state
        self.Gamma_B = np.zeros((1, d, 1), dtype=complex)
        self.Gamma_B[0, 0, 0] = 1.0  # |0> state
        self.lambda_A = np.array([1.0])  # bond between ...B-A
        self.lambda_B = np.array([1.0])  # bond between A-B...

        self._h_bond: Optional[np.ndarray] = None

    def set_hamiltonian(self, h_bond: np.ndarray) -> None:
        """Set the two-site Hamiltonian density (4x4 matrix)."""
        self._h_bond = np.asarray(h_bond, dtype=complex).reshape(4, 4)

    def evolve(self, dt: float, n_steps: int, order: int = 2) -> None:
        """Real-time evolution: exp(-i*dt*H)."""
        self._evolve_steps(dt, n_steps, order, imaginary=False)

    def evolve_imaginary(self, dt: float, n_steps: int, order: int = 2) -> None:
        """Imaginary-time evolution: exp(-dt*H) for ground state finding."""
        self._evolve_steps(dt, n_steps, order, imaginary=True)

    def _evolve_steps(self, dt: float, n_steps: int,
                      order: int, imaginary: bool) -> None:
        if self._h_bond is None:
            raise RuntimeError("Set Hamiltonian first via set_hamiltonian()")

        factor = -dt if imaginary else -1j * dt

        if order == 1:
            U = expm(factor * self._h_bond)
            for _ in range(n_steps):
                self._apply_gate_AB(U)
                self._apply_gate_BA(U)
                if imaginary:
                    self._normalize()
        elif order == 2:
            U_half = expm(0.5 * factor * self._h_bond)
            U_full = expm(factor * self._h_bond)
            for _ in range(n_steps):
                self._apply_gate_AB(U_half)
                self._apply_gate_BA(U_full)
                self._apply_gate_AB(U_half)
                if imaginary:
                    self._normalize()
        elif order == 4:
            p = 1.0 / (2.0 - 2.0**(1.0/3.0))
            dts = [p * dt, p * dt, (1 - 4 * p) * dt]
            for _ in range(n_steps):
                for sub_dt in [dts[0], dts[1], dts[2], dts[1], dts[0]]:
                    sub_factor = -sub_dt if imaginary else -1j * sub_dt
                    U = expm(sub_factor * self._h_bond)
                    self._apply_gate_AB(U)
                    self._apply_gate_BA(U)
                if imaginary:
                    self._normalize()
        else:
            raise ValueError(f"Order must be 1, 2, or 4, got {order}")

    def _apply_gate_AB(self, U: np.ndarray) -> None:
        """Apply two-site gate to A-B bond."""
        d = self.d
        chi_A = len(self.lambda_A)
        chi_B = len(self.lambda_B)

        # theta[a, s_A, s_B, b] = lambda_A[a] * Gamma_A[a, s_A, :] * lambda_B[:] * Gamma_B[:, s_B, b] * lambda_A[b]
        # = diag(lambda_A) @ Gamma_A @ diag(lambda_B) @ Gamma_B @ diag(lambda_A)
        theta = np.einsum(
            'a,asc,c,cte,e->aste',
            self.lambda_A, self.Gamma_A, self.lambda_B,
            self.Gamma_B, self.lambda_A
        )

        # Apply gate
        theta_flat = theta.reshape(chi_A, d * d, chi_A).transpose(1, 0, 2).reshape(d * d, -1)
        # Reshape for gate application
        theta = theta.reshape(chi_A * d, d * chi_A)
        theta_gate = np.einsum(
            'ijkl,akle->aije',
            U.reshape(d, d, d, d),
            theta.reshape(chi_A, d, d, chi_A)
        )

        # SVD
        M = theta_gate.reshape(chi_A * d, d * chi_A)
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        threshold = self.eps * S[0] if S[0] > 0 else self.eps
        keep = max(1, min(self.chi, int(np.sum(S > threshold))))
        U_svd = U_svd[:, :keep]
        S = S[:keep]
        Vt = Vt[:keep, :]

        # Normalize singular values
        nrm = np.linalg.norm(S)
        if nrm > 1e-14:
            S /= nrm
        self.lambda_B = S

        # Update Gamma tensors
        # Gamma_A_new = diag(1/lambda_A) @ U_svd reshaped
        inv_lA = np.where(self.lambda_A > 1e-14, 1.0 / self.lambda_A, 0.0)
        GA = U_svd.reshape(chi_A, d, keep)
        self.Gamma_A = np.einsum('a,asc->asc', inv_lA, GA)

        inv_lA_r = np.where(self.lambda_A > 1e-14, 1.0 / self.lambda_A, 0.0)
        GB = Vt.reshape(keep, d, chi_A)
        self.Gamma_B = np.einsum('asc,c->asc', GB, inv_lA_r[:chi_A])

    def _apply_gate_BA(self, U: np.ndarray) -> None:
        """Apply two-site gate to B-A bond (shifted unit cell)."""
        d = self.d
        chi_A = len(self.lambda_A)
        chi_B = len(self.lambda_B)

        # theta = lambda_B @ Gamma_B @ lambda_A @ Gamma_A @ lambda_B
        theta = np.einsum(
            'a,asc,c,cte,e->aste',
            self.lambda_B, self.Gamma_B, self.lambda_A,
            self.Gamma_A, self.lambda_B
        )

        # Apply gate
        theta_gate = np.einsum(
            'ijkl,akle->aije',
            U.reshape(d, d, d, d),
            theta
        )

        # SVD
        M = theta_gate.reshape(chi_B * d, d * chi_B)
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        threshold = self.eps * S[0] if S[0] > 0 else self.eps
        keep = max(1, min(self.chi, int(np.sum(S > threshold))))
        U_svd = U_svd[:, :keep]
        S = S[:keep]
        Vt = Vt[:keep, :]

        nrm = np.linalg.norm(S)
        if nrm > 1e-14:
            S /= nrm
        self.lambda_A = S

        inv_lB = np.where(self.lambda_B > 1e-14, 1.0 / self.lambda_B, 0.0)
        GB = U_svd.reshape(chi_B, d, keep)
        self.Gamma_B = np.einsum('a,asc->asc', inv_lB, GB)

        inv_lB_r = np.where(self.lambda_B > 1e-14, 1.0 / self.lambda_B, 0.0)
        GA = Vt.reshape(keep, d, chi_B)
        self.Gamma_A = np.einsum('asc,c->asc', GA, inv_lB_r[:chi_B])

    def _normalize(self) -> None:
        """Normalize lambda vectors."""
        for lam in [self.lambda_A, self.lambda_B]:
            nrm = np.linalg.norm(lam)
            if nrm > 1e-14:
                lam /= nrm

    def energy(self) -> float:
        """Energy per site: <psi|h_AB|psi> (= E/N for translation-invariant systems)."""
        if self._h_bond is None:
            raise RuntimeError("Set Hamiltonian first")
        d = self.d
        h4 = self._h_bond.reshape(d, d, d, d)

        # theta_AB[a, s_A, s_B, b] = lambda_A[a] * Gamma_A * lambda_B * Gamma_B * lambda_A[b]
        theta = np.einsum(
            'a,asc,c,cte,e->aste',
            self.lambda_A, self.Gamma_A, self.lambda_B,
            self.Gamma_B, self.lambda_A
        )
        # <h_AB> = sum_{s,t,s',t'} h[s',t',s,t] * sum_a conj(theta[a,s',t',b]) * theta[a,s,t,b]
        e_AB = np.real(np.einsum('aste,ijst,aije->', np.conj(theta), h4, theta))

        # theta_BA
        theta_BA = np.einsum(
            'a,asc,c,cte,e->aste',
            self.lambda_B, self.Gamma_B, self.lambda_A,
            self.Gamma_A, self.lambda_B
        )
        e_BA = np.real(np.einsum('aste,ijst,aije->', np.conj(theta_BA), h4, theta_BA))

        return float((e_AB + e_BA) / 2.0)

    def entanglement_entropy(self) -> float:
        """Von Neumann entropy at the A-B bond in bits."""
        sv2 = self.lambda_B ** 2
        total = sv2.sum()
        if total < 1e-14:
            return 0.0
        sv2 = sv2 / total
        sv2 = sv2[sv2 > 1e-14]
        return float(-np.sum(sv2 * np.log2(sv2)))

    def correlation_length(self) -> float:
        """Correlation length from the transfer matrix spectrum."""
        d = self.d
        # Transfer matrix T for one unit cell: contract through A-B
        # T[a,b ; a',b'] = sum_{s,t} (lambda_B * Gamma_A * lambda_B)* @ (lambda_B * Gamma_A * lambda_B)
        chi_B = len(self.lambda_B)
        chi_A = len(self.lambda_A)

        # One-cell transfer matrix
        # Combine A and B into one two-site tensor
        M = np.einsum('asc,c,cte->aste', self.Gamma_A, self.lambda_B, self.Gamma_B)
        # M[chi_A, d, d, chi_A] — one unit cell

        # Transfer matrix: T[a,b,a',b'] = sum_{s,t} conj(M[a,s,t,b]) * M[a',s,t,b']
        T = np.einsum('aste,cpte->acsp',
                       np.conj(M) * self.lambda_A[:, None, None, None],
                       M * self.lambda_A[None, None, None, :])
        # T has shape [chi_A, chi_A, chi_A, chi_A]
        chi = chi_A
        T_mat = T.reshape(chi * chi, chi * chi)

        eigvals = np.linalg.eigvals(T_mat)
        eigvals = np.sort(np.abs(eigvals))[::-1]

        if len(eigvals) < 2 or eigvals[1] < 1e-14:
            return float('inf')  # Product state, infinite correlation length

        # xi = -1 / ln(|lambda_2/lambda_1|)
        ratio = eigvals[1] / eigvals[0]
        if ratio >= 1.0 - 1e-14:
            return float('inf')
        return float(-1.0 / np.log(ratio))

    def bond_dimension(self) -> int:
        """Current bond dimension."""
        return max(len(self.lambda_A), len(self.lambda_B))

    # ----------------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def ising(cls, J: float = 1.0, h: float = 0.5,
              chi: int = 64, **kwargs) -> 'iTEBD':
        """Transverse-field Ising: h = -J*ZZ - h/2*(X_1 + X_2)."""
        sim = cls(d=2, chi=chi, **kwargs)
        h_bond = -J * np.kron(_Z, _Z) - 0.5 * h * (np.kron(_X, _I) + np.kron(_I, _X))
        sim.set_hamiltonian(h_bond)
        return sim

    @classmethod
    def heisenberg(cls, J: float = 1.0, chi: int = 64, **kwargs) -> 'iTEBD':
        """Heisenberg: h = J * (XX + YY + ZZ)."""
        sim = cls(d=2, chi=chi, **kwargs)
        h_bond = J * (np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z))
        sim.set_hamiltonian(h_bond)
        return sim

    @classmethod
    def xxz(cls, J: float = 1.0, Delta: float = 1.0,
            chi: int = 64, **kwargs) -> 'iTEBD':
        """XXZ: h = J * (XX + YY + Delta*ZZ)."""
        sim = cls(d=2, chi=chi, **kwargs)
        h_bond = J * (np.kron(_X, _X) + np.kron(_Y, _Y) + Delta * np.kron(_Z, _Z))
        sim.set_hamiltonian(h_bond)
        return sim
