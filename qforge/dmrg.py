# -*- coding: utf-8 -*-
# author: vinhpx
"""
DMRG (Density Matrix Renormalization Group) solver for qforge.

Implements two-site DMRG sweeps with Lanczos eigensolver to find
ground states of 1D spin chain Hamiltonians. Uses MPS representation.

Usage::

    from qforge.dmrg import DMRG

    # Heisenberg chain, 20 sites
    dmrg = DMRG.heisenberg(n_sites=20, max_bond_dim=64)
    energy, psi = dmrg.run(n_sweeps=10, verbose=True)
    print(f"Ground state energy per site: {energy / 20:.6f}")

    # Magnetization profile
    print(dmrg.magnetization_profile())

    # Entanglement entropy at bond 9
    print(psi.entanglement_entropy(9))
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from qforge.mps import MatrixProductState, _init_product_state, _entropy_py

try:
    from qforge._qforge_mps import (
        MPO as _MPO_core,
        build_heisenberg_mpo,
        build_ising_mpo,
        build_xxz_mpo,
        dmrg_sweep as _dmrg_sweep_cpp,
        dmrg_sweep_1site as _dmrg_sweep_1site_cpp,
        dmrg_sweep_excited as _dmrg_sweep_excited_cpp,
    )
    _HAS_MPS_CPP = True
except ImportError:
    _HAS_MPS_CPP = False


class DMRG:
    """Variational ground-state finder via two-site DMRG sweeps.

    Create via factory class methods (recommended) or directly.

    Args:
        n_sites:      Number of lattice sites (= qubits).
        hamiltonian:  MPO Hamiltonian object (from factory) or dict spec (Python path).
        max_bond_dim: Maximum MPS bond dimension chi. Default 32.
        eps:          SVD truncation threshold. Default 1e-10.
        lanczos_dim:  Krylov subspace size for local eigensolver. Default 20.
        backend:      'auto', 'cpu', or 'python'.
    """

    def __init__(
        self,
        n_sites: int,
        hamiltonian: object,
        max_bond_dim: int = 32,
        eps: float = 1e-10,
        lanczos_dim: int = 20,
        backend: str = 'auto',
        _hamiltonian_py_spec=None,
    ) -> None:
        if n_sites < 2:
            raise ValueError("n_sites must be >= 2")
        self.n_sites = n_sites
        self.max_bond_dim = max_bond_dim
        self.eps = eps
        self.lanczos_dim = lanczos_dim
        self._energy_history: List[float] = []

        if backend == 'auto':
            backend = 'cpu' if _HAS_MPS_CPP else 'python'
        self.backend = backend

        if _HAS_MPS_CPP and isinstance(hamiltonian, _MPO_core):
            self._hamiltonian_cpp = hamiltonian
            # Always store Python spec for Python fallback path
            self._hamiltonian_py = _hamiltonian_py_spec
        else:
            self._hamiltonian_cpp = None
            self._hamiltonian_py = hamiltonian if _hamiltonian_py_spec is None else _hamiltonian_py_spec

        # Initialize MPS in |00...0> random-ish state for DMRG
        self._psi = MatrixProductState(n_sites, max_bond_dim=max_bond_dim,
                                       eps=eps, backend=backend)
        self._randomize_mps()

    # ----------------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------------

    @classmethod
    def heisenberg(cls, n_sites: int, J: float = 1.0,
                   max_bond_dim: int = 32, **kwargs) -> 'DMRG':
        """Heisenberg chain: H = J * sum_i (XX + YY + ZZ)."""
        py_spec = {'type': 'heisenberg', 'J': J, 'n_sites': n_sites}
        H = build_heisenberg_mpo(n_sites, J) if _HAS_MPS_CPP else py_spec
        return cls(n_sites, H, max_bond_dim=max_bond_dim,
                   _hamiltonian_py_spec=py_spec, **kwargs)

    @classmethod
    def ising(cls, n_sites: int, J: float = 1.0, h: float = 0.5,
              max_bond_dim: int = 32, **kwargs) -> 'DMRG':
        """Transverse-field Ising: H = -J * sum_i ZZ - h * sum_i X."""
        py_spec = {'type': 'ising', 'J': J, 'h': h, 'n_sites': n_sites}
        H = build_ising_mpo(n_sites, J, h) if _HAS_MPS_CPP else py_spec
        return cls(n_sites, H, max_bond_dim=max_bond_dim,
                   _hamiltonian_py_spec=py_spec, **kwargs)

    @classmethod
    def xxz(cls, n_sites: int, Delta: float = 1.0, J: float = 1.0,
            max_bond_dim: int = 32, **kwargs) -> 'DMRG':
        """XXZ chain: H = J * sum_i (XX + YY + Delta * ZZ)."""
        py_spec = {'type': 'xxz', 'Delta': Delta, 'J': J, 'n_sites': n_sites}
        H = build_xxz_mpo(n_sites, Delta, J) if _HAS_MPS_CPP else py_spec
        return cls(n_sites, H, max_bond_dim=max_bond_dim,
                   _hamiltonian_py_spec=py_spec, **kwargs)

    @classmethod
    def custom(cls, n_sites: int, terms: list = None,
               builder: object = None, max_bond_dim: int = 32, **kwargs) -> 'DMRG':
        """Custom Hamiltonian from MPOBuilder terms.

        Args:
            n_sites: Number of sites.
            terms: List of (coeff, [(op, site), ...]) tuples.
            builder: MPOBuilder instance (alternative to terms).
            max_bond_dim: Maximum bond dimension.

        Usage::

            from qforge.mpo_builder import MPOBuilder
            builder = MPOBuilder(10)
            builder.add_term(-1.0, [('Z', 0), ('Z', 1)])
            dmrg = DMRG.custom(10, builder=builder)
        """
        from qforge.mpo_builder import MPOBuilder
        if builder is None:
            if terms is None:
                raise ValueError("Provide either 'terms' or 'builder'")
            builder = MPOBuilder(n_sites)
            for coeff, ops in terms:
                builder.add_term(coeff, ops)
        py_spec = builder.build_py()
        return cls(n_sites, py_spec, max_bond_dim=max_bond_dim,
                   _hamiltonian_py_spec=py_spec, **kwargs)

    # ----------------------------------------------------------------
    # Main interface
    # ----------------------------------------------------------------

    def run(
        self,
        n_sweeps: int = 10,
        convergence_tol: float = 1e-8,
        verbose: bool = False,
        algorithm: str = '2site',
    ) -> Tuple[float, MatrixProductState]:
        """Run DMRG sweeps until convergence or max sweeps.

        Args:
            n_sweeps:        Maximum number of sweeps.
            convergence_tol: Stop when |E_new - E_old| < tol.
            verbose:         Print energy per sweep.
            algorithm:       '2site' (default) or '1site'.
                             Single-site is more memory efficient but can
                             get stuck in local minima for small bond dims.

        Returns:
            (energy, psi): Ground state energy and MPS ground state.
        """
        if algorithm not in ('2site', '1site'):
            raise ValueError(f"algorithm must be '2site' or '1site', got {algorithm!r}")
        self._energy_history = []

        use_cpp = (self.backend == 'cpu' and _HAS_MPS_CPP
                   and self._hamiltonian_cpp is not None
                   and self._psi._mps is not None)

        if use_cpp:
            energy = self._run_cpp(n_sweeps, convergence_tol, verbose, algorithm)
        else:
            energy = self._run_python(n_sweeps, convergence_tol, verbose, algorithm)

        return energy, self._psi

    def run_excited(
        self,
        n_states: int = 3,
        n_sweeps: int = 10,
        convergence_tol: float = 1e-8,
        verbose: bool = False,
        weight: float = 10.0,
    ) -> List[Tuple[float, MatrixProductState]]:
        """Find multiple eigenstates via sequential DMRG with penalty method.

        Args:
            n_states:        Number of states to find (ground + excited).
            n_sweeps:        Sweeps per state.
            convergence_tol: Convergence threshold per state.
            verbose:         Print progress.
            weight:          Penalty weight for orthogonalization.

        Returns:
            List of (energy, psi) tuples, sorted by energy.
        """
        results: List[Tuple[float, MatrixProductState]] = []
        prev_mps_cpp = []  # C++ MPS objects for penalty

        for state_idx in range(n_states):
            if verbose:
                print(f"\n--- Finding state {state_idx} ---")

            # Fresh MPS for each state
            self._psi = MatrixProductState(
                self.n_sites, max_bond_dim=self.max_bond_dim,
                eps=self.eps, backend=self.backend)
            self._randomize_mps()
            self._energy_history = []

            use_cpp = (self.backend == 'cpu' and _HAS_MPS_CPP
                       and self._hamiltonian_cpp is not None
                       and self._psi._mps is not None)

            if use_cpp and state_idx == 0:
                # Ground state: standard DMRG
                energy = self._run_cpp(n_sweeps, convergence_tol, verbose, '2site')
            elif use_cpp:
                # Excited state: penalty method
                energy = self._run_excited_cpp(
                    n_sweeps, convergence_tol, verbose,
                    prev_mps_cpp, weight)
            elif state_idx == 0:
                energy = self._run_python(n_sweeps, convergence_tol, verbose, '2site')
            else:
                energy = self._run_excited_python(
                    n_sweeps, convergence_tol, verbose,
                    [r[1] for r in results], weight)

            results.append((energy, self._psi))
            if use_cpp and self._psi._mps is not None:
                prev_mps_cpp.append(self._psi._mps)

        return results

    def _run_cpp(self, n_sweeps: int, tol: float, verbose: bool,
                 algorithm: str) -> float:
        sweep_fn = (_dmrg_sweep_1site_cpp if algorithm == '1site'
                    else _dmrg_sweep_cpp)
        energy = 0.0
        for sweep in range(n_sweeps):
            prev = energy
            energy = sweep_fn(
                self._psi._mps, self._hamiltonian_cpp,
                self.max_bond_dim, self.eps, self.lanczos_dim,
            )
            self._energy_history.append(energy)
            if verbose:
                chi = max(self._psi.bond_dimensions())
                print(f"Sweep {sweep + 1:3d}: E = {energy:.10f},  "
                      f"max chi = {chi}")
            if sweep > 0 and abs(energy - prev) < tol:
                if verbose:
                    print(f"Converged after {sweep + 1} sweeps.")
                break
        return energy

    def _run_excited_cpp(self, n_sweeps: int, tol: float, verbose: bool,
                         prev_mps_cpp: list, weight: float) -> float:
        energy = 0.0
        for sweep in range(n_sweeps):
            prev = energy
            energy = _dmrg_sweep_excited_cpp(
                self._psi._mps, self._hamiltonian_cpp,
                self.max_bond_dim, self.eps, self.lanczos_dim,
                prev_mps_cpp, weight,
            )
            self._energy_history.append(energy)
            if verbose:
                chi = max(self._psi.bond_dimensions())
                print(f"Sweep {sweep + 1:3d}: E = {energy:.10f},  "
                      f"max chi = {chi}")
            if sweep > 0 and abs(energy - prev) < tol:
                if verbose:
                    print(f"Converged after {sweep + 1} sweeps.")
                break
        return energy

    def _run_python(self, n_sweeps: int, tol: float, verbose: bool,
                    algorithm: str) -> float:
        """Pure-Python DMRG. Correct but slow — for testing/small systems."""
        sweep_fn = (_dmrg_sweep_1site_py if algorithm == '1site'
                    else _dmrg_sweep_py)
        energy = 0.0
        H_py = self._hamiltonian_py
        tensors = self._psi._tensors

        for sweep in range(n_sweeps):
            prev = energy
            energy = sweep_fn(
                tensors, H_py, self.max_bond_dim, self.eps, self.lanczos_dim
            )
            self._energy_history.append(energy)
            if verbose:
                print(f"Sweep {sweep + 1:3d}: E = {energy:.10f}")
            if sweep > 0 and abs(energy - prev) < tol:
                if verbose:
                    print(f"Converged after {sweep + 1} sweeps.")
                break
        return energy

    def _run_excited_python(self, n_sweeps: int, tol: float, verbose: bool,
                            prev_psi_list: list, weight: float) -> float:
        energy = 0.0
        H_py = self._hamiltonian_py
        tensors = self._psi._tensors
        prev_tensors = [p._tensors for p in prev_psi_list]

        for sweep in range(n_sweeps):
            prev = energy
            energy = _dmrg_sweep_excited_py(
                tensors, H_py, self.max_bond_dim, self.eps,
                self.lanczos_dim, prev_tensors, weight
            )
            self._energy_history.append(energy)
            if verbose:
                print(f"Sweep {sweep + 1:3d}: E = {energy:.10f}")
            if sweep > 0 and abs(energy - prev) < tol:
                if verbose:
                    print(f"Converged after {sweep + 1} sweeps.")
                break
        return energy

    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------

    @property
    def energy_history(self) -> List[float]:
        """Energy per sweep."""
        return list(self._energy_history)

    def ground_state_mps(self) -> MatrixProductState:
        """Return the ground state MPS."""
        return self._psi

    def magnetization_profile(self) -> np.ndarray:
        """<Z_i> for each site i."""
        Z = np.array([1, 0, 0, -1], dtype=complex)  # row-major 2x2 Z
        psi = self._psi
        if psi._mps is not None:
            return np.array([
                psi._mps.single_site_expectation(i, Z).real
                for i in range(self.n_sites)
            ])
        return np.array([
            _single_site_expect_py(psi._tensors, Z.reshape(2, 2), i).real
            for i in range(self.n_sites)
        ])

    def correlation_function(self, op_i: str, op_j: str,
                             i: int, j: int) -> float:
        """<O_i O_j> - <O_i><O_j> (connected correlator)."""
        _PAULI = {
            'X': np.array([0, 1, 1, 0], dtype=complex),
            'Y': np.array([0, -1j, 1j, 0], dtype=complex),
            'Z': np.array([1, 0, 0, -1], dtype=complex),
        }
        oi = _PAULI[op_i]
        oj = _PAULI[op_j]
        psi = self._psi
        if psi._mps is not None:
            eij = psi._mps.two_site_expectation(i, j, oi, oj).real
            ei = psi._mps.single_site_expectation(i, oi).real
            ej = psi._mps.single_site_expectation(j, oj).real
        else:
            eij = _two_site_expect_py(psi._tensors, oi.reshape(2, 2),
                                      oj.reshape(2, 2), i, j).real
            ei = _single_site_expect_py(psi._tensors, oi.reshape(2, 2), i).real
            ej = _single_site_expect_py(psi._tensors, oj.reshape(2, 2), j).real
        return float(eij - ei * ej)

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _randomize_mps(self) -> None:
        """Randomize MPS tensors to avoid starting in a symmetry-protected eigenstate."""
        psi = self._psi
        if psi._mps is not None:
            pass  # C++ DMRG handles initialization internally
        else:
            # Randomize each tensor with small noise so |00...0> eigenstate is broken
            rng = np.random.default_rng(42)
            for i, t in enumerate(psi._tensors):
                psi._tensors[i] = t + 0.1 * rng.standard_normal(t.shape).astype(complex)
                # Renormalize
                nrm = np.linalg.norm(psi._tensors[i])
                if nrm > 1e-14:
                    psi._tensors[i] /= nrm


# ================================================================
# Pure Python DMRG sweep (reference implementation)
# For correctness testing on small systems (n<=12, chi<=16)
# ================================================================

def _mpo_tensors_py(H_spec: dict) -> list:
    """Build list of MPO tensors as numpy arrays from spec dict."""
    n = H_spec['n_sites']
    t = H_spec['type']
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)

    if t == 'heisenberg':
        J = H_spec['J']
        # MPO W[wl, wr, sb, sk], bond dim W=5
        # H = J * sum_i (XX + YY + ZZ) = J * sum_i (2*SpSm + 2*SmSp + ZZ)
        # States: 0=done(I), 1=active-Sp, 2=active-Sm, 3=active-Z, 4=vacuum(I)
        W_bulk = np.zeros((5, 5, 2, 2), dtype=complex)
        W_bulk[0, 0] = I
        W_bulk[4, 4] = I
        W_bulk[4, 1] = J * Sp   # start Sp on site i
        W_bulk[4, 2] = J * Sm   # start Sm on site i
        W_bulk[4, 3] = J * Z    # start Z on site i
        W_bulk[1, 0] = 2 * Sm   # complete: J*Sp * 2*Sm = 2J*SpSm (half of XX+YY)
        W_bulk[2, 0] = 2 * Sp   # complete: J*Sm * 2*Sp = 2J*SmSp (other half)
        W_bulk[3, 0] = Z        # complete: J*Z * Z = J*ZZ

        W_first = W_bulk[4:5, :, :, :]  # shape [1, 5, 2, 2]
        W_last  = W_bulk[:, 0:1, :, :]  # shape [5, 1, 2, 2]

        tensors = [W_first] + [W_bulk] * (n - 2) + [W_last]
        return tensors

    elif t == 'ising':
        J = H_spec['J']
        h = H_spec['h']
        W_bulk = np.zeros((3, 3, 2, 2), dtype=complex)
        W_bulk[0, 0] = I
        W_bulk[2, 2] = I
        W_bulk[2, 1] = -J * Z
        W_bulk[1, 0] = Z
        W_bulk[2, 0] = -h * X

        W_first = W_bulk[2:3, :, :, :]
        W_last  = W_bulk[:, 0:1, :, :]
        tensors = [W_first] + [W_bulk] * (n - 2) + [W_last]
        return tensors

    elif t == 'xxz':
        J = H_spec['J']
        Delta = H_spec['Delta']
        # H = J * sum_i (XX + YY + Delta*ZZ) = J * sum_i (2*SpSm + 2*SmSp + Delta*ZZ)
        W_bulk = np.zeros((5, 5, 2, 2), dtype=complex)
        W_bulk[0, 0] = I
        W_bulk[4, 4] = I
        W_bulk[4, 1] = J * Sp
        W_bulk[4, 2] = J * Sm
        W_bulk[4, 3] = J * Delta * Z
        W_bulk[1, 0] = 2 * Sm
        W_bulk[2, 0] = 2 * Sp
        W_bulk[3, 0] = Z

        W_first = W_bulk[4:5, :, :, :]
        W_last  = W_bulk[:, 0:1, :, :]
        tensors = [W_first] + [W_bulk] * (n - 2) + [W_last]
        return tensors

    elif t == 'custom' and 'tensors' in H_spec:
        return H_spec['tensors']

    raise ValueError(f"Unknown Hamiltonian type: {t}")


def _build_left_env_py(mps_tensors: list, mpo_tensors: list,
                       up_to: int) -> np.ndarray:
    """Build left environment L[chi, w, chi] up to (not including) site `up_to`."""
    L = np.ones((1, 1, 1), dtype=complex)
    for i in range(up_to):
        A = mps_tensors[i]   # [chi_l, d, chi_r]
        W = mpo_tensors[i]   # [wl, wr, d, d]
        # L_new[a', wr, b'] = sum_{a, wl, b, sb, sk} conj(A[a, sb, a']) L[a, wl, b] W[wl, wr, sb, sk] A[b, sk, b']
        chi_l = A.shape[0]; chi_r = A.shape[2]; d = A.shape[1]
        wl = W.shape[0]; wr = W.shape[1]
        L_new = np.zeros((chi_r, wr, chi_r), dtype=complex)
        for a in range(chi_l):
            for wl_i in range(wl):
                for b in range(chi_l):
                    lval = L[a, wl_i, b]
                    if abs(lval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[wl_i, :, sb, sk]  # [wr]
                            tmp = lval * np.outer(np.conj(A[a, sb, :]), A[b, sk, :])  # [chi_r, chi_r]
                            L_new[:, :, :] += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        L = L_new
    return L


def _build_right_env_py(mps_tensors: list, mpo_tensors: list,
                        from_site: int) -> np.ndarray:
    """Build right environment R[chi, w, chi] from (not including) site `from_site`."""
    R = np.ones((1, 1, 1), dtype=complex)
    n = len(mps_tensors)
    for i in range(n - 1, from_site, -1):
        A = mps_tensors[i]
        W = mpo_tensors[i]
        chi_l = A.shape[0]; chi_r = A.shape[2]; d = A.shape[1]
        wl = W.shape[0]; wr = W.shape[1]
        R_new = np.zeros((chi_l, wl, chi_l), dtype=complex)
        for a in range(chi_r):
            for wr_i in range(wr):
                for b in range(chi_r):
                    rval = R[a, wr_i, b]
                    if abs(rval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[:, wr_i, sb, sk]  # [wl]
                            tmp = rval * np.outer(np.conj(A[:, sb, a]), A[:, sk, b])  # [chi_l, chi_l]
                            R_new[:, :, :] += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        R = R_new
    return R


def _apply_heff_py(L, W1, W2, R, theta, chi_l, chi_r):
    """Apply effective Hamiltonian to two-site theta tensor."""
    d = 2
    # L: [chi_l, wl, chi_l], W1: [wl, wm, d, d], W2: [wm, wr, d, d], R: [chi_r, wr, chi_r]
    # theta: [chi_l, d, d, chi_r]
    theta = theta.reshape(chi_l, d, d, chi_r)
    result = np.einsum('aub,uvsd,vwte,cwf,bdef->astc',
                       L, W1, W2, R, theta.reshape(chi_l, d, d, chi_r),
                       optimize=True)
    return result.reshape(-1)


def _lanczos_py(matvec, v0, dim, max_iter, tol):
    """Lanczos eigensolver (Python fallback)."""
    max_iter = min(max_iter, dim)
    v0 = v0.copy()
    nrm = np.linalg.norm(v0)
    if nrm < 1e-14:
        v0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    else:
        v0 /= nrm

    V = [v0]
    alpha, beta = [], []
    energy = 0.0

    for j in range(max_iter):
        w = matvec(V[j])
        aj = np.real(np.dot(np.conj(V[j]), w))
        alpha.append(aj)
        w -= aj * V[j]
        if j > 0:
            w -= beta[-1] * V[j - 1]
        bj = np.linalg.norm(w)

        m = len(alpha)
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        eigvals, eigvecs = np.linalg.eigh(T)
        energy = eigvals[0]

        if bj < tol or j == max_iter - 1:
            y = eigvecs[:, 0]
            vec = sum(float(y[k]) * V[k] for k in range(m))
            return energy, vec

        beta.append(bj)
        V.append(w / bj)

    return energy, V[0]


def _single_site_expect_py(tensors, op, site):
    n = len(tensors)
    L = np.ones((1, 1), dtype=complex)
    for i in range(n):
        A = tensors[i]  # [chi_l, d, chi_r]
        if i == site:
            L = np.einsum('ab,asc,st,btd->cd', L, np.conj(A), op, A)
        else:
            L = np.einsum('ab,asc,bsd->cd', L, np.conj(A), A)
    return complex(L[0, 0])


def _two_site_expect_py(tensors, oi, oj, si, sj):
    n = len(tensors)
    I = np.eye(2, dtype=complex)
    L = np.ones((1, 1), dtype=complex)
    for i in range(n):
        A = tensors[i]
        op = oi if i == si else (oj if i == sj else I)
        L = np.einsum('ab,asc,st,btd->cd', L, np.conj(A), op, A)
    return complex(L[0, 0])


def _dmrg_sweep_py(tensors: list, H_spec: dict,
                   max_chi: int, eps: float, lanczos_dim: int) -> float:
    """Pure-Python two-site DMRG sweep."""
    n = len(tensors)
    d = 2
    W_tensors = _mpo_tensors_py(H_spec)
    energy = 0.0

    # Build right environments
    R_envs = [None] * (n + 1)
    R_envs[n] = np.ones((1, 1, 1), dtype=complex)
    for i in range(n - 1, 0, -1):
        A = tensors[i]; W = W_tensors[i]
        chi_r = A.shape[2]; wr = W.shape[1]
        R_new = np.zeros((A.shape[0], W.shape[0], A.shape[0]), dtype=complex)
        for a in range(chi_r):
            for wr_i in range(wr):
                for b in range(chi_r):
                    rval = R_envs[i + 1][a, wr_i, b]
                    if abs(rval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[:, wr_i, sb, sk]
                            tmp = rval * np.outer(np.conj(A[:, sb, a]), A[:, sk, b])
                            R_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        R_envs[i] = R_new

    L = np.ones((1, 1, 1), dtype=complex)

    # Left sweep
    for i in range(n - 1):
        A = tensors[i]; B = tensors[i + 1]
        chi_l = A.shape[0]; chi_m = A.shape[2]; chi_r = B.shape[2]
        W1 = W_tensors[i]; W2 = W_tensors[i + 1]
        R = R_envs[i + 2] if i + 2 <= n else np.ones((1, 1, 1), dtype=complex)

        # Merge theta = A * B
        theta = np.einsum('asb,bte->aste', A, B).reshape(-1)
        theta_dim = chi_l * d * d * chi_r

        def matvec(v):
            return _apply_heff_py(L, W1, W2, R, v, chi_l, chi_r)

        e, theta_opt = _lanczos_py(matvec, theta, theta_dim, lanczos_dim, 1e-10)
        energy = e

        # SVD split
        M = theta_opt.reshape(chi_l * d, d * chi_r)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        threshold = eps * S[0] if S[0] > 0 else eps
        keep = max(1, min(max_chi, int(np.sum(S > threshold))))
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep, :]
        tensors[i] = U.reshape(chi_l, d, keep)
        tensors[i + 1] = (np.diag(S) @ Vt).reshape(keep, d, chi_r)

        # Update L
        A_new = tensors[i]; W = W_tensors[i]
        chi_r2 = A_new.shape[2]; wr = W.shape[1]
        L_new = np.zeros((chi_r2, wr, chi_r2), dtype=complex)
        chi_l_L = L.shape[0]; wl = W.shape[0]
        for a in range(chi_l_L):
            for wl_i in range(wl):
                for b in range(chi_l_L):
                    lval = L[a, wl_i, b]
                    if abs(lval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[wl_i, :, sb, sk]
                            tmp = lval * np.outer(np.conj(A_new[a, sb, :]),
                                                  A_new[b, sk, :])
                            L_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        L = L_new

    return float(energy)


def _apply_heff_1site_py(L, W, R, theta, chi_l, chi_r):
    """Apply effective single-site Hamiltonian to theta tensor."""
    d = 2
    theta = theta.reshape(chi_l, d, chi_r)
    result = np.einsum('aub,uvsd,cwf,bdf->asc',
                       L, W, R, theta,
                       optimize=True)
    return result.reshape(-1)


def _dmrg_sweep_1site_py(tensors: list, H_spec: dict,
                          max_chi: int, eps: float, lanczos_dim: int) -> float:
    """Pure-Python single-site DMRG sweep."""
    n = len(tensors)
    d = 2
    W_tensors = _mpo_tensors_py(H_spec)
    energy = 0.0

    # Build right environments
    R_envs = [None] * (n + 1)
    R_envs[n] = np.ones((1, 1, 1), dtype=complex)
    for i in range(n - 1, -1, -1):
        A = tensors[i]; W = W_tensors[i]
        chi_l_A = A.shape[0]; chi_r_A = A.shape[2]
        wl = W.shape[0]; wr = W.shape[1]
        R_new = np.zeros((chi_l_A, wl, chi_l_A), dtype=complex)
        for a in range(chi_r_A):
            for wr_i in range(wr):
                for b in range(chi_r_A):
                    rval = R_envs[i + 1][a, wr_i, b]
                    if abs(rval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[:, wr_i, sb, sk]
                            tmp = rval * np.outer(np.conj(A[:, sb, a]), A[:, sk, b])
                            R_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        R_envs[i] = R_new

    L = np.ones((1, 1, 1), dtype=complex)

    # Left-to-right sweep
    for i in range(n):
        A = tensors[i]
        chi_l = A.shape[0]; chi_r = A.shape[2]
        W = W_tensors[i]
        R = R_envs[i + 1]
        theta = A.reshape(-1).copy()
        theta_dim = chi_l * d * chi_r

        def matvec(v, _L=L, _W=W, _R=R, _cl=chi_l, _cr=chi_r):
            return _apply_heff_1site_py(_L, _W, _R, v, _cl, _cr)

        e, theta_opt = _lanczos_py(matvec, theta, theta_dim, lanczos_dim, 1e-10)
        energy = e

        if i < n - 1:
            # QR decomposition to shift ortho center right
            M = theta_opt.reshape(chi_l * d, chi_r)
            Q, R_mat = np.linalg.qr(M)
            k = Q.shape[1]
            tensors[i] = Q.reshape(chi_l, d, k)
            # Absorb R into next tensor
            B = tensors[i + 1]
            tensors[i + 1] = np.einsum('ab,bsc->asc', R_mat, B)

            # Update left environment
            A_new = tensors[i]; W_cur = W_tensors[i]
            chi_r2 = A_new.shape[2]; wr = W_cur.shape[1]
            L_new = np.zeros((chi_r2, wr, chi_r2), dtype=complex)
            chi_l_L = L.shape[0]; wl = W_cur.shape[0]
            for a in range(chi_l_L):
                for wl_i in range(wl):
                    for b in range(chi_l_L):
                        lval = L[a, wl_i, b]
                        if abs(lval) < 1e-15: continue
                        for sb in range(d):
                            for sk in range(d):
                                wval = W_cur[wl_i, :, sb, sk]
                                tmp = lval * np.outer(np.conj(A_new[a, sb, :]),
                                                      A_new[b, sk, :])
                                L_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
            L = L_new
        else:
            tensors[i] = theta_opt.reshape(chi_l, d, chi_r)

    # Build left environments for right-to-left sweep
    L_envs = [None] * (n + 1)
    L_envs[0] = np.ones((1, 1, 1), dtype=complex)
    for i in range(n):
        A = tensors[i]; W = W_tensors[i]
        chi_l_A = A.shape[0]; chi_r_A = A.shape[2]
        wl = W.shape[0]; wr = W.shape[1]
        L_new = np.zeros((chi_r_A, wr, chi_r_A), dtype=complex)
        for a in range(chi_l_A):
            for wl_i in range(wl):
                for b in range(chi_l_A):
                    lval = L_envs[i][a, wl_i, b]
                    if abs(lval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[wl_i, :, sb, sk]
                            tmp = lval * np.outer(np.conj(A[a, sb, :]), A[b, sk, :])
                            L_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        L_envs[i + 1] = L_new

    R = np.ones((1, 1, 1), dtype=complex)

    # Right-to-left sweep
    for i in range(n - 1, -1, -1):
        A = tensors[i]
        chi_l = A.shape[0]; chi_r = A.shape[2]
        W = W_tensors[i]
        L = L_envs[i]
        theta = A.reshape(-1).copy()
        theta_dim = chi_l * d * chi_r

        def matvec(v, _L=L, _W=W, _R=R, _cl=chi_l, _cr=chi_r):
            return _apply_heff_1site_py(_L, _W, _R, v, _cl, _cr)

        e, theta_opt = _lanczos_py(matvec, theta, theta_dim, lanczos_dim, 1e-10)
        energy = e

        if i > 0:
            # QR to shift ortho center left (on transposed matrix)
            M = theta_opt.reshape(chi_l, d * chi_r).T  # [d*chi_r, chi_l]
            Q, R_mat = np.linalg.qr(M)
            k = Q.shape[1]
            tensors[i] = Q.T.reshape(k, d, chi_r)
            # Absorb R^T into previous tensor
            A_prev = tensors[i - 1]
            tensors[i - 1] = np.einsum('asc,cb->asb', A_prev, R_mat.T)

            # Update right environment
            A_new = tensors[i]; W_cur = W_tensors[i]
            chi_l2 = A_new.shape[0]; wl = W_cur.shape[0]
            R_new = np.zeros((chi_l2, wl, chi_l2), dtype=complex)
            chi_r_R = R.shape[0]; wr_R = W_cur.shape[1]
            for a in range(chi_r):
                for wr_i in range(wr_R):
                    for b in range(chi_r):
                        rval = R[a, wr_i, b]
                        if abs(rval) < 1e-15: continue
                        for sb in range(d):
                            for sk in range(d):
                                wval = W_cur[:, wr_i, sb, sk]
                                tmp = rval * np.outer(np.conj(A_new[:, sb, a]),
                                                      A_new[:, sk, b])
                                R_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
            R = R_new
        else:
            tensors[i] = theta_opt.reshape(chi_l, d, chi_r)

    return float(energy)


def _dmrg_sweep_excited_py(tensors: list, H_spec: dict,
                            max_chi: int, eps: float, lanczos_dim: int,
                            prev_tensors_list: list, weight: float) -> float:
    """Pure-Python excited-state DMRG sweep with penalty method."""
    n = len(tensors)
    d = 2
    W_tensors = _mpo_tensors_py(H_spec)
    energy = 0.0

    # Build right environments for H
    R_envs = [None] * (n + 1)
    R_envs[n] = np.ones((1, 1, 1), dtype=complex)
    for i in range(n - 1, 0, -1):
        A = tensors[i]; W = W_tensors[i]
        chi_r = A.shape[2]; wr = W.shape[1]
        R_new = np.zeros((A.shape[0], W.shape[0], A.shape[0]), dtype=complex)
        for a in range(chi_r):
            for wr_i in range(wr):
                for b in range(chi_r):
                    rval = R_envs[i + 1][a, wr_i, b]
                    if abs(rval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W[:, wr_i, sb, sk]
                            tmp = rval * np.outer(np.conj(A[:, sb, a]), A[:, sk, b])
                            R_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        R_envs[i] = R_new

    # Build overlap environments for each previous state
    n_prev = len(prev_tensors_list)
    P_L = [[None] * (n + 1) for _ in range(n_prev)]
    P_R = [[None] * (n + 1) for _ in range(n_prev)]

    for j in range(n_prev):
        P_L[j][0] = np.ones((1, 1), dtype=complex)
        P_R[j][n] = np.ones((1, 1), dtype=complex)
        prev_t = prev_tensors_list[j]
        for i in range(n):
            A_psi = tensors[i]; A_prev = prev_t[i]
            P_L[j][i + 1] = np.einsum('ab,asc,bsd->cd',
                                       P_L[j][i], np.conj(A_psi), A_prev)
        for i in range(n - 1, -1, -1):
            A_psi = tensors[i]; A_prev = prev_t[i]
            P_R[j][i] = np.einsum('ab,asc,bsd->cd',
                                   np.conj(A_psi).transpose(2, 1, 0).reshape(A_psi.shape[2], -1).reshape(A_psi.shape[2], A_psi.shape[1], A_psi.shape[0]),
                                   # Simpler: just contract directly
                                   P_R[j][i + 1],
                                   A_prev.transpose(2, 1, 0).reshape(A_prev.shape[2], A_prev.shape[1], A_prev.shape[0]))
        # Redo right overlap correctly
        for i in range(n - 1, -1, -1):
            A_psi = tensors[i]; A_prev = prev_t[i]
            R_ov = P_R[j][i + 1]
            new_ov = np.zeros((A_psi.shape[0], A_prev.shape[0]), dtype=complex)
            for s in range(d):
                new_ov += np.conj(A_psi[:, s, :]) @ R_ov @ A_prev[:, s, :].T
            P_R[j][i] = new_ov

    L = np.ones((1, 1, 1), dtype=complex)

    # Left sweep
    for i in range(n - 1):
        A = tensors[i]; B = tensors[i + 1]
        chi_l = A.shape[0]; chi_m = A.shape[2]; chi_r = B.shape[2]
        W1 = W_tensors[i]; W2 = W_tensors[i + 1]
        R = R_envs[i + 2] if i + 2 <= n else np.ones((1, 1, 1), dtype=complex)

        theta = np.einsum('asb,bte->aste', A, B).reshape(-1)
        theta_dim = chi_l * d * d * chi_r

        # Build projected vectors for penalty
        proj_vecs = []
        for j in range(n_prev):
            prev_t = prev_tensors_list[j]
            Ap = prev_t[i]; Bp = prev_t[i + 1]
            phi = np.einsum('ab,asc,cte,df->sdf',
                            P_L[j][i], Ap, Bp, P_R[j][i + 2]).reshape(-1)
            # Correct: phi[al,s0,s1,ar] = P_L[al,bl]*Ap[bl,s0,bm]*Bp[bm,s1,br]*P_R[ar,br]
            phi = np.einsum('ab,bsc,cte,fd->asted',
                            P_L[j][i], Ap, Bp, P_R[j][i + 2])
            # shape: [chi_l_psi, d, d, chi_r_psi] via broadcasting
            # Actually need careful contraction
            Lp = P_L[j][i]  # [chi_l_psi, chi_l_prev]
            Rp = P_R[j][i + 2]  # [chi_r_psi, chi_r_prev]
            phi = np.einsum('ab,bsc,cte,fe->astf',
                            Lp, Ap, Bp, Rp).reshape(-1)
            proj_vecs.append(phi)

        def matvec(v, _L=L, _W1=W1, _W2=W2, _R=R, _cl=chi_l, _cr=chi_r, _pv=proj_vecs):
            out = _apply_heff_py(_L, _W1, _W2, _R, v, _cl, _cr)
            for phi in _pv:
                overlap = np.vdot(phi, v)
                out += weight * overlap * phi
            return out

        e, theta_opt = _lanczos_py(matvec, theta, theta_dim, lanczos_dim, 1e-10)
        energy = e

        # SVD split
        M = theta_opt.reshape(chi_l * d, d * chi_r)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        threshold = eps * S[0] if S[0] > 0 else eps
        keep = max(1, min(max_chi, int(np.sum(S > threshold))))
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep, :]
        tensors[i] = U.reshape(chi_l, d, keep)
        tensors[i + 1] = (np.diag(S) @ Vt).reshape(keep, d, chi_r)

        # Update L
        A_new = tensors[i]; W_cur = W_tensors[i]
        chi_r2 = A_new.shape[2]; wr = W_cur.shape[1]
        L_new = np.zeros((chi_r2, wr, chi_r2), dtype=complex)
        chi_l_L = L.shape[0]; wl = W_cur.shape[0]
        for a in range(chi_l_L):
            for wl_i in range(wl):
                for b in range(chi_l_L):
                    lval = L[a, wl_i, b]
                    if abs(lval) < 1e-15: continue
                    for sb in range(d):
                        for sk in range(d):
                            wval = W_cur[wl_i, :, sb, sk]
                            tmp = lval * np.outer(np.conj(A_new[a, sb, :]),
                                                  A_new[b, sk, :])
                            L_new += tmp[:, np.newaxis, :] * wval[np.newaxis, :, np.newaxis]
        L = L_new

        # Update overlap environments
        for j in range(n_prev):
            prev_t = prev_tensors_list[j]
            P_L[j][i + 1] = np.einsum('ab,asc,bsd->cd',
                                       P_L[j][i], np.conj(tensors[i]), prev_t[i])

    return float(energy)
