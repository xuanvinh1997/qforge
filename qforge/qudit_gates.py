# -*- coding: utf-8 -*-
# author: vinhpx
"""
Qudit gate library for qforge.

Provides single-qudit, controlled, and entangling gates for arbitrary
dimension d, with C++ backend dispatch and Python fallback.
Includes standard qutrit (d=3) gates based on Gell-Mann matrices.
"""
from __future__ import annotations
import numpy as np

from qforge import _HAS_CPP

# ============================================================
# Gell-Mann matrices (SU(3) generators for qutrits)
# ============================================================

GELL_MANN = [
    # lambda_1: off-diagonal real (0-1)
    np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]], dtype=complex),
    # lambda_2: off-diagonal imaginary (0-1)
    np.array([[0, -1j, 0],
              [1j, 0, 0],
              [0, 0, 0]], dtype=complex),
    # lambda_3: diagonal (0-1)
    np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, 0]], dtype=complex),
    # lambda_4: off-diagonal real (0-2)
    np.array([[0, 0, 1],
              [0, 0, 0],
              [1, 0, 0]], dtype=complex),
    # lambda_5: off-diagonal imaginary (0-2)
    np.array([[0, 0, -1j],
              [0, 0, 0],
              [1j, 0, 0]], dtype=complex),
    # lambda_6: off-diagonal real (1-2)
    np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]], dtype=complex),
    # lambda_7: off-diagonal imaginary (1-2)
    np.array([[0, 0, 0],
              [0, 0, -1j],
              [0, 1j, 0]], dtype=complex),
    # lambda_8: diagonal
    np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, -2]], dtype=complex) / np.sqrt(3),
]


# ============================================================
# Python fallback kernels
# ============================================================

def _apply_single_qudit_gate_py(wavefunction, target: int, gate: np.ndarray) -> None:
    """Apply a d×d gate to a single qudit (Python fallback)."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    n = len(states[0])
    d = _infer_dimension(wavefunction)
    dim = len(amplitude)
    gate = np.asarray(gate, dtype=complex).reshape(d, d)

    new_amplitude = np.zeros(dim, dtype=complex)
    stride = d ** (n - target - 1)

    for block_start in range(0, dim, d * stride):
        for offset in range(stride):
            indices = [block_start + s * stride + offset for s in range(d)]
            old_vals = np.array([amplitude[idx] for idx in indices])
            new_vals = gate @ old_vals
            for s in range(d):
                new_amplitude[indices[s]] = new_vals[s]

    wavefunction.amplitude = new_amplitude


def _infer_dimension(wavefunction) -> int:
    """Infer the local dimension d from the wavefunction."""
    if wavefunction._sv is not None:
        return wavefunction._sv.dimension
    # Infer from basis state labels
    states = wavefunction.state
    n = len(states[0])
    d = round(len(states) ** (1.0 / n))
    return d


# ============================================================
# Gate application dispatch
# ============================================================

def apply_qudit_gate(wavefunction, target: int, gate: np.ndarray) -> None:
    """Apply a d×d unitary gate to qudit at position `target`."""
    gate_flat = np.asarray(gate, dtype=complex).ravel()
    if wavefunction._sv is not None:
        wavefunction._sv.apply_single_qudit_gate(target, gate_flat)
    else:
        _apply_single_qudit_gate_py(wavefunction, target, gate_flat)


def apply_controlled_qudit_gate(wavefunction, control: int, ctrl_val: int,
                                 target: int, gate: np.ndarray) -> None:
    """Apply a d×d gate to `target`, conditioned on `control` == `ctrl_val`."""
    gate_flat = np.asarray(gate, dtype=complex).ravel()
    if wavefunction._sv is not None:
        wavefunction._sv.apply_controlled_qudit_gate(control, ctrl_val, target, gate_flat)
    else:
        # Python fallback for controlled gate
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        d = _infer_dimension(wavefunction)
        n = len(states[0])
        dim = len(amplitude)
        gate_mat = gate_flat.reshape(d, d)

        new_amplitude = amplitude.copy()
        ctrl_stride = d ** (n - control - 1)
        tgt_stride = d ** (n - target - 1)

        for idx in range(dim):
            cv = (idx // ctrl_stride) % d
            tv = (idx // tgt_stride) % d
            if cv != ctrl_val or tv != 0:
                continue
            indices = [idx + s * tgt_stride for s in range(d)]
            old_vals = np.array([amplitude[i] for i in indices])
            new_vals = gate_mat @ old_vals
            for s in range(d):
                new_amplitude[indices[s]] = new_vals[s]

        wavefunction.amplitude = new_amplitude


# ============================================================
# Standard qutrit gates (d=3)
# ============================================================

def _givens_gate(d: int, i: int, j: int, theta: float) -> np.ndarray:
    """Givens rotation in the (i,j) subspace: R_{ij}(theta).

    Acts as a 2D rotation between levels |i> and |j>,
    identity on all other levels.
    """
    gate = np.eye(d, dtype=complex)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    gate[i, i] = c
    gate[j, j] = c
    gate[i, j] = -s
    gate[j, i] = s
    return gate


def _phase_givens_gate(d: int, i: int, j: int, theta: float) -> np.ndarray:
    """Phase rotation in the (i,j) subspace with imaginary coupling.

    Like RY analog: exp(-i theta lambda_k / 2) for the imaginary off-diagonal
    Gell-Mann matrix connecting levels i and j.
    """
    gate = np.eye(d, dtype=complex)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    gate[i, i] = c
    gate[j, j] = c
    gate[i, j] = -1j * s
    gate[j, i] = -1j * s
    return gate


# --- Subspace swap gates (permutations between pairs of levels) ---

def X01(wavefunction, target: int) -> None:
    """Swap levels |0> and |1> (qutrit X gate for 0-1 subspace)."""
    d = _infer_dimension(wavefunction)
    gate = np.eye(d, dtype=complex)
    gate[0, 0] = 0; gate[0, 1] = 1
    gate[1, 0] = 1; gate[1, 1] = 0
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'X01'])


def X02(wavefunction, target: int) -> None:
    """Swap levels |0> and |2> (qutrit X gate for 0-2 subspace)."""
    d = _infer_dimension(wavefunction)
    gate = np.eye(d, dtype=complex)
    gate[0, 0] = 0; gate[0, 2] = 1
    gate[2, 0] = 1; gate[2, 2] = 0
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'X02'])


def X12(wavefunction, target: int) -> None:
    """Swap levels |1> and |2> (qutrit X gate for 1-2 subspace)."""
    d = _infer_dimension(wavefunction)
    gate = np.eye(d, dtype=complex)
    gate[1, 1] = 0; gate[1, 2] = 1
    gate[2, 1] = 1; gate[2, 2] = 0
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'X12'])


# --- Cyclic shift (increment mod d) ---

def CLOCK(wavefunction, target: int) -> None:
    """Cyclic shift gate: |k> -> |k+1 mod d>."""
    d = _infer_dimension(wavefunction)
    gate = np.zeros((d, d), dtype=complex)
    for k in range(d):
        gate[(k + 1) % d, k] = 1.0
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'CLK'])


# --- Ternary phase gate ---

def ZPHASE(wavefunction, target: int) -> None:
    """Ternary phase gate: diag(1, omega, omega^2) where omega = exp(2pi*i/d)."""
    d = _infer_dimension(wavefunction)
    omega = np.exp(2j * np.pi / d)
    gate = np.diag([omega**k for k in range(d)]).astype(complex)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'Zd'])


# --- Qutrit Hadamard (DFT matrix / d) ---

def Hd(wavefunction, target: int) -> None:
    """Qudit Hadamard (Discrete Fourier Transform / sqrt(d))."""
    d = _infer_dimension(wavefunction)
    omega = np.exp(2j * np.pi / d)
    gate = np.array([[omega**(j * k) for k in range(d)] for j in range(d)],
                    dtype=complex) / np.sqrt(d)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, 'Hd'])


# --- Parameterized rotations via Gell-Mann generators ---

def RGM(wavefunction, target: int, generator: int, angle: float) -> None:
    """Rotation by Gell-Mann generator: exp(-i * angle * lambda_k / 2).

    Args:
        target: Qudit index.
        generator: Gell-Mann matrix index (1-8 for qutrits).
        angle: Rotation angle in radians.
    """
    if generator < 1 or generator > len(GELL_MANN):
        raise ValueError(f"generator must be in [1, {len(GELL_MANN)}]")
    lam = GELL_MANN[generator - 1]
    # Matrix exponential: exp(-i * angle/2 * lambda)
    gate = _matrix_exp_hermitian(-1j * angle / 2.0 * lam)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, f'RGM{generator}', f'RGM{generator}({angle:.2f})'])


def _matrix_exp_hermitian(M: np.ndarray) -> np.ndarray:
    """Matrix exponential for Hermitian-like matrices via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(
        (M + M.conj().T) / 2.0  # ensure Hermitian for numerical stability
    )
    # But M might not be Hermitian (it's -i*angle/2 * lambda which is anti-Hermitian)
    # Use full eig instead
    eigvals, eigvecs = np.linalg.eig(M)
    exp_diag = np.diag(np.exp(eigvals))
    return eigvecs @ exp_diag @ np.linalg.inv(eigvecs)


# --- Givens rotation gates ---

def R01(wavefunction, target: int, theta: float) -> None:
    """Givens rotation in |0>-|1> subspace."""
    d = _infer_dimension(wavefunction)
    gate = _givens_gate(d, 0, 1, theta)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, f'R01({theta:.2f})', 'R01'])


def R02(wavefunction, target: int, theta: float) -> None:
    """Givens rotation in |0>-|2> subspace."""
    d = _infer_dimension(wavefunction)
    gate = _givens_gate(d, 0, 2, theta)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, f'R02({theta:.2f})', 'R02'])


def R12(wavefunction, target: int, theta: float) -> None:
    """Givens rotation in |1>-|2> subspace."""
    d = _infer_dimension(wavefunction)
    gate = _givens_gate(d, 1, 2, theta)
    apply_qudit_gate(wavefunction, target, gate)
    wavefunction.visual.append([target, f'R12({theta:.2f})', 'R12'])


# --- Entangling gates ---

def CSUM(wavefunction, control: int, target: int) -> None:
    """CSUM gate: |c, t> -> |c, (t + c) mod d>.

    This is the qutrit analog of CNOT.
    """
    if wavefunction._sv is not None:
        wavefunction._sv.csum(control, target)
    else:
        d = _infer_dimension(wavefunction)
        amplitude = wavefunction.amplitude.copy()
        n = len(wavefunction.state[0])
        dim = len(amplitude)
        ctrl_stride = d ** (n - control - 1)
        tgt_stride = d ** (n - target - 1)

        new_amplitude = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            cv = (idx // ctrl_stride) % d
            tv = (idx // tgt_stride) % d
            new_tv = (tv + cv) % d
            new_idx = idx + (new_tv - tv) * tgt_stride
            new_amplitude[new_idx] = amplitude[idx]
        wavefunction.amplitude = new_amplitude

    wavefunction.visual.append([control, target, 'CSUM'])


def QUDIT_SWAP(wavefunction, t1: int, t2: int) -> None:
    """SWAP two qudits."""
    if wavefunction._sv is not None:
        wavefunction._sv.qudit_swap(t1, t2)
    else:
        d = _infer_dimension(wavefunction)
        amplitude = wavefunction.amplitude.copy()
        n = len(wavefunction.state[0])
        dim = len(amplitude)
        s1 = d ** (n - t1 - 1)
        s2 = d ** (n - t2 - 1)

        new_amplitude = np.zeros(dim, dtype=complex)
        for idx in range(dim):
            v1 = (idx // s1) % d
            v2 = (idx // s2) % d
            new_idx = idx + (v2 - v1) * s1 + (v1 - v2) * s2
            new_amplitude[new_idx] = amplitude[idx]
        wavefunction.amplitude = new_amplitude

    wavefunction.visual.append([t1, t2, 'QSWAP'])


# ============================================================
# Qudit measurement
# ============================================================

def measure_qudit(wavefunction, qudit: int) -> np.ndarray:
    """Return probabilities [P(0), P(1), ..., P(d-1)] for a single qudit."""
    if wavefunction._sv is not None:
        probs = wavefunction._sv.measure_qudit_probs(qudit)
        return np.array(probs)

    d = _infer_dimension(wavefunction)
    n = len(wavefunction.state[0])
    dim = len(wavefunction.amplitude)
    stride = d ** (n - qudit - 1)
    amplitude = wavefunction.amplitude

    probs = np.zeros(d)
    for i in range(dim):
        val = (i // stride) % d
        probs[val] += abs(amplitude[i]) ** 2
    return probs


def collapse_qudit(wavefunction, qudit: int) -> int:
    """Collapse a qudit via projective measurement. Returns the measured value."""
    probs = measure_qudit(wavefunction, qudit)
    probs = probs / probs.sum()  # normalize for numerical safety
    d = len(probs)
    result = int(np.random.choice(d, p=probs))

    if wavefunction._sv is not None:
        wavefunction._sv.collapse_qudit(qudit, result)
    else:
        amplitude = wavefunction.amplitude
        n = len(wavefunction.state[0])
        dim = len(amplitude)
        stride = d ** (n - qudit - 1)
        import math
        prob = probs[result]
        norm = 1.0 / math.sqrt(prob) if prob > 1e-15 else 0.0

        new_amp = np.zeros(dim, dtype=complex)
        for i in range(dim):
            val = (i // stride) % d
            if val == result:
                new_amp[i] = amplitude[i] * norm
        wavefunction.amplitude = new_amp

    return result


def qudit_expectation(wavefunction, qudit: int, operator: np.ndarray) -> complex:
    """Expectation value of a d×d operator on a single qudit."""
    op_flat = np.asarray(operator, dtype=complex).ravel()
    if wavefunction._sv is not None:
        return wavefunction._sv.qudit_expectation(qudit, op_flat)

    d = _infer_dimension(wavefunction)
    op = op_flat.reshape(d, d)
    n = len(wavefunction.state[0])
    dim = len(wavefunction.amplitude)
    stride = d ** (n - qudit - 1)
    amplitude = wavefunction.amplitude

    result = 0.0 + 0.0j
    for idx in range(dim):
        val = (idx // stride) % d
        if val != 0:
            continue
        for row in range(d):
            idx_row = idx + row * stride
            for col in range(d):
                idx_col = idx + col * stride
                result += np.conj(amplitude[idx_row]) * op[row, col] * amplitude[idx_col]
    return result
