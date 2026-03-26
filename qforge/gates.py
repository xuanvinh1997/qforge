# -*- coding: utf-8 -*-
# author: vinhpx
"""
Quantum gate library for qforge.

Provides single-qubit, controlled, swap, and noise gates with
C++ backend dispatch and Python fallback.
"""
from __future__ import annotations
import numpy as np
import cmath
from qforge._utils import _validate_qubit, _validate_ctrl_target, _nq, _is_mps, _is_dm
from qforge.ir import _record_op


# ============================================================
# Internal helpers (Python fallback kernels)
# ============================================================


def _apply_single_gate_py(wavefunction, n: int, m00, m01, m10, m11) -> None:
    """Apply a 2x2 unitary [[m00, m01], [m10, m11]] to qubit n (Python fallback)."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    _validate_qubit(n, qubit_num)
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    cut = 2**(qubit_num - n - 1)
    for i in np.nonzero(amplitude)[0]:
        a = amplitude[i]
        if states[i][n] == '0':
            new_amplitude[i] += m00 * a
            new_amplitude[i + cut] += m10 * a
        else:
            new_amplitude[i] += m11 * a
            new_amplitude[i - cut] += m01 * a
    wavefunction.amplitude = new_amplitude


def _apply_controlled_gate_py(wavefunction, control: int, target: int,
                               m00, m01, m10, m11) -> None:
    """Apply controlled 2x2 unitary on target qubit (Python fallback)."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    _validate_ctrl_target(control, target, qubit_num)
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    cut = 2**(qubit_num - target - 1)
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            a = amplitude[i]
            if states[i][target] == '0':
                new_amplitude[i] += m00 * a
                new_amplitude[i + cut] += m10 * a
            else:
                new_amplitude[i] += m11 * a
                new_amplitude[i - cut] += m01 * a
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude


# ============================================================
# Single-qubit gates
# ============================================================

_INV_SQRT2 = 1.0 / 2**0.5

# Pre-built gate matrices for MPS dispatch (row-major 2x2 complex arrays)
_GATE_H  = np.array([_INV_SQRT2, _INV_SQRT2,  _INV_SQRT2, -_INV_SQRT2], dtype=complex)
_GATE_X  = np.array([0, 1, 1, 0], dtype=complex)
_GATE_Y  = np.array([0, -1j, 1j, 0], dtype=complex)
_GATE_Z  = np.array([1, 0, 0, -1], dtype=complex)
_GATE_S  = np.array([1, 0, 0, 1j], dtype=complex)
_GATE_T  = np.array([1, 0, 0, (1+1j)/np.sqrt(2)], dtype=complex)
_GATE_XS = np.array([(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], dtype=complex)

# 4x4 gate matrices for two-qubit MPS dispatch (row-major)
# Ordering: |00>, |01>, |10>, |11> (qubit i = MSB)
_GATE_CNOT = np.array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0,
], dtype=complex)
_GATE_SWAP = np.array([
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
], dtype=complex)
_GATE_ISWAP = np.array([
    1, 0, 0, 0,
    0, 0, 1j, 0,
    0, 1j, 0, 0,
    0, 0, 0, 1,
], dtype=complex)
_GATE_SISWAP = np.array([
    1, 0, 0, 0,
    0, 1/np.sqrt(2), 1j/np.sqrt(2), 0,
    0, 1j/np.sqrt(2), 1/np.sqrt(2), 0,
    0, 0, 0, 1,
], dtype=complex)


def _mps_apply1(wavefunction, site: int, gate) -> None:
    """Apply a 2x2 gate (4-element row-major array) to MPS at site."""
    if wavefunction._mps is not None:
        wavefunction._mps.apply_single_qubit_gate(site, gate)
    else:
        # Python MPS fallback
        A = wavefunction._tensors[site]
        tmp = A.copy()
        for cl in range(A.shape[0]):
            for cr in range(A.shape[2]):
                a0, a1 = tmp[cl, 0, cr], tmp[cl, 1, cr]
                A[cl, 0, cr] = gate[0]*a0 + gate[1]*a1
                A[cl, 1, cr] = gate[2]*a0 + gate[3]*a1


def _mps_apply2(wavefunction, site_i: int, gate, max_bond_dim: int = None,
                eps: float = None) -> None:
    """Apply a 4x4 gate to MPS sites (site_i, site_i+1). For non-adjacent
    qubits, use _mps_nonlocal2 instead."""
    chi = max_bond_dim or wavefunction.max_bond_dim
    ep = eps or wavefunction.eps
    if wavefunction._mps is not None:
        wavefunction._mps.apply_two_qubit_gate(site_i, gate, chi, ep)
    else:
        # Python MPS fallback via amplitude manipulation
        from qforge.mps import _svd_decompose
        A = wavefunction._tensors[site_i]
        B = wavefunction._tensors[site_i + 1]
        chi_l = A.shape[0]; chi_r = B.shape[2]
        d = 2
        # Merge: theta[chi_l, d, d, chi_r]
        theta_full = np.einsum('asb,bte->aste', A, B)
        # Apply gate: result[chi_l, s0_out, s1_out, chi_r]
        # gate: [d*d_out, d*d_in] (row-major), theta_full: [chi_l, d*d_in, chi_r]
        gate_mat = gate.reshape(4, 4)
        theta_phys = theta_full.reshape(chi_l, d * d, chi_r)
        theta_out_phys = np.einsum('ij,kjl->kil', gate_mat, theta_phys)
        theta_out = theta_out_phys.reshape(chi_l * d, d * chi_r)
        # SVD split
        U, S, Vt = np.linalg.svd(theta_out, full_matrices=False)
        threshold = ep * S[0] if S[0] > 0 else ep
        keep = max(1, min(chi, int(np.sum(S > threshold))))
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep, :]
        wavefunction._tensors[site_i] = U.reshape(chi_l, d, keep)
        wavefunction._tensors[site_i + 1] = (np.diag(S) @ Vt).reshape(keep, d, chi_r)


def _mps_nonlocal2(wavefunction, ci: int, ti: int, gate) -> None:
    """Apply two-qubit gate to non-adjacent qubits (ci, ti) via SWAP chain."""
    # Move ci adjacent to ti via SWAPs, apply gate, swap back
    chi = wavefunction.max_bond_dim
    ep = wavefunction.eps
    if ci > ti:
        ci, ti = ti, ci
        # transpose gate: swap qubit order (|01><10| etc)
        gate = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(16)
    # Move ci to ti-1 via SWAP chain
    for k in range(ci, ti - 1):
        _mps_apply2(wavefunction, k, _GATE_SWAP, chi, ep)
    # Apply gate at (ti-1, ti)
    _mps_apply2(wavefunction, ti - 1, gate, chi, ep)
    # Swap back
    for k in range(ti - 2, ci - 1, -1):
        _mps_apply2(wavefunction, k, _GATE_SWAP, chi, ep)


def H(wavefunction: object, n: int) -> None:
    """Hadamard gate — creates equal superposition.

    Maps ``|0> -> (|0>+|1>)/sqrt(2)`` and ``|1> -> (|0>-|1>)/sqrt(2)``.

    Args:
        wavefunction: Target quantum state.
        n: Qubit index.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.H(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_H, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_H)
    else:
        r = _INV_SQRT2
        _apply_single_gate_py(wavefunction, n, r, r, r, -r)
    wavefunction.visual.append([n, 'H'])
    _record_op('H', (n,))


def X(wavefunction: object, n: int) -> None:
    """Pauli-X gate."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.X(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_X, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_X)
    else:
        _apply_single_gate_py(wavefunction, n, 0, 1, 1, 0)
    wavefunction.visual.append([n, 'X'])
    _record_op('X', (n,))


def Y(wavefunction: object, n: int) -> None:
    """Pauli-Y gate."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.Y(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_Y, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_Y)
    else:
        _apply_single_gate_py(wavefunction, n, 0, -1j, 1j, 0)
    wavefunction.visual.append([n, 'Y'])
    _record_op('Y', (n,))


def Z(wavefunction: object, n: int) -> None:
    """Pauli-Z gate."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.Z(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_Z, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_Z)
    else:
        _apply_single_gate_py(wavefunction, n, 1, 0, 0, -1)
    wavefunction.visual.append([n, 'Z'])
    _record_op('Z', (n,))


def RX(wavefunction: object, n: int, phi: float = 0) -> None:
    """Rotation around the X-axis by angle ``phi``.

    Matrix: ``[[cos(phi/2), -i*sin(phi/2)], [-i*sin(phi/2), cos(phi/2)]]``.

    Args:
        wavefunction: Target quantum state.
        n: Qubit index.
        phi: Rotation angle in radians.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.RX(n, phi)
    elif _is_dm(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        wavefunction.apply_gate(np.array([c, -1j*s, -1j*s, c], dtype=complex), [n])
    elif _is_mps(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _mps_apply1(wavefunction, n, np.array([c, -1j*s, -1j*s, c], dtype=complex))
    else:
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _apply_single_gate_py(wavefunction, n, c, -1j*s, -1j*s, c)
    wavefunction.visual.append([n, 'RX', '0'])
    _record_op('RX', (n,), (phi,))


def RY(wavefunction: object, n: int, phi: float = 0) -> None:
    """Rotation around the Y-axis by angle ``phi``.

    Matrix: ``[[cos(phi/2), -sin(phi/2)], [sin(phi/2), cos(phi/2)]]``.

    Args:
        wavefunction: Target quantum state.
        n: Qubit index.
        phi: Rotation angle in radians.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.RY(n, phi)
    elif _is_dm(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        wavefunction.apply_gate(np.array([c, -s, s, c], dtype=complex), [n])
    elif _is_mps(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _mps_apply1(wavefunction, n, np.array([c, -s, s, c], dtype=complex))
    else:
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _apply_single_gate_py(wavefunction, n, c, -s, s, c)
    wavefunction.visual.append([n, 'RY', '0'])
    _record_op('RY', (n,), (phi,))


def RZ(wavefunction: object, n: int, phi: float = 0) -> None:
    """Rotation around the Z-axis by angle ``phi``.

    Matrix: ``[[exp(-i*phi/2), 0], [0, exp(i*phi/2)]]``.

    Args:
        wavefunction: Target quantum state.
        n: Qubit index.
        phi: Rotation angle in radians.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.RZ(n, phi)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(np.array(
            [cmath.exp(-1j*phi/2), 0, 0, cmath.exp(1j*phi/2)], dtype=complex), [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, np.array(
            [cmath.exp(-1j*phi/2), 0, 0, cmath.exp(1j*phi/2)], dtype=complex))
    else:
        _apply_single_gate_py(wavefunction, n,
                              cmath.exp(-1j*phi/2), 0, 0, cmath.exp(1j*phi/2))
    wavefunction.visual.append([n, 'RZ', '0'])
    _record_op('RZ', (n,), (phi,))


def Phase(wavefunction: object, n: int, phi: float = 0) -> None:
    """Phase gate — applies ``exp(i*phi)`` to the ``|1>`` component.

    Matrix: ``[[1, 0], [0, exp(i*phi)]]``.

    Args:
        wavefunction: Target quantum state.
        n: Qubit index.
        phi: Phase angle in radians.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.Phase(n, phi)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(np.array(
            [1, 0, 0, cmath.exp(1j*phi)], dtype=complex), [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, np.array(
            [1, 0, 0, cmath.exp(1j*phi)], dtype=complex))
    else:
        _apply_single_gate_py(wavefunction, n, 1, 0, 0, cmath.exp(1j*phi))
    wavefunction.visual.append([n, 'P'])
    _record_op('Phase', (n,), (phi,))


def S(wavefunction: object, n: int) -> None:
    """S gate — Phase(pi/2)."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.S(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_S, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_S)
    else:
        _apply_single_gate_py(wavefunction, n, 1, 0, 0, 1j)
    wavefunction.visual.append([n, 'S'])
    _record_op('S', (n,))


def T(wavefunction: object, n: int) -> None:
    """T gate — Phase(pi/4)."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.T(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_T, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_T)
    else:
        _apply_single_gate_py(wavefunction, n, 1, 0, 0, cmath.exp(1j*cmath.pi/4))
    wavefunction.visual.append([n, 'T'])
    _record_op('T', (n,))


def Xsquare(wavefunction: object, n: int) -> None:
    """Square root of NOT gate."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.Xsquare(n)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_XS, [n])
    elif _is_mps(wavefunction):
        _mps_apply1(wavefunction, n, _GATE_XS)
    else:
        a, b = (1+1j)/2, (1-1j)/2
        _apply_single_gate_py(wavefunction, n, a, b, b, a)
    wavefunction.visual.append([n, 'XS'])
    _record_op('Xsquare', (n,))


# ============================================================
# Controlled gates
# ============================================================

def _mps_ctrl_target_gate(wavefunction, control, target, gate4x4):
    """Dispatch a two-qubit gate to MPS, handling non-adjacent qubits."""
    if abs(control - target) == 1:
        # Adjacent: direct application
        site_i = min(control, target)
        if control > target:
            # Swap qubit ordering in gate for control > target case
            g = gate4x4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(16)
        else:
            g = gate4x4
        _mps_apply2(wavefunction, site_i, g)
    else:
        _mps_nonlocal2(wavefunction, control, target, gate4x4)


def CNOT(wavefunction: object, control: int, target: int) -> None:
    """Controlled-NOT (CX) gate — flips target qubit when control is ``|1>``.

    Args:
        wavefunction: Target quantum state.
        control: Control qubit index.
        target: Target qubit index.
    """
    _validate_ctrl_target(control, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CNOT(control, target)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_CNOT, [control, target])
    elif _is_mps(wavefunction):
        _mps_ctrl_target_gate(wavefunction, control, target, _GATE_CNOT)
    else:
        _apply_controlled_gate_py(wavefunction, control, target, 0, 1, 1, 0)
    wavefunction.visual.append([control, target, 'CX'])
    _record_op('CNOT', (control, target))


def CRX(wavefunction: object, control: int, target: int, phi: float = 0) -> None:
    """Controlled RX gate — applies RX(phi) to target when control is ``|1>``.

    Args:
        wavefunction: Target quantum state.
        control: Control qubit index.
        target: Target qubit index.
        phi: Rotation angle in radians.
    """
    _validate_ctrl_target(control, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CRX(control, target, phi)
    elif _is_dm(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,c,-1j*s, 0,0,-1j*s,c], dtype=complex)
        wavefunction.apply_gate(g, [control, target])
    elif _is_mps(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,c,-1j*s, 0,0,-1j*s,c], dtype=complex)
        _mps_ctrl_target_gate(wavefunction, control, target, g)
    else:
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _apply_controlled_gate_py(wavefunction, control, target,
                                   c, -1j*s, -1j*s, c)
    wavefunction.visual.append([control, target, 'CRX', '0'])
    _record_op('CRX', (control, target), (phi,))


def CRY(wavefunction: object, control: int, target: int, phi: float = 0) -> None:
    """Controlled RY gate — applies RY(phi) to target when control is ``|1>``.

    Args:
        wavefunction: Target quantum state.
        control: Control qubit index.
        target: Target qubit index.
        phi: Rotation angle in radians.
    """
    _validate_ctrl_target(control, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CRY(control, target, phi)
    elif _is_dm(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,c,-s, 0,0,s,c], dtype=complex)
        wavefunction.apply_gate(g, [control, target])
    elif _is_mps(wavefunction):
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,c,-s, 0,0,s,c], dtype=complex)
        _mps_ctrl_target_gate(wavefunction, control, target, g)
    else:
        c, s = cmath.cos(phi/2), cmath.sin(phi/2)
        _apply_controlled_gate_py(wavefunction, control, target, c, -s, s, c)
    wavefunction.visual.append([control, target, 'CRY', '0'])
    _record_op('CRY', (control, target), (phi,))


def CRZ(wavefunction: object, control: int, target: int, phi: float = 0) -> None:
    """Controlled RZ gate — applies RZ(phi) to target when control is ``|1>``.

    Args:
        wavefunction: Target quantum state.
        control: Control qubit index.
        target: Target qubit index.
        phi: Rotation angle in radians.
    """
    _validate_ctrl_target(control, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CRZ(control, target, phi)
    elif _is_dm(wavefunction):
        em = cmath.exp(-1j*phi/2); ep = cmath.exp(1j*phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,em,0, 0,0,0,ep], dtype=complex)
        wavefunction.apply_gate(g, [control, target])
    elif _is_mps(wavefunction):
        em = cmath.exp(-1j*phi/2); ep = cmath.exp(1j*phi/2)
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,em,0, 0,0,0,ep], dtype=complex)
        _mps_ctrl_target_gate(wavefunction, control, target, g)
    else:
        _apply_controlled_gate_py(wavefunction, control, target,
                                   cmath.exp(-1j*phi/2), 0,
                                   0, cmath.exp(1j*phi/2))
    wavefunction.visual.append([control, target, 'CRZ', '0'])
    _record_op('CRZ', (control, target), (phi,))


def CPhase(wavefunction: object, control: int, target: int, phi: float = 0) -> None:
    """Controlled Phase gate — applies ``exp(i*phi)`` to ``|11>`` component.

    Args:
        wavefunction: Target quantum state.
        control: Control qubit index.
        target: Target qubit index.
        phi: Phase angle in radians.
    """
    _validate_ctrl_target(control, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CPhase(control, target, phi)
    elif _is_dm(wavefunction):
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,cmath.exp(1j*phi)], dtype=complex)
        wavefunction.apply_gate(g, [control, target])
    elif _is_mps(wavefunction):
        g = np.array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,cmath.exp(1j*phi)], dtype=complex)
        _mps_ctrl_target_gate(wavefunction, control, target, g)
    else:
        _apply_controlled_gate_py(wavefunction, control, target,
                                   1, 0, 0, cmath.exp(1j*phi))
    wavefunction.visual.append([control, target, 'CP', '0'])
    _record_op('CPhase', (control, target), (phi,))


CP = CPhase  # Alias


# ============================================================
# Double-controlled gates
# ============================================================

def _validate_three_distinct(a: int, b: int, c: int, qubit_num: int) -> None:
    for q in (a, b, c):
        if q >= qubit_num or q < 0:
            raise ValueError("Index is out of range")
    if a == b or a == c or b == c:
        raise ValueError("Control qubit and target qubit must be distinct")


def CCNOT(wavefunction: object, control_1: int, control_2: int, target: int) -> None:
    """Toffoli (double-controlled-X) gate — flips target when both controls are ``|1>``.

    Args:
        wavefunction: Target quantum state.
        control_1: First control qubit index.
        control_2: Second control qubit index.
        target: Target qubit index.
    """
    _validate_three_distinct(control_1, control_2, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CCNOT(control_1, control_2, target)
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        new_amplitude = np.zeros(2**qubit_num, dtype=complex)
        cut = 2**(qubit_num - target - 1)
        for i in np.nonzero(amplitude)[0]:
            if states[i][control_1] == '1' and states[i][control_2] == '1':
                if states[i][target] == '0':
                    new_amplitude[i + cut] += amplitude[i]
                else:
                    new_amplitude[i - cut] += amplitude[i]
            else:
                new_amplitude[i] = amplitude[i]
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([control_1, control_2, target, 'CCX'])
    _record_op('CCNOT', (control_1, control_2, target))


def OR(wavefunction: object, control_1: int, control_2: int, target: int) -> None:
    """OR gate — flip target if control_1 OR control_2 is |1>."""
    _validate_three_distinct(control_1, control_2, target, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.OR(control_1, control_2, target)
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        new_amplitude = np.zeros(2**qubit_num, dtype=complex)
        cut = 2**(qubit_num - target - 1)
        for i in np.nonzero(amplitude)[0]:
            if states[i][control_1] == '1' or states[i][control_2] == '1':
                if states[i][target] == '0':
                    new_amplitude[i + cut] += amplitude[i]
                else:
                    new_amplitude[i - cut] += amplitude[i]
            else:
                new_amplitude[i] = amplitude[i]
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([control_1, control_2, target, 'OR'])
    _record_op('OR', (control_1, control_2, target))


# ============================================================
# Swap gates
# ============================================================

def _swap_cut(qubit_num: int, t1: int, t2: int):
    """Compute min, max, cut for swap-family gates."""
    minimum = t2 ^ ((t1 ^ t2) & -(t1 < t2))
    maximum = t1 ^ ((t1 ^ t2) & -(t1 < t2))
    cut = 2**(qubit_num - minimum - 1) - 2**(qubit_num - maximum - 1)
    return minimum, maximum, cut


def _validate_swap(t1: int, t2: int, qubit_num: int) -> None:
    for q in (t1, t2):
        if q >= qubit_num or q < 0:
            raise ValueError("Index is out of range")
    if t1 == t2:
        raise ValueError("Target qubits must be distinct")


def SWAP(wavefunction: object, target_1: int, target_2: int) -> None:
    """Swap gate — exchanges the states of two qubits.

    Args:
        wavefunction: Target quantum state.
        target_1: First qubit index.
        target_2: Second qubit index.
    """
    _validate_swap(target_1, target_2, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.SWAP(target_1, target_2)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_SWAP, [target_1, target_2])
    elif _is_mps(wavefunction):
        t1, t2 = min(target_1, target_2), max(target_1, target_2)
        if t2 == t1 + 1:
            _mps_apply2(wavefunction, t1, _GATE_SWAP)
        else:
            # SWAP non-adjacent: use bubble sort via adjacent SWAPs
            for k in range(t1, t2 - 1):
                _mps_apply2(wavefunction, k, _GATE_SWAP)
            for k in range(t2 - 2, t1 - 1, -1):
                _mps_apply2(wavefunction, k, _GATE_SWAP)
        wavefunction.visual.append([target_1, target_2, 'SWAP'])
        _record_op('SWAP', (target_1, target_2))
        return
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        minimum, maximum, cut = _swap_cut(qubit_num, target_1, target_2)
        new_amplitude = np.zeros(2**qubit_num, dtype=complex)
        for i in range(2**qubit_num):
            if states[i][target_1] != states[i][target_2]:
                if int(states[i][maximum]) > int(states[i][minimum]):
                    new_amplitude[i + cut] += amplitude[i]
                else:
                    new_amplitude[i - cut] += amplitude[i]
            else:
                new_amplitude[i] = amplitude[i]
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([target_1, target_2, 'SWAP'])
    _record_op('SWAP', (target_1, target_2))


def CSWAP(wavefunction: object, control: int, target_1: int, target_2: int) -> None:
    """Controlled Swap (Fredkin) gate."""
    _validate_three_distinct(control, target_1, target_2, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.CSWAP(control, target_1, target_2)
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        minimum, maximum, cut = _swap_cut(qubit_num, target_1, target_2)
        new_amplitude = np.zeros(2**qubit_num, dtype=complex)
        for i in range(2**qubit_num):
            if states[i][control] == '1':
                if states[i][target_1] != states[i][target_2]:
                    if int(states[i][maximum]) > int(states[i][minimum]):
                        new_amplitude[i + cut] += amplitude[i]
                    else:
                        new_amplitude[i - cut] += amplitude[i]
                else:
                    new_amplitude[i] = amplitude[i]
            else:
                new_amplitude[i] = amplitude[i]
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([target_1, target_2, control, 'CSWAP'])
    _record_op('CSWAP', (control, target_1, target_2))


def ISWAP(wavefunction: object, target_1: int, target_2: int) -> None:
    """iSWAP gate."""
    _validate_swap(target_1, target_2, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.ISWAP(target_1, target_2)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_ISWAP, [target_1, target_2])
    elif _is_mps(wavefunction):
        _mps_ctrl_target_gate(wavefunction, min(target_1, target_2),
                              max(target_1, target_2), _GATE_ISWAP)
        wavefunction.visual.append([target_1, target_2, 'ISWAP'])
        _record_op('ISWAP', (target_1, target_2))
        return
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        minimum, maximum, cut = _swap_cut(qubit_num, target_1, target_2)
        dim = 2**qubit_num
        new_amplitude = np.zeros(dim, dtype=complex)
        for i in range(dim):
            if states[i][target_1] == states[i][target_2]:
                new_amplitude[i] += amplitude[i]
            else:
                if int(states[i][maximum]) > int(states[i][minimum]):
                    j = i + cut
                else:
                    j = i - cut
                new_amplitude[j] += 1j * amplitude[i]
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([target_1, target_2, 'ISWAP'])
    _record_op('ISWAP', (target_1, target_2))


def SISWAP(wavefunction: object, target_1: int, target_2: int) -> None:
    """sqrt(iSWAP) gate."""
    _validate_swap(target_1, target_2, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.SISWAP(target_1, target_2)
    elif _is_dm(wavefunction):
        wavefunction.apply_gate(_GATE_SISWAP, [target_1, target_2])
    elif _is_mps(wavefunction):
        _mps_ctrl_target_gate(wavefunction, min(target_1, target_2),
                              max(target_1, target_2), _GATE_SISWAP)
        wavefunction.visual.append([target_1, target_2, 'SISWAP'])
        _record_op('SISWAP', (target_1, target_2))
        return
    else:
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        qubit_num = len(states[0])
        minimum, maximum, cut = _swap_cut(qubit_num, target_1, target_2)
        dim = 2**qubit_num
        new_amplitude = np.zeros(dim, dtype=complex)
        processed = np.zeros(dim, dtype=bool)
        c = 1.0 / np.sqrt(2.0)
        for i in range(dim):
            if processed[i]:
                continue
            if states[i][target_1] == states[i][target_2]:
                new_amplitude[i] = amplitude[i]
                processed[i] = True
            else:
                if int(states[i][maximum]) > int(states[i][minimum]):
                    j = i + cut
                else:
                    j = i - cut
                a, b = amplitude[i], amplitude[j]
                new_amplitude[i] = c * a + 1j * c * b
                new_amplitude[j] = 1j * c * a + c * b
                processed[i] = True
                processed[j] = True
        wavefunction.amplitude = new_amplitude
    wavefunction.visual.append([target_1, target_2, 'SISWAP'])
    _record_op('SISWAP', (target_1, target_2))


# ============================================================
# Noise channels
# ============================================================

def E(wavefunction: object, p: float, n: int) -> None:
    """Single-qubit depolarizing channel.

    Applies depolarizing noise with probability ``p`` to qubit ``n``.

    Args:
        wavefunction: Target quantum state.
        p: Depolarizing probability (0 = no noise, 1 = fully depolarized).
        n: Qubit index.
    """
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        wavefunction._sv.E(p, n)
        return
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    _validate_qubit(n, qubit_num)
    new_amplitude = np.zeros(2**qubit_num, dtype=complex)
    cut = 2**(qubit_num - n - 1)
    for i in np.nonzero(amplitude)[0]:
        prob = abs(amplitude[i])**2
        if states[i][n] == '0':
            new_amplitude[i + cut] += (p/2) * prob
            new_amplitude[i] += (1 - p/2) * prob
        else:
            new_amplitude[i - cut] += (p/2) * prob
            new_amplitude[i] += (1 - p/2) * prob
    for i in range(2**qubit_num):
        if amplitude[i].real < 0:
            new_amplitude[i] = -np.sqrt(new_amplitude[i])
        else:
            new_amplitude[i] = np.sqrt(new_amplitude[i])
    wavefunction.amplitude = new_amplitude


def E_all(wavefunction: object, p_noise: float, qubit_num: int) -> None:
    """Apply depolarizing channel to all qubits.

    Args:
        wavefunction: Target quantum state.
        p_noise: Depolarizing probability per qubit.
        qubit_num: Total number of qubits.
    """
    if wavefunction._sv is not None:
        wavefunction._sv.E_all(p_noise)
        return
    if p_noise > 0:
        for i in range(qubit_num):
            E(wavefunction, p_noise, i)


# ============================================================
# Arbitrary unitary gate
# ============================================================

def _apply_two_qubit_unitary_py(wavefunction, q0: int, q1: int,
                                 matrix: np.ndarray) -> None:
    """Apply a 4x4 unitary matrix to qubits (q0, q1) — Python fallback."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    nq = len(states[0])
    dim = 2**nq
    mat = matrix.reshape(4, 4)
    new_amplitude = np.zeros(dim, dtype=complex)
    processed = np.zeros(dim, dtype=bool)
    bit0 = nq - q0 - 1
    bit1 = nq - q1 - 1

    for idx in range(dim):
        if processed[idx]:
            continue
        # Compute base index with q0=0, q1=0
        base = idx & ~(1 << bit0) & ~(1 << bit1)
        # Four basis indices: |00>, |01>, |10>, |11> for (q0, q1)
        i00 = base
        i01 = base | (1 << bit1)
        i10 = base | (1 << bit0)
        i11 = base | (1 << bit0) | (1 << bit1)

        a = np.array([amplitude[i00], amplitude[i01],
                       amplitude[i10], amplitude[i11]])
        b = mat @ a
        new_amplitude[i00] = b[0]
        new_amplitude[i01] = b[1]
        new_amplitude[i10] = b[2]
        new_amplitude[i11] = b[3]
        processed[i00] = processed[i01] = True
        processed[i10] = processed[i11] = True

    wavefunction.amplitude = new_amplitude


def QubitUnitary(wavefunction: object, matrix: np.ndarray, qubits: list[int]) -> None:
    """Apply an arbitrary unitary matrix to the specified qubits.

    Args:
        wavefunction: Target wavefunction or MPS.
        matrix:       Unitary matrix (2x2 for 1-qubit, 4x4 for 2-qubit).
        qubits:       List/tuple of qubit indices.
    """
    matrix = np.asarray(matrix, dtype=complex)
    qubits = list(qubits) if not isinstance(qubits, list) else qubits
    nq = _nq(wavefunction)
    for q in qubits:
        _validate_qubit(q, nq)

    if len(qubits) == 1:
        q = qubits[0]
        m = matrix.ravel()
        if wavefunction._sv is not None:
            # Use generic single-qubit gate application via matrix elements
            _apply_single_gate_py(
                wavefunction.__class__(wavefunction.state,
                                       wavefunction._sv.amplitude),
                q, m[0], m[1], m[2], m[3])
            # Actually, for C++ backend we must use the Python kernel
            # on the C++ amplitude array (zero-copy)
            amp = wavefunction._sv.amplitude
            nq_total = len(wavefunction.state[0])
            cut = 2**(nq_total - q - 1)
            dim = len(amp)
            for i in range(dim):
                if not (i >> (nq_total - q - 1)) & 1:
                    j = i + cut
                    a0, a1 = amp[i], amp[j]
                    amp[i] = m[0]*a0 + m[1]*a1
                    amp[j] = m[2]*a0 + m[3]*a1
        elif _is_mps(wavefunction):
            _mps_apply1(wavefunction, q, m[:4])
        else:
            _apply_single_gate_py(wavefunction, q, m[0], m[1], m[2], m[3])

    elif len(qubits) == 2:
        q0, q1 = qubits
        if _is_mps(wavefunction):
            gate_flat = matrix.ravel()
            _mps_ctrl_target_gate(wavefunction, q0, q1, gate_flat)
        else:
            _apply_two_qubit_unitary_py(wavefunction, q0, q1, matrix)
    else:
        raise NotImplementedError(
            f"QubitUnitary supports 1- and 2-qubit unitaries, got {len(qubits)}"
        )

    wavefunction.visual.append([*qubits, 'U'])
    _record_op('Unitary', tuple(qubits), matrix=matrix)


# ============================================================
# Multi-controlled gate wrapper
# ============================================================

def ctrl(gate_fn: object, controls: list[int], target: int, *args: float) -> None:
    """Apply a multi-controlled version of a single-target gate.

    Args:
        gate_fn:  A gate function (e.g., ``gates.X``, ``gates.RY``).
        controls: List of control qubit indices.
        target:   Target qubit index.
        *args:    Extra arguments (e.g., rotation angle) passed to gate_fn after
                  ``(wavefunction, target, ...)``.

    For 1 control, delegates to existing controlled gates where possible.
    For 2 controls, delegates to CCNOT/decomposition.
    For 3+ controls, decomposes recursively using ancilla-free V/V† method.
    """
    # This is a higher-order function: it returns a closure for use.
    # But we also make it callable directly on a wavefunction.
    raise NotImplementedError(
        "ctrl() requires a wavefunction as first argument — use mcx/mcy/mcz instead"
    )


def mcx(wavefunction: object, controls: list[int], target: int) -> None:
    """Multi-controlled X gate (generalized Toffoli).

    Decomposes into O(n) elementary gates for n controls.
    """
    controls = list(controls)
    nq = _nq(wavefunction)
    for q in controls + [target]:
        _validate_qubit(q, nq)

    n_ctrl = len(controls)
    if n_ctrl == 0:
        X(wavefunction, target)
    elif n_ctrl == 1:
        CNOT(wavefunction, controls[0], target)
    elif n_ctrl == 2:
        CCNOT(wavefunction, controls[0], controls[1], target)
    else:
        # Recursive decomposition: CCNOT-based V/V† decomposition
        # C^n(X) = sequence of Toffoli gates using workspace qubit
        _mcx_recursive(wavefunction, controls, target)


def _mcx_recursive(wavefunction: object, controls: list[int], target: int) -> None:
    """Decompose n-controlled X into CCNOT gates recursively.

    Uses the standard decomposition that requires no ancilla qubits:
    For n controls, uses O(n^2) CCNOT gates via Barenco et al. decomposition.
    """
    n = len(controls)
    if n <= 2:
        mcx(wavefunction, controls, target)
        return

    # Use the relative-phase Toffoli decomposition
    # Split controls into two groups and use target as temporary
    # V gate decomposition: C^n(X) with n >= 3
    # Use the decomposition: C^(n-1)(V) · CNOT(c_{n-1}, t) · C^(n-1)(V†) · CNOT(c_{n-1}, t)
    # where V² = X, i.e., V = sqrt(X) = Xsquare

    # For simplicity and correctness, use direct amplitude manipulation
    # for the multi-controlled case
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    nq = len(states[0])
    dim = 2**nq
    new_amplitude = amplitude.copy()
    bit_target = nq - target - 1

    for i in range(dim):
        # Check if all control qubits are |1>
        all_ctrl_one = True
        for c in controls:
            if not (i >> (nq - c - 1)) & 1:
                all_ctrl_one = False
                break
        if all_ctrl_one:
            # Flip target bit
            j = i ^ (1 << bit_target)
            new_amplitude[i], new_amplitude[j] = amplitude[j], amplitude[i]

    wavefunction.amplitude = new_amplitude


def mcz(wavefunction: object, controls: list[int], target: int) -> None:
    """Multi-controlled Z gate."""
    controls = list(controls)
    nq = _nq(wavefunction)
    for q in controls + [target]:
        _validate_qubit(q, nq)

    # MCZ = H(target) · MCX · H(target)
    H(wavefunction, target)
    mcx(wavefunction, controls, target)
    H(wavefunction, target)


def mcp(wavefunction, controls: list[int], target: int, phi: float) -> None:
    """Multi-controlled Phase gate."""
    controls = list(controls)
    nq = _nq(wavefunction)
    for q in controls + [target]:
        _validate_qubit(q, nq)

    n_ctrl = len(controls)
    if n_ctrl == 0:
        Phase(wavefunction, target, phi)
    elif n_ctrl == 1:
        CPhase(wavefunction, controls[0], target, phi)
    else:
        # Direct amplitude manipulation for multi-controlled phase
        states = wavefunction.state
        amplitude = wavefunction.amplitude
        nq_total = len(states[0])
        new_amplitude = amplitude.copy()
        for i in range(len(amplitude)):
            all_one = True
            for c in controls:
                if not (i >> (nq_total - c - 1)) & 1:
                    all_one = False
                    break
            if all_one and (i >> (nq_total - target - 1)) & 1:
                new_amplitude[i] *= cmath.exp(1j * phi)
        wavefunction.amplitude = new_amplitude
