# -*- coding: utf-8 -*-
# author: vinhpx
"""Quantum measurement operations."""
from __future__ import annotations
import numpy as np
import math
from Qforge._utils import _validate_qubit, _nq, _is_mps


def measure_all(wavefunction, n_samples: int):
    inds = np.random.choice(len(wavefunction.state), n_samples, p=wavefunction.probabilities())
    return np.unique(np.array(wavefunction.state[inds]), return_counts=True)

def _mps_prob0(wavefunction, n: int) -> float:
    """Get P(qubit n = |0>) from an MPS wavefunction."""
    if hasattr(wavefunction, '_mps') and wavefunction._mps is not None:
        return float(wavefunction._mps.measure_prob0(n))
    # Python fallback: full contraction
    from Qforge.mps import _contract_py
    amp = _contract_py(wavefunction._tensors)
    nq = len(wavefunction._tensors)
    prob_0 = sum(abs(amp[i])**2 for i in range(len(amp))
                 if not (i >> (nq - n - 1)) & 1)
    return float(prob_0)


def measure_one(wavefunction, n: int) -> np.ndarray:
    """Return probabilities [P(|0>), P(|1>)] for qubit n."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        prob_0 = round(wavefunction._sv.measure_one_prob0(n), 10)
        return np.array([prob_0, 1 - prob_0])
    if _is_mps(wavefunction):
        prob_0 = round(_mps_prob0(wavefunction, n), 10)
        return np.array([prob_0, 1 - prob_0])
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    prob_0 = 0
    if n >= qubit_num or n < 0:
        raise ValueError("Index is out of range")
    for i in range(2**qubit_num):
        if states[i][n] == '0':
            prob_0 += abs(amplitude[i])**2
    prob_0 = round(prob_0, 10)
    return np.array([prob_0, 1 - prob_0])

def collapse_one(wavefunction, n: int) -> None:
    """Collapse qubit n into |0> or |1> via projective measurement."""
    _validate_qubit(n, _nq(wavefunction))
    if wavefunction._sv is not None:
        prob_0 = wavefunction._sv.measure_one_prob0(n)
        result_measure = np.random.choice(['0', '1'], 1, p=[prob_0, 1 - prob_0])[0]
        value = 0 if result_measure == '0' else 1
        wavefunction._sv.collapse_one(n, value)
        return
    if _is_mps(wavefunction):
        prob_0 = _mps_prob0(wavefunction, n)
        result_measure = np.random.choice(['0', '1'], 1, p=[prob_0, 1 - prob_0])[0]
        # Collapse: apply projector P0 or P1, then renormalize via amplitude setter
        from Qforge.mps import _contract_py, _svd_decompose
        amp = (_contract_py(wavefunction._tensors)
               if wavefunction._mps is None
               else wavefunction.amplitude)
        nq = _nq(wavefunction)
        mask = 1 << (nq - n - 1)
        if result_measure == '0':
            amp = np.where(np.arange(len(amp)) & mask == 0, amp, 0)
        else:
            amp = np.where(np.arange(len(amp)) & mask != 0, amp, 0)
        norm = np.linalg.norm(amp)
        if norm > 1e-14:
            amp /= norm
        wavefunction.amplitude = amp
        return
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    prob_0 = 0
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise ValueError("Index is out of range")
    for i in range(2**qubit_num):
        if states[i][n] == '0':
            prob_0 += abs(amplitude[i])**2
    result_measure = np.random.choice(['0', '1'], 1, p = [prob_0, 1 - prob_0])[0]
    if result_measure == '0':
        for i in range(2**qubit_num):
            if states[i][n] == '0':
                new_amplitude[i] = amplitude[i]/math.sqrt(prob_0)
    elif result_measure == '1':
        for i in range(2**qubit_num):
            if states[i][n] == '1':
                new_amplitude[i] = amplitude[i]/math.sqrt(1 - prob_0)
    wavefunction.amplitude = new_amplitude

_PAULI_MAP = {'X': 0, 'Y': 1, 'Z': 2}

def pauli_expectation(wavefunction, qubit_idx: int, pauli_type: str) -> float:
    """Expectation value of Pauli operator ('X', 'Y', or 'Z') on qubit."""
    _validate_qubit(qubit_idx, _nq(wavefunction))
    if wavefunction._sv is not None:
        return wavefunction._sv.pauli_expectation(qubit_idx, _PAULI_MAP[pauli_type])
    if _is_mps(wavefunction):
        _PAULI_OPS = {
            'X': np.array([0, 1, 1, 0], dtype=complex),
            'Y': np.array([0, -1j, 1j, 0], dtype=complex),
            'Z': np.array([1, 0, 0, -1], dtype=complex),
        }
        op = _PAULI_OPS[pauli_type]
        if wavefunction._mps is not None:
            return float(wavefunction._mps.single_site_expectation(qubit_idx, op).real)
        from Qforge.dmrg import _single_site_expect_py
        return float(_single_site_expect_py(
            wavefunction._tensors, op.reshape(2, 2), qubit_idx).real)
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])

    if qubit_idx >= qubit_num or qubit_idx < 0:
        raise ValueError("Index is out of range")

    expectation = 0.0

    if pauli_type == 'X':
        # <X> = sum of amplitude[i]*conj(amplitude[j]) where state[i] and state[j] differ at qubit_idx
        cut = 2**(qubit_num - qubit_idx - 1)
        for i in range(2**qubit_num):
            if states[i][qubit_idx] == '0':
                # <i|X|j> where j is i with flipped bit at qubit_idx
                j = i + cut
                expectation += 2 * np.real(np.conj(amplitude[i]) * amplitude[j])

    elif pauli_type == 'Y':
        # <Y> similar to X but with imaginary component
        cut = 2**(qubit_num - qubit_idx - 1)
        for i in range(2**qubit_num):
            if states[i][qubit_idx] == '0':
                j = i + cut
                expectation += 2 * np.real(-1j * np.conj(amplitude[i]) * amplitude[j])

    elif pauli_type == 'Z':
        # <Z> = P(|0>) - P(|1>)
        for i in range(2**qubit_num):
            prob = abs(amplitude[i])**2
            if states[i][qubit_idx] == '0':
                expectation += prob
            else:
                expectation -= prob

    return expectation
