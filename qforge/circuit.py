# -*- coding: utf-8 -*-
# author: vinhpx
"""Quantum circuit initialization."""
from __future__ import annotations
import os
from qforge.wavefunction import Wavefunction
from qforge import (
    _HAS_CPP, _HAS_CUDA, _HAS_METAL, _HAS_DISTRIBUTED,
    _resolve_backend, get_backend,
)
import itertools
import numpy as np

# GPU backends (Metal/CUDA) have fixed per-kernel dispatch overhead
# (~100-200 µs on Metal) that only amortizes for circuits with enough
# amplitudes. Below this threshold, the C++ CPU backend is orders of
# magnitude faster. Override via ``QFORGE_GPU_MIN_QUBITS``.
GPU_MIN_QUBITS = int(os.environ.get('QFORGE_GPU_MIN_QUBITS', '14'))

if _HAS_CPP:
    from qforge._qforge_core import StateVector as _StateVector
if _HAS_CUDA:
    from qforge._qforge_cuda import CudaStateVector as _CudaStateVector
if _HAS_METAL:
    from qforge._qforge_metal import MetalStateVector as _MetalStateVector
if _HAS_DISTRIBUTED:
    from qforge._qforge_distributed import DistributedStateVector as _DistributedStateVector

# https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long
def Qubit(qubit_num: int, backend: str = 'auto') -> Wavefunction:
    """Create a quantum circuit.

    Args:
        qubit_num: Number of qubits.
        backend: ``'auto'``, ``'cpu'``, ``'cuda'``, ``'metal'``, or ``'python'``.
            ``'auto'`` uses the global default set by :func:`qforge.set_backend`.
    """
    states = ["".join(seq) for seq in itertools.product("01", repeat=qubit_num)]
    amplitude_vector = np.zeros(2**qubit_num, dtype=complex)
    amplitude_vector[0] = 1.0

    if backend == 'auto':
        backend = _resolve_backend(get_backend())
        # GPU backends are a pessimization for tiny circuits — their
        # per-kernel dispatch cost dwarfs the actual compute. Fall back
        # to the C++ CPU backend below the threshold.
        if backend in ('metal', 'cuda') and qubit_num < GPU_MIN_QUBITS and _HAS_CPP:
            backend = 'cpu'

    if backend == 'cuda' and _HAS_CUDA:
        sv = _CudaStateVector(qubit_num)
        return Wavefunction(np.array(states), amplitude_vector, _sv=sv,
                            backend='cuda')
    elif backend == 'metal' and _HAS_METAL:
        sv = _MetalStateVector(qubit_num)
        return Wavefunction(np.array(states), amplitude_vector, _sv=sv,
                            backend='metal')
    elif backend == 'distributed' and _HAS_DISTRIBUTED:
        sv = _DistributedStateVector(qubit_num)
        return Wavefunction(np.array(states), amplitude_vector, _sv=sv,
                            backend='distributed')
    elif backend == 'numpy':
        return Wavefunction(np.array(states), amplitude_vector,
                            _sv=None, backend='numpy')
    elif backend in ('cpu', 'auto') and _HAS_CPP:
        sv = _StateVector(qubit_num)
        return Wavefunction(np.array(states), amplitude_vector, _sv=sv,
                            backend='cpu')
    return Wavefunction(np.array(states), amplitude_vector, backend='python')

def Qudit(n_qudits: int, dimension: int = 3, backend: str = 'auto') -> Wavefunction:
    """Create a qudit quantum circuit with arbitrary local dimension.

    Args:
        n_qudits: Number of qudits.
        dimension: Local Hilbert space dimension (d=2 for qubits, d=3 for qutrits).
        backend: ``'auto'``, ``'cpu'``, or ``'python'``.
    """
    states = [
        "".join(str(v) for v in seq)
        for seq in itertools.product(range(dimension), repeat=n_qudits)
    ]
    amplitude_vector = np.zeros(dimension**n_qudits, dtype=complex)
    amplitude_vector[0] = 1.0

    if backend == 'auto':
        backend = _resolve_backend(get_backend())

    if backend in ('cpu', 'auto') and _HAS_CPP:
        sv = _StateVector(n_qudits, dimension)
        return Wavefunction(np.array(states), amplitude_vector, _sv=sv,
                            backend='cpu')
    return Wavefunction(np.array(states), amplitude_vector, backend='python')


def Walk_Qubit(qubit_num=1, dim=1):
    """create a initial quantum state for hadamard coin"""
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    else:
        qubit_num += 1
        if dim == 1:
            #initial state: (|0> - i|1>)x|n=0>/(sqrt(2))
            states = ['0' + str(i) for i in range(2*qubit_num-1)]
            states += ['1' + str(i) for i in range(2*qubit_num-1)]
        
            amplitude_vector = np.zeros(4*qubit_num-2, dtype = complex)
            amplitude_vector[qubit_num-1] = 2**-0.5
            amplitude_vector[3*qubit_num-2] = (-2)**-0.5
            return Wavefunction(np.array(states), amplitude_vector, backend='python')
        else:
            #initial state: ((|0> + i|1>)/sqrt(2))x((|0> + i|1>)/sqrt(2))x|n=0>x|n=0>
            states = ['0' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['1' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['2' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['3' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            
            amplitude_vector = np.zeros(4*(2*qubit_num-1)**2, dtype = complex)
            index = int(((2*qubit_num-1)**2-1)/2)
            amplitude_vector[index] = 1/2
            amplitude_vector[index+(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+2*(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+3*(2*qubit_num-1)**2] = -1/2
            return Wavefunction(np.array(states), amplitude_vector, backend='python')