# -*- coding: utf-8 -*-
# author: vinhpx
"""
Distributed state vector simulation via MPI.

Shards the 2^n amplitude vector across MPI ranks for simulating
30-40+ qubit circuits. Each rank holds 2^(n-k) amplitudes where
k = log2(n_ranks). Gates on local qubits require no communication;
gates on global qubits use MPI pairwise exchange.

Usage::

    # Run with: mpirun -np 4 python my_circuit.py
    from qforge.distributed import DistributedQubit
    from qforge import gates

    wf = DistributedQubit(30)  # 30 qubits across 4 ranks
    gates.H(wf, 0)
    for i in range(29):
        gates.CNOT(wf, i, i + 1)
    # Measurement works across ranks
    from qforge import measurement
    print(measurement.measure_one(wf, 0))
"""
from __future__ import annotations
import itertools
import numpy as np
from qforge.wavefunction import Wavefunction

try:
    from qforge._qforge_distributed import DistributedStateVector as _DistSV
    _HAS_DISTRIBUTED = True
except ImportError:
    _HAS_DISTRIBUTED = False


def DistributedQubit(qubit_num: int) -> Wavefunction:
    """Create a distributed quantum circuit.

    Requires MPI: install with ``QFORGE_MPI=1 pip install -e .``
    Run with: ``mpirun -np <N> python script.py`` where N is a power of 2.

    Args:
        qubit_num: Number of qubits. Practical range: 25-40.

    Returns:
        Wavefunction object backed by MPI-distributed state vector.
        Compatible with all gate/measurement functions.
    """
    if not _HAS_DISTRIBUTED:
        raise RuntimeError(
            "Distributed backend not available. "
            "Build with: QFORGE_MPI=1 pip install -e ."
        )

    sv = _DistSV(qubit_num)

    # For large qubit counts, don't enumerate all basis states
    # (that would require 2^n strings). Use a placeholder.
    if qubit_num <= 20:
        states = np.array(
            ["".join(seq) for seq in itertools.product("01", repeat=qubit_num)]
        )
    else:
        states = np.array(['0' * qubit_num])

    amplitude_vector = np.zeros(2**min(qubit_num, 20), dtype=complex)
    amplitude_vector[0] = 1.0

    return Wavefunction(states, amplitude_vector, _sv=sv)
