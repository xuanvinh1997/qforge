# -*- coding: utf-8 -*-
# author: vinhpx
"""Hartree-Fock state preparation.

Prepares the Hartree-Fock reference state by flipping the first n_electrons
qubits from |0> to |1>, corresponding to occupying the lowest-energy
spin-orbitals in the Jordan-Wigner encoding.
"""
from __future__ import annotations

from qforge import gates


def hartree_fock_state(wf: object, n_electrons: int) -> None:
    """Prepare the Hartree-Fock reference state on a wavefunction.

    In the Jordan-Wigner mapping, the HF state |1...10...0> is obtained
    by applying X gates to the first ``n_electrons`` qubits (which start
    in |0>).

    Args:
        wf:           Wavefunction initialized to |00...0>.
        n_electrons:  Number of electrons to occupy.

    Example::

        from qforge.circuit import Qubit
        from qforge.chem import hartree_fock_state

        wf = Qubit(4)
        hartree_fock_state(wf, 2)
        # wf is now |1100>
    """
    if n_electrons < 0:
        raise ValueError(f"n_electrons must be non-negative, got {n_electrons}")

    for i in range(n_electrons):
        gates.X(wf, i)
