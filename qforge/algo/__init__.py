# -*- coding: utf-8 -*-
# author: vinhpx
"""qforge.algo — QML algorithm support.

Modules
-------
hamiltonian  Hamiltonian and Pauli-string expectations
gradient     parameter_shift gradient rule
optimizers   GradientDescent, Adam
ansatz       hardware_efficient_ansatz, strongly_entangling_ansatz
vqa          VQA (general variational framework)
vqe          VQE (Variational Quantum Eigensolver)
qaoa         QAOA (Quantum Approximate Optimization Algorithm)

Quick start::

    from qforge.algo import Hamiltonian, VQE, QAOA, Adam
    import numpy as np

    # VQE
    H = Hamiltonian([-1.0, 0.5], [[('Z', 0)], [('X', 0), ('X', 1)]])
    vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=1)
    params, history = vqe.optimize(np.zeros(4), steps=100)

    # QAOA (Max-Cut)
    qaoa = QAOA(n_qubits=4, edges=[(0,1),(1,2),(2,3),(3,0)], p_layers=2)
    p0 = np.random.uniform(0, np.pi, qaoa.n_params)
    params, history = qaoa.optimize(p0, steps=100)
    sol = qaoa.get_solution(params)
"""

from qforge.algo.hamiltonian import Hamiltonian
from qforge.algo.gradient import parameter_shift, parallel_parameter_shift
from qforge.algo.optimizers import GradientDescent, Adam, SPSA, LBFGS
from qforge.algo.ansatz import hardware_efficient_ansatz, strongly_entangling_ansatz
from qforge.algo.vqa import VQA
from qforge.algo.vqe import VQE
from qforge.algo.qaoa import QAOA
from qforge.mps import MatrixProductState
from qforge.dmrg import DMRG

__all__ = [
    "Hamiltonian",
    "parameter_shift",
    "parallel_parameter_shift",
    "GradientDescent",
    "Adam",
    "SPSA",
    "LBFGS",
    "hardware_efficient_ansatz",
    "strongly_entangling_ansatz",
    "VQA",
    "VQE",
    "QAOA",
    "MatrixProductState",
    "DMRG",
]
