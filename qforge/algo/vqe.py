# -*- coding: utf-8 -*-
# author: vinhpx
"""VQE — Variational Quantum Eigensolver."""
from __future__ import annotations
from typing import Callable
import numpy as np
from qforge.algo.hamiltonian import Hamiltonian
from qforge.algo.ansatz import hardware_efficient_ansatz
from qforge.algo.vqa import VQA


class VQE(VQA):
    """Variational Quantum Eigensolver.

    Finds the ground-state energy of a Hamiltonian using a parameterized ansatz.

    Args:
        n_qubits:    Number of qubits.
        hamiltonian: :class:`~qforge.algo.Hamiltonian` object.
        circuit_fn:  Ansatz — ``callable(wf, params) → None``.
                     Defaults to :func:`~qforge.algo.hardware_efficient_ansatz`.
        n_layers:    Layers for the default ansatz (ignored if ``circuit_fn`` given).
        backend:     qforge backend.

    Example::

        from qforge.algo import Hamiltonian, VQE
        import numpy as np

        # Toy H₂ Hamiltonian (2 qubits)
        H = Hamiltonian(
            coeffs=[-1.0523, 0.3979, -0.3979, -0.0112, 0.1809],
            terms=[
                [],
                [('Z', 0)],
                [('Z', 1)],
                [('Z', 0), ('Z', 1)],
                [('X', 0), ('X', 1)],
            ],
        )
        n_p = VQE.n_params_hardware_efficient(n_qubits=2, n_layers=2)
        vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=2)
        params, history = vqe.optimize(np.zeros(n_p), steps=150)
        print("Ground-state energy:", history[-1])
    """

    def __init__(self, n_qubits: int, hamiltonian: Hamiltonian,
                 circuit_fn: Callable | None = None, n_layers: int = 1, backend: str = 'auto'):
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        if circuit_fn is None:
            def _default_ansatz(wf, params):
                hardware_efficient_ansatz(wf, params, n_layers)
            circuit_fn = _default_ansatz
        super().__init__(
            n_qubits=n_qubits,
            circuit_fn=circuit_fn,
            cost_fn=hamiltonian.expectation,
            backend=backend,
        )

    @staticmethod
    def n_params_hardware_efficient(n_qubits: int, n_layers: int = 1) -> int:
        """Number of parameters for :func:`~qforge.algo.hardware_efficient_ansatz`."""
        return n_qubits * (n_layers + 1)

    @staticmethod
    def n_params_strongly_entangling(n_qubits: int, n_layers: int = 1) -> int:
        """Number of parameters for :func:`~qforge.algo.strongly_entangling_ansatz`."""
        return n_layers * n_qubits * 3
