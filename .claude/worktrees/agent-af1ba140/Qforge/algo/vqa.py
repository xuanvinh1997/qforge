# -*- coding: utf-8 -*-
# author: vinhpx
"""VQA — general variational quantum algorithm framework."""
from __future__ import annotations
import numpy as np
from Qforge.circuit import Qubit
from Qforge.algo.gradient import parameter_shift, parallel_parameter_shift
from Qforge.algo.optimizers import Adam


class VQA:
    """Variational Quantum Algorithm — general framework.

    Args:
        n_qubits:   Number of qubits.
        circuit_fn: Ansatz — ``callable(wf, params) → None``.
        cost_fn:    Cost function — ``callable(wf) → float``.
        backend:    qforge backend (``'auto'``, ``'cpu'``, ``'cuda'``, ``'metal'``).

    Example::

        from Qforge.gates import RY, CNOT
        from Qforge.algo import Hamiltonian, VQA
        import numpy as np

        def my_ansatz(wf, params):
            RY(wf, 0, params[0])
            CNOT(wf, 0, 1)
            RY(wf, 1, params[1])

        H = Hamiltonian([-1.0], [[('Z', 0)]])
        vqa = VQA(n_qubits=2, circuit_fn=my_ansatz, cost_fn=H.expectation)
        params, history = vqa.optimize(np.zeros(2), steps=80)
    """

    def __init__(self, n_qubits: int, circuit_fn, cost_fn, backend: str = 'auto'):
        self.n_qubits = n_qubits
        self.circuit_fn = circuit_fn
        self.cost_fn = cost_fn
        self.backend = backend

    def _evaluate(self, params: np.ndarray) -> float:
        """Create a fresh |0…0> state, run the circuit, return the cost."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        self.circuit_fn(wf, params)
        return self.cost_fn(wf)

    def gradient(
        self,
        params: np.ndarray,
        shift: float = np.pi / 2,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> np.ndarray:
        """Parameter-shift gradient of the cost w.r.t. ``params``.

        Args:
            params:      Parameter array.
            shift:       Shift amount (default π/2).
            parallel:    If ``True``, evaluate shifted circuits concurrently
                         via :func:`parallel_parameter_shift`.  Each evaluation
                         creates its own independent ``Qubit`` state, so the
                         circuit function must not rely on shared mutable state.
            max_workers: Thread-pool size when ``parallel=True``.
        """
        params = np.asarray(params, dtype=float)
        if parallel:
            return parallel_parameter_shift(self._evaluate, params, shift, max_workers)
        return parameter_shift(self._evaluate, params, shift)

    def optimize(
        self,
        params: np.ndarray,
        optimizer=None,
        steps: int = 100,
        callback=None,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        """Optimize the variational parameters.

        Args:
            params:      Initial parameters (copied internally).
            optimizer:   Instance with ``step(params, grad) → new_params``.
                         Defaults to ``Adam(lr=0.05)``.
            steps:       Number of gradient-descent steps.
            callback:    Optional ``callable(step, params, cost)`` called each step.
            parallel:    If ``True``, use concurrent parameter-shift evaluations.
            max_workers: Thread-pool size when ``parallel=True``.

        Returns:
            ``(optimal_params, cost_history)`` — ``cost_history[0]`` is the
            cost before any update.
        """
        if optimizer is None:
            optimizer = Adam(lr=0.05)
        params = np.asarray(params, dtype=float).copy()
        history: list[float] = []
        for step in range(steps):
            cost = self._evaluate(params)
            history.append(cost)
            grad = self.gradient(params, parallel=parallel, max_workers=max_workers)
            params = optimizer.step(params, grad)
            if callback is not None:
                callback(step, params, cost)
        return params, history
