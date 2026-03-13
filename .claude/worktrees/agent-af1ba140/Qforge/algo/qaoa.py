# -*- coding: utf-8 -*-
# author: vinhpx
"""QAOA — Quantum Approximate Optimization Algorithm."""
from __future__ import annotations
import numpy as np
from Qforge.circuit import Qubit
from Qforge.gates import H, RX, RZ, CNOT
from Qforge.algo.hamiltonian import Hamiltonian
from Qforge.algo.vqa import VQA


def _maxcut_hamiltonian(edges: list[tuple[int, int]]) -> Hamiltonian:
    """Build H_C = ½ Σ_{(i,j)∈E} (I − ZᵢZⱼ) for the Max-Cut problem."""
    coeffs, terms = [], []
    for (i, j) in edges:
        coeffs.append(0.5)
        terms.append([])                        # +0.5 · I
        coeffs.append(-0.5)
        terms.append([('Z', i), ('Z', j)])      # −0.5 · ZᵢZⱼ
    return Hamiltonian(coeffs, terms)


class QAOA(VQA):
    """Quantum Approximate Optimization Algorithm for Max-Cut.

    Builds and optimizes a p-layer QAOA circuit over an undirected graph.
    The cost Hamiltonian is H_C = ½ Σ_{(i,j)∈E} (I − ZᵢZⱼ); maximizing
    ⟨H_C⟩ is equivalent to maximizing the number of cut edges.

    Args:
        n_qubits: Number of qubits (= number of graph nodes).
        edges:    List of ``(i, j)`` edge tuples (0-indexed nodes).
        p_layers: Number of QAOA layers (higher → more expressive).
        backend:  qforge backend.

    Parameter layout::

        params[:p_layers]  = γ  (problem Hamiltonian angles)
        params[p_layers:]  = β  (mixer Hamiltonian angles)

    Example::

        from Qforge.algo import QAOA
        import numpy as np

        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]   # 4-cycle graph
        qaoa = QAOA(n_qubits=4, edges=edges, p_layers=2)
        params0 = np.random.uniform(0, np.pi, qaoa.n_params)
        params, history = qaoa.optimize(params0, steps=120)
        sol = qaoa.get_solution(params)
        print("Best bitstring:", sol['bitstring'], "| Cut:", sol['cut_value'])
    """

    def __init__(self, n_qubits: int, edges: list[tuple[int, int]],
                 p_layers: int = 1, backend: str = 'auto'):
        self.edges = list(edges)
        self.p_layers = p_layers
        self._cost_hamiltonian = _maxcut_hamiltonian(self.edges)

        edges_ref = self.edges
        p_ref = p_layers

        def _qaoa_circuit(wf, params):
            gammas = params[:p_ref]
            betas  = params[p_ref:]
            # Initial state |+>^n
            for q in range(n_qubits):
                H(wf, q)
            # p layers: problem unitary + mixer unitary
            for l in range(p_ref):
                # Problem unitary e^{−iγ H_C}:
                # CNOT(i,j) → RZ(j, −γ) → CNOT(i,j) implements e^{+iγ/2 ZᵢZⱼ}
                for (i, j) in edges_ref:
                    CNOT(wf, i, j)
                    RZ(wf, j, -gammas[l])
                    CNOT(wf, i, j)
                # Mixer unitary e^{−iβ H_B} = Π_q RX(q, 2β)
                for q in range(n_qubits):
                    RX(wf, q, 2.0 * betas[l])

        super().__init__(
            n_qubits=n_qubits,
            circuit_fn=_qaoa_circuit,
            cost_fn=lambda wf: -self._cost_hamiltonian.expectation(wf),
            backend=backend,
        )

    @property
    def n_params(self) -> int:
        """Total number of variational parameters ``2 * p_layers``."""
        return 2 * self.p_layers

    def get_solution(self, params: np.ndarray, n_samples: int = 1000) -> dict:
        """Sample the optimized circuit and return the best bitstring found.

        Args:
            params:    Optimized variational parameters.
            n_samples: Number of measurement shots.

        Returns:
            Dict with keys:

            * ``'bitstring'``: most-probable measurement outcome (str).
            * ``'cut_value'``: number of cut edges in this partition (int).
            * ``'samples'``:   all unique sampled bitstrings (array).
            * ``'counts'``:    corresponding shot counts (array).
        """
        from Qforge.measurement import measure_all
        wf = Qubit(self.n_qubits, backend=self.backend)
        self.circuit_fn(wf, np.asarray(params, dtype=float))
        bitstrings, counts = measure_all(wf, n_samples)
        best_idx = int(np.argmax(counts))
        best = bitstrings[best_idx]
        cut = sum(1 for (i, j) in self.edges if best[i] != best[j])
        return {
            'bitstring': best,
            'cut_value': cut,
            'samples': bitstrings,
            'counts': counts,
        }
