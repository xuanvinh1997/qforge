# -*- coding: utf-8 -*-
"""Gradient computation benchmark suite."""
from __future__ import annotations

import time
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import RY, CNOT
from qforge.algo import Hamiltonian
from qforge.algo.gradient import parameter_shift


class GradientSuite:
    name = "gradient"
    description = "Parameter-shift gradient scaling"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        if backends is None:
            backends = ['cpu']
        qubit_range = self.config.get("grad_qubits", [2, 4, 6, 8, 10])
        n_layers = 2
        results = {}

        for backend in backends:
            backend_results = {}
            for nq in qubit_range:
                n_params = nq * (n_layers + 1)
                np.random.seed(42)
                p0 = np.random.uniform(-np.pi, np.pi, n_params)

                ham = Hamiltonian([-1.0] * nq, [[('Z', i)] for i in range(nq)])

                def cost(params, _nq=nq, _nl=n_layers, _ham=ham, _b=backend):
                    wf = Qubit(_nq, backend=_b)
                    idx = 0
                    for layer in range(_nl + 1):
                        for q in range(_nq): RY(wf, q, params[idx]); idx += 1
                        if layer < _nl:
                            for q in range(_nq - 1): CNOT(wf, q, q + 1)
                    return _ham.expectation(wf)

                t0 = time.perf_counter()
                parameter_shift(cost, p0)
                elapsed = time.perf_counter() - t0

                backend_results[str(nq)] = {
                    "time": elapsed,
                    "n_params": n_params,
                    "evals": 2 * n_params,
                }
            results[backend] = backend_results
        return results
