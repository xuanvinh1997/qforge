# -*- coding: utf-8 -*-
"""VQE & QAOA benchmark suite."""
from __future__ import annotations

import time
import numpy as np
from qforge.algo import Hamiltonian, VQE, QAOA


class VQESuite:
    name = "vqe"
    description = "VQE/QAOA convergence and timing"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        if backends is None:
            backends = ['cpu']
        steps = self.config.get("steps", 30)
        results = {"vqe": {}, "qaoa": {}}

        vqe_qubits = self.config.get("vqe_qubits", [2, 4, 6])
        qaoa_nodes = self.config.get("qaoa_nodes", [3, 4, 5, 6])

        for backend in backends:
            vqe_results = {}
            for nq in vqe_qubits:
                J, h = 1.0, 0.5
                coeffs, terms = [], []
                for i in range(nq):
                    coeffs.append(-h); terms.append([('X', i)])
                for i in range(nq - 1):
                    coeffs.append(-J); terms.append([('Z', i), ('Z', i + 1)])
                ham = Hamiltonian(coeffs, terms)
                n_layers = 2

                np.random.seed(42)
                p0 = np.random.uniform(-0.1, 0.1, nq * (n_layers + 1))
                vqe = VQE(n_qubits=nq, hamiltonian=ham, n_layers=n_layers, backend=backend)
                t0 = time.perf_counter()
                _, hist = vqe.optimize(p0, steps=steps)
                elapsed = time.perf_counter() - t0
                vqe_results[str(nq)] = {
                    "time": elapsed,
                    "final_energy": float(hist[-1]),
                    "history": [float(x) for x in hist],
                }
            results["vqe"][backend] = vqe_results

            qaoa_results = {}
            for nn in qaoa_nodes:
                edges = [(i, (i + 1) % nn) for i in range(nn)]
                np.random.seed(42)
                p0 = np.random.uniform(0, np.pi, 4)
                qaoa = QAOA(n_qubits=nn, edges=edges, p_layers=2, backend=backend)
                t0 = time.perf_counter()
                _, hist = qaoa.optimize(p0, steps=steps)
                elapsed = time.perf_counter() - t0
                qaoa_results[str(nn)] = {
                    "time": elapsed,
                    "final_cost": float(hist[-1]),
                }
            results["qaoa"][backend] = qaoa_results

        return results
