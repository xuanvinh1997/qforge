# -*- coding: utf-8 -*-
"""Gate-level performance suite."""
from __future__ import annotations

import time
import numpy as np
from qforge.circuit import Qubit
from qforge import gates as G


class GatePerfSuite:
    name = "gate_perf"
    description = "Gate-level circuit and single-gate timing"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        if backends is None:
            backends = ['cpu']
        qubit_range = self.config.get("qubit_range", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        depth = self.config.get("depth", 10)
        repeats = self.config.get("repeats", 3)
        results = {}
        for backend in backends:
            results[backend] = self._run_backend(backend, qubit_range, depth, repeats)
        return results

    def _run_backend(self, backend, qubit_range, depth, repeats):
        circuit_times = {}
        single_gate_times = {}

        for n in qubit_range:
            # Circuit benchmark: H+CNOT chain
            best = float('inf')
            for _ in range(repeats):
                wf = Qubit(n, backend=backend)
                t0 = time.perf_counter()
                for _ in range(depth):
                    for q in range(n): G.H(wf, q)
                    for q in range(n - 1): G.CNOT(wf, q, q + 1)
                if backend == 'metal':
                    _ = wf.amplitude  # force sync
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            total_gates = depth * (n + n - 1)
            circuit_times[str(n)] = {
                "time": best, "gates": total_gates,
                "throughput": total_gates / best,
            }

            # Single H gate
            if n >= 4:
                wf = Qubit(n, backend=backend)
                times = []
                for _ in range(200):
                    t0 = time.perf_counter()
                    G.H(wf, 0)
                    times.append(time.perf_counter() - t0)
                single_gate_times[str(n)] = {
                    "median_us": float(np.median(times) * 1e6),
                    "min_us": float(min(times) * 1e6),
                }

        return {
            "circuit": circuit_times,
            "single_gate": single_gate_times,
            "depth": depth,
        }
