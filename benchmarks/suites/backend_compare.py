# -*- coding: utf-8 -*-
"""Backend comparison: CPU vs Metal (vs CUDA if available)."""
from __future__ import annotations

import time
import numpy as np
from qforge import _HAS_METAL, _HAS_CUDA
from qforge.circuit import Qubit
from qforge import gates as G


class BackendCompareSuite:
    name = "backend"
    description = "CPU vs Metal vs CUDA gate/circuit performance"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        if backends is None:
            backends = ['cpu']
            if _HAS_METAL:
                backends.append('metal')
            if _HAS_CUDA:
                backends.append('cuda')

        qubit_range = self.config.get("qubit_range", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
        depth = self.config.get("depth", 10)
        repeats = self.config.get("repeats", 3)

        results = {"backends": backends, "circuit": {}, "single_gate": {}}

        for n in qubit_range:
            circuit_entry = {}
            gate_entry = {}

            for backend in backends:
                # Circuit benchmark
                best = float('inf')
                for _ in range(repeats):
                    wf = Qubit(n, backend=backend)
                    t0 = time.perf_counter()
                    for _ in range(depth):
                        for q in range(n): G.H(wf, q)
                        for q in range(n - 1): G.CNOT(wf, q, q + 1)
                    if backend in ('metal', 'cuda'):
                        _ = wf.amplitude  # force sync
                    elapsed = time.perf_counter() - t0
                    best = min(best, elapsed)
                circuit_entry[backend] = best

                # Single gate
                if n >= 4:
                    wf = Qubit(n, backend=backend)
                    times = []
                    for _ in range(100):
                        t0 = time.perf_counter()
                        G.H(wf, 0)
                        times.append(time.perf_counter() - t0)
                    gate_entry[backend] = float(np.median(times) * 1e6)

            results["circuit"][str(n)] = circuit_entry
            if gate_entry:
                results["single_gate"][str(n)] = gate_entry

        # Compute crossover point (where Metal beats CPU)
        if 'metal' in backends:
            crossover = None
            for n in qubit_range:
                e = results["circuit"].get(str(n), {})
                if e.get("metal", float('inf')) < e.get("cpu", 0):
                    crossover = n
                    break
            results["metal_crossover_qubits"] = crossover

        return results
