# -*- coding: utf-8 -*-
"""Correctness validation suite."""
from __future__ import annotations

import numpy as np
from qforge.circuit import Qubit
from qforge import gates as G, measurement as M, data as D


class CorrectnessSuite:
    name = "correctness"
    description = "Gate/measurement/entropy correctness tests"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        if backends is None:
            backends = ['cpu']
        tol = self.config.get("tol", 1e-12)
        results = {}
        for backend in backends:
            results[backend] = self._run_backend(backend, tol)
        return results

    def _run_backend(self, backend, tol):
        tests = []

        # Bell state
        c = Qubit(2, backend=backend)
        G.H(c, 0); G.CNOT(c, 0, 1)
        exp = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        diff = float(np.max(np.abs(c.amplitude - exp)))
        tests.append({"name": "Bell state", "diff": diff, "pass": diff < tol})

        # GHZ
        c = Qubit(3, backend=backend)
        G.H(c, 0); G.CNOT(c, 0, 1); G.CNOT(c, 1, 2)
        exp = np.zeros(8, dtype=complex); exp[0] = exp[7] = 1/np.sqrt(2)
        diff = float(np.max(np.abs(c.amplitude - exp)))
        tests.append({"name": "GHZ state", "diff": diff, "pass": diff < tol})

        # Single-qubit unitarity
        for name, fn, args in [
            ("X", G.X, []), ("Y", G.Y, []), ("Z", G.Z, []), ("H", G.H, []),
            ("RX", G.RX, [np.pi/3]), ("RY", G.RY, [np.pi/4]), ("RZ", G.RZ, [np.pi/6]),
        ]:
            c = Qubit(1, backend=backend); fn(c, 0, *args)
            norm = float(np.sum(np.abs(c.amplitude)**2))
            diff = abs(norm - 1.0)
            tests.append({"name": f"{name} unitarity", "diff": diff, "pass": diff < tol})

        # CNOT
        c = Qubit(2, backend=backend); G.X(c, 0); G.CNOT(c, 0, 1)
        diff = abs(float(np.abs(c.amplitude[3])) - 1.0)
        tests.append({"name": "CNOT", "diff": diff, "pass": diff < tol})

        # SWAP
        c = Qubit(2, backend=backend); G.X(c, 0); G.SWAP(c, 0, 1)
        diff = abs(float(np.abs(c.amplitude[1])) - 1.0)
        tests.append({"name": "SWAP", "diff": diff, "pass": diff < tol})

        # Measurement
        c = Qubit(2, backend=backend); G.H(c, 0); G.CNOT(c, 0, 1)
        probs = M.measure_one(c, 0)
        diff = float(np.max(np.abs(np.array(probs) - [0.5, 0.5])))
        tests.append({"name": "measure_one", "diff": diff, "pass": diff < tol})

        # Entropy
        c = Qubit(2, backend=backend); G.H(c, 0); G.CNOT(c, 0, 1)
        ee = D.EntanglementEntropy(c)
        ent = ee.von_neumann_entropy(keep_qubits=[0])
        diff = abs(ent - 1.0)
        tests.append({"name": "entropy", "diff": diff, "pass": diff < tol})

        all_pass = all(t["pass"] for t in tests)
        return {"tests": tests, "all_pass": all_pass, "n_pass": sum(t["pass"] for t in tests)}
