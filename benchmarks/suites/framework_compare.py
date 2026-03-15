# -*- coding: utf-8 -*-
"""Cross-framework comparison: Qforge vs Qiskit vs PennyLane."""
from __future__ import annotations

import time
import numpy as np


class FrameworkCompareSuite:
    name = "framework"
    description = "Qforge vs Qiskit vs PennyLane (gates, VQE, gradient)"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        qubit_range = self.config.get("qubit_range", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        depth = self.config.get("depth", 10)
        repeats = self.config.get("repeats", 3)
        results = {"circuit": {}, "gradient": {}}

        # ---- Circuit execution ----
        for n in qubit_range:
            entry = {}

            # Qforge CPU
            from qforge.circuit import Qubit
            from qforge import gates as G
            best = float('inf')
            for _ in range(repeats):
                wf = Qubit(n, backend='cpu')
                t0 = time.perf_counter()
                for _ in range(depth):
                    for q in range(n): G.H(wf, q)
                    for q in range(n - 1): G.CNOT(wf, q, q + 1)
                best = min(best, time.perf_counter() - t0)
            entry["qforge"] = best

            # Qiskit
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                best = float('inf')
                for _ in range(repeats):
                    qc = QuantumCircuit(n)
                    for _ in range(depth):
                        for q in range(n): qc.h(q)
                        for q in range(n - 1): qc.cx(q, q + 1)
                    t0 = time.perf_counter()
                    Statevector(qc)
                    best = min(best, time.perf_counter() - t0)
                entry["qiskit"] = best
            except ImportError:
                entry["qiskit"] = None

            # PennyLane
            try:
                import pennylane as qml
                dev = qml.device('default.qubit', wires=n)
                @qml.qnode(dev)
                def circuit():
                    for _ in range(depth):
                        for q in range(n): qml.Hadamard(wires=q)
                        for q in range(n - 1): qml.CNOT(wires=[q, q + 1])
                    return qml.state()
                best = float('inf')
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    circuit()
                    best = min(best, time.perf_counter() - t0)
                entry["pennylane"] = best
            except ImportError:
                entry["pennylane"] = None

            results["circuit"][str(n)] = entry

        # ---- Gradient ----
        grad_qubits = self.config.get("grad_qubits", [2, 4, 6, 8])
        n_layers = 2
        for nq in grad_qubits:
            n_params = nq * (n_layers + 1)
            np.random.seed(42)
            p0 = np.random.uniform(-np.pi, np.pi, n_params)
            entry = {"n_params": n_params}

            # Qforge
            from qforge.circuit import Qubit
            from qforge.gates import RY, CNOT
            from qforge.algo import Hamiltonian
            from qforge.algo.gradient import parameter_shift
            coeffs = [-1.0] * nq
            terms = [[('Z', i)] for i in range(nq)]
            ham = Hamiltonian(coeffs, terms)
            def qf_cost(params):
                wf = Qubit(nq, backend='cpu')
                idx = 0
                for layer in range(n_layers + 1):
                    for q in range(nq): RY(wf, q, params[idx]); idx += 1
                    if layer < n_layers:
                        for q in range(nq - 1): CNOT(wf, q, q + 1)
                return ham.expectation(wf)
            t0 = time.perf_counter()
            parameter_shift(qf_cost, p0.copy())
            entry["qforge"] = time.perf_counter() - t0

            # Qiskit
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector, SparsePauliOp
                def qi_cost(params):
                    qc = QuantumCircuit(nq)
                    idx = 0
                    for layer in range(n_layers + 1):
                        for q in range(nq): qc.ry(params[idx], q); idx += 1
                        if layer < n_layers:
                            for q in range(nq - 1): qc.cx(q, q + 1)
                    sv = Statevector(qc)
                    pl = [(''.join(['Z' if j == nq-1-i else 'I' for j in range(nq)]), -1.0) for i in range(nq)]
                    return float(np.real(sv.expectation_value(SparsePauliOp.from_list(pl))))
                t0 = time.perf_counter()
                shift = np.pi / 2
                grad = np.zeros(n_params)
                for i in range(n_params):
                    pp = p0.copy(); pp[i] += shift
                    pm = p0.copy(); pm[i] -= shift
                    grad[i] = (qi_cost(pp) - qi_cost(pm)) / 2
                entry["qiskit"] = time.perf_counter() - t0
            except ImportError:
                entry["qiskit"] = None

            # PennyLane
            try:
                import pennylane as qml
                dev = qml.device('default.qubit', wires=nq)
                @qml.qnode(dev, diff_method='parameter-shift')
                def pl_cost(params):
                    idx = 0
                    for layer in range(n_layers + 1):
                        for q in range(nq): qml.RY(params[idx], wires=q); idx += 1
                        if layer < n_layers:
                            for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
                    return qml.expval(qml.sum(*[-1.0 * qml.PauliZ(i) for i in range(nq)]))
                grad_fn = qml.grad(pl_cost)
                t0 = time.perf_counter()
                grad_fn(qml.numpy.array(p0.copy(), requires_grad=True))
                entry["pennylane"] = time.perf_counter() - t0
            except ImportError:
                entry["pennylane"] = None

            results["gradient"][str(nq)] = entry

        return results
