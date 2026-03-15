# -*- coding: utf-8 -*-
"""Category 4: QAOA Algorithm (Max-Cut) benchmark."""
from __future__ import annotations

import os
import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)

SHIFT = np.pi / 2
QAOA_RING_4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
QAOA_GRAPH_8 = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 7), (2, 3),
                (2, 6), (3, 4), (4, 5), (4, 7), (5, 6), (6, 7)]

_BENCH_CUDA = os.environ.get("QFORGE_BENCH_CUDA", "") == "1"


class QAOABenchmarkSuite(BaseBenchmarkSuite):
    name = "qaoa"
    description = "QAOA Algorithm (Max-Cut)"

    def run(self):
        section(f"CATEGORY 4: {self.description}",
                f"p=1, {self.config.steps} steps, lr={self.config.lr}, parameter-shift gradient")

        problems = [("4-node ring", QAOA_RING_4, 4)]
        if self.config.max_qubits >= 8:
            problems.append(("8-node 3-reg", QAOA_GRAPH_8, 8))

        for prob_name, edges, nq in problems:
            print(f"\n  --- {prob_name}, {nq} qubits ---")
            headers = ["Framework", "ms/step", "total(s)", "cut value"]
            rows = []

            from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
            for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA and _BENCH_CUDA), ("metal", _HAS_METAL)]:
                if not available:
                    continue
                try:
                    r = self._qaoa_qforge(edges, nq, 1, backend)
                    ms = r.median / self.config.steps * 1000
                    rows.append([f"Qforge ({backend})", f"{ms:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_{backend}", {"time": r.median, "cut": float(r.result), "framework": f"qforge_{backend}", "problem": prob_name})
                except Exception as e:
                    rows.append([f"Qforge ({backend})", "ERR", "ERR", str(e)[:30]])

            try:
                r = self._qaoa_pennylane(edges, nq, 1)
                if r:
                    qml = get_pennylane()
                    ms = r.median / self.config.steps * 1000
                    rows.append([f"PennyLane (v{qml.__version__})", f"{ms:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_pennylane", {"time": r.median, "cut": float(r.result), "framework": "pennylane", "problem": prob_name})
                else:
                    rows.append(["PennyLane", "N/A", "N/A", "not installed"])
            except Exception as e:
                rows.append(["PennyLane", "ERR", "ERR", str(e)[:30]])

            try:
                r = self._qaoa_qiskit(edges, nq, 1)
                if r:
                    import qiskit
                    ms = r.median / self.config.steps * 1000
                    rows.append([f"Qiskit (v{qiskit.__version__})", f"{ms:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_qiskit", {"time": r.median, "cut": float(r.result), "framework": "qiskit", "problem": prob_name})
                else:
                    rows.append(["Qiskit", "N/A", "N/A", "not installed"])
            except Exception as e:
                rows.append(["Qiskit", "ERR", "ERR", str(e)[:30]])

            table(headers, rows)
        return self._results

    def _qaoa_qforge(self, edges, n_qubits, p_layers, backend):
        from qforge.algo import QAOA, GradientDescent
        lr, steps = self.config.lr, self.config.steps
        def run():
            qaoa = QAOA(n_qubits=n_qubits, edges=edges, p_layers=p_layers, backend=backend)
            params0 = np.full(qaoa.n_params, 0.5)
            _, history = qaoa.optimize(params0, optimizer=GradientDescent(lr=lr), steps=steps)
            return -history[-1]
        return bench(run, n_warmup=1, n_runs=3)

    def _qaoa_pennylane(self, edges, n_qubits, p_layers):
        qml = get_pennylane()
        if qml is None:
            return None
        pnp = qml.numpy
        dev = qml.device("default.qubit", wires=n_qubits)

        obs_list, coeffs_list = [], []
        for (i, j) in edges:
            obs_list.append(qml.Identity(0)); coeffs_list.append(0.5)
            obs_list.append(qml.PauliZ(i) @ qml.PauliZ(j)); coeffs_list.append(-0.5)
        try:
            H_cost = qml.Hamiltonian(pnp.array(coeffs_list, requires_grad=False), obs_list)
        except Exception:
            H_cost = qml.ops.LinearCombination(coeffs_list, obs_list)

        @qml.qnode(dev, diff_method="parameter-shift")
        def qaoa_circuit(params):
            gammas, betas = params[:p_layers], params[p_layers:]
            for w in range(n_qubits): qml.Hadamard(wires=w)
            for l in range(p_layers):
                for (i, j) in edges:
                    qml.CNOT(wires=[i, j]); qml.RZ(-gammas[l], wires=j); qml.CNOT(wires=[i, j])
                for w in range(n_qubits): qml.RX(2.0 * betas[l], wires=w)
            return qml.expval(H_cost)

        def neg_qaoa(params):
            return -qaoa_circuit(params)

        opt = qml.GradientDescentOptimizer(stepsize=self.config.lr)
        steps = self.config.steps

        def run():
            params = pnp.array([0.5] * (2 * p_layers), requires_grad=True)
            neg_cost = None
            for _ in range(steps):
                params, neg_cost = opt.step_and_cost(neg_qaoa, params)
            return float(-neg_cost)
        return bench(run, n_warmup=1, n_runs=3)

    def _qaoa_qiskit(self, edges, n_qubits, p_layers):
        if not get_qiskit_available():
            return None
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        pauli_list = []
        for (i, j) in edges:
            pauli_list.append(("I" * n_qubits, 0.5))
            label_zz = ["I"] * n_qubits
            label_zz[n_qubits - 1 - i] = "Z"; label_zz[n_qubits - 1 - j] = "Z"
            pauli_list.append(("".join(label_zz), -0.5))
        H_cost_qk = SparsePauliOp.from_list(pauli_list)

        gamma = ParameterVector("g", p_layers)
        beta = ParameterVector("b", p_layers)
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for l in range(p_layers):
            for (i, j) in edges:
                qc.cx(i, j); qc.rz(-gamma[l], j); qc.cx(i, j)
            for w in range(n_qubits): qc.rx(2.0 * beta[l], w)

        estimator = StatevectorEstimator()
        lr, steps = self.config.lr, self.config.steps

        def evaluate(params):
            job = estimator.run([(qc, H_cost_qk, params)])
            return float(job.result()[0].data.evs)

        def grad_param_shift(params):
            g = np.zeros_like(params)
            denom = 2.0 * np.sin(SHIFT)
            for i in range(len(params)):
                pp = params.copy(); pp[i] += SHIFT
                pm = params.copy(); pm[i] -= SHIFT
                g[i] = (evaluate(pp) - evaluate(pm)) / denom
            return g

        def run():
            params = np.array([0.5] * (2 * p_layers))
            cut = None
            for _ in range(steps):
                cut = evaluate(params)
                params += lr * grad_param_shift(params)
            return cut
        return bench(run, n_warmup=1, n_runs=3)
