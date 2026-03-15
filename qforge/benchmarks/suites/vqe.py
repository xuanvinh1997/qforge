# -*- coding: utf-8 -*-
"""Category 3: VQE Algorithm benchmark."""
from __future__ import annotations

import os
import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42
SHIFT = np.pi / 2

# H2 Hamiltonian coefficients
VQE_COEFFS_H2 = [-1.0523, 0.3979, -0.3979, -0.0112, 0.1809]
VQE_TERMS_H2 = [[], [('Z', 0)], [('Z', 1)], [('Z', 0), ('Z', 1)], [('X', 0), ('X', 1)]]


def _lih_hamiltonian():
    rng = np.random.RandomState(SEED)
    coeffs, terms = [], []
    coeffs.append(-7.498); terms.append([])
    for q in range(6):
        coeffs.append(rng.uniform(-0.5, 0.5)); terms.append([('Z', q)])
    for i in range(5):
        coeffs.append(rng.uniform(-0.3, 0.3)); terms.append([('Z', i), ('Z', i + 1)])
    for i in range(0, 4, 2):
        coeffs.append(rng.uniform(0.05, 0.2)); terms.append([('X', i), ('X', i + 1)])
    return coeffs, terms


LIH_COEFFS, LIH_TERMS = _lih_hamiltonian()

_BENCH_CUDA = os.environ.get("QFORGE_BENCH_CUDA", "") == "1"


class VQEBenchmarkSuite(BaseBenchmarkSuite):
    name = "vqe"
    description = "VQE Algorithm"

    def run(self):
        section(f"CATEGORY 3: {self.description}",
                f"{self.config.steps} steps, lr={self.config.lr}, parameter-shift gradient")

        problems = [("H2 (2q)", VQE_COEFFS_H2, VQE_TERMS_H2, 2, 1)]
        if self.config.max_qubits >= 6:
            problems.append(("LiH-like (6q)", LIH_COEFFS, LIH_TERMS, 6, 1))

        for prob_name, coeffs, terms, nq, nl in problems:
            print(f"\n  --- {prob_name}, {nq} qubits, {nl} layer ---")
            headers = ["Framework", "ms/step", "total(s)", "energy"]
            rows = []

            # Qforge backends
            from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
            for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA and _BENCH_CUDA), ("metal", _HAS_METAL)]:
                if not available:
                    continue
                try:
                    r = self._vqe_qforge(coeffs, terms, nq, nl, backend)
                    ms_step = r.median / self.config.steps * 1000
                    rows.append([f"Qforge ({backend})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_{backend}", {"time": r.median, "energy": float(r.result), "framework": f"qforge_{backend}", "problem": prob_name})
                except Exception as e:
                    rows.append([f"Qforge ({backend})", "ERR", "ERR", str(e)[:30]])

            # PennyLane
            try:
                r = self._vqe_pennylane(coeffs, terms, nq, nl)
                if r:
                    qml = get_pennylane()
                    ms_step = r.median / self.config.steps * 1000
                    rows.append([f"PennyLane (v{qml.__version__})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_pennylane", {"time": r.median, "energy": float(r.result), "framework": "pennylane", "problem": prob_name})
                else:
                    rows.append(["PennyLane", "N/A", "N/A", "not installed"])
            except Exception as e:
                rows.append(["PennyLane", "ERR", "ERR", str(e)[:30]])

            # Qiskit
            try:
                r = self._vqe_qiskit(coeffs, terms, nq, nl)
                if r:
                    import qiskit
                    ms_step = r.median / self.config.steps * 1000
                    rows.append([f"Qiskit (v{qiskit.__version__})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                    self._store(f"{prob_name}_qiskit", {"time": r.median, "energy": float(r.result), "framework": "qiskit", "problem": prob_name})
                else:
                    rows.append(["Qiskit", "N/A", "N/A", "not installed"])
            except Exception as e:
                rows.append(["Qiskit", "ERR", "ERR", str(e)[:30]])

            table(headers, rows)
        return self._results

    def _vqe_qforge(self, coeffs, terms, n_qubits, n_layers, backend):
        from qforge.algo import Hamiltonian, VQE, GradientDescent
        H = Hamiltonian(coeffs=coeffs, terms=terms)
        n_p = VQE.n_params_hardware_efficient(n_qubits, n_layers=n_layers)
        lr, steps = self.config.lr, self.config.steps
        def run():
            vqe = VQE(n_qubits=n_qubits, hamiltonian=H, n_layers=n_layers, backend=backend)
            _, history = vqe.optimize(np.zeros(n_p), optimizer=GradientDescent(lr=lr), steps=steps)
            return history[-1]
        return bench(run, n_warmup=1, n_runs=3)

    def _vqe_pennylane(self, coeffs, terms, n_qubits, n_layers):
        qml = get_pennylane()
        if qml is None:
            return None
        pnp = qml.numpy
        dev = qml.device("default.qubit", wires=n_qubits)

        obs_list = []
        for t in terms:
            if not t:
                obs_list.append(qml.Identity(0))
            else:
                ops = []
                for pauli, q in t:
                    if pauli == 'X': ops.append(qml.PauliX(q))
                    elif pauli == 'Y': ops.append(qml.PauliY(q))
                    elif pauli == 'Z': ops.append(qml.PauliZ(q))
                ob = ops[0]
                for o in ops[1:]:
                    ob = ob @ o
                obs_list.append(ob)
        try:
            H_pl = qml.Hamiltonian(pnp.array(coeffs, requires_grad=False), obs_list)
        except Exception:
            H_pl = qml.ops.LinearCombination(coeffs, obs_list)

        n_p = n_qubits * (n_layers + 1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            idx = 0
            for layer in range(n_layers + 1):
                for q in range(n_qubits):
                    qml.RY(params[idx], wires=q); idx += 1
                if layer < n_layers:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            return qml.expval(H_pl)

        opt = qml.GradientDescentOptimizer(stepsize=self.config.lr)
        steps = self.config.steps

        def run():
            params = pnp.zeros(n_p, requires_grad=True)
            cost = None
            for _ in range(steps):
                params, cost = opt.step_and_cost(circuit, params)
            return float(cost)
        return bench(run, n_warmup=1, n_runs=3)

    def _vqe_qiskit(self, coeffs, terms, n_qubits, n_layers):
        if not get_qiskit_available():
            return None
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        pauli_list = []
        for c, t in zip(coeffs, terms):
            label = ["I"] * n_qubits
            for pauli, q in t:
                label[n_qubits - 1 - q] = pauli
            pauli_list.append(("".join(label), c))
        H_qk = SparsePauliOp.from_list(pauli_list)

        ansatz = TwoLocal(n_qubits, rotation_blocks=["ry"], entanglement_blocks="cx",
                          reps=n_layers, entanglement="linear")
        estimator = StatevectorEstimator()
        lr, steps = self.config.lr, self.config.steps

        def evaluate(params):
            job = estimator.run([(ansatz, H_qk, params)])
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
            params = np.zeros(ansatz.num_parameters)
            cost = None
            for _ in range(steps):
                cost = evaluate(params)
                params -= lr * grad_param_shift(params)
            return cost
        return bench(run, n_warmup=1, n_runs=3)
