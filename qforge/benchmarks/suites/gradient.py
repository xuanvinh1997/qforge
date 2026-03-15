# -*- coding: utf-8 -*-
"""Category 5: Gradient Computation benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42
SHIFT = np.pi / 2


class GradientBenchmarkSuite(BaseBenchmarkSuite):
    name = "gradient"
    description = "Gradient Computation"

    def run(self):
        section(f"CATEGORY 5: {self.description}", "Parameter-shift gradient time (ms)")

        configs = [(4, 1, 8), (8, 1, 16), (4, 2, 12)]
        configs = [(nq, nl, np_) for nq, nl, np_ in configs if nq <= self.config.max_qubits]

        headers = ["Config", "n_params", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "Speedup"]
        rows = []

        for nq, nl, n_p in configs:
            label = f"{nq}q/{nl}L"
            t_qf = t_pl = t_qk = None

            from qforge import _HAS_CPP
            if _HAS_CPP:
                try:
                    r = self._grad_qforge(nq, nl)
                    t_qf = r.median * 1000
                except Exception:
                    pass
            try:
                r = self._grad_pennylane(nq, nl)
                if r: t_pl = r.median * 1000
            except Exception:
                pass
            try:
                r = self._grad_qiskit(nq, nl)
                if r: t_qk = r.median * 1000
            except Exception:
                pass

            others = [t for t in [t_pl, t_qk] if t is not None]
            speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"

            rows.append([
                label, n_p,
                f"{t_qf:.1f}" if t_qf else "N/A",
                f"{t_pl:.1f}" if t_pl else "N/A",
                f"{t_qk:.1f}" if t_qk else "N/A",
                speedup,
            ])
            self._store(label, {"n_params": n_p, "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})

        table(headers, rows)
        return self._results

    def _grad_qforge(self, n_qubits, n_layers):
        from qforge.algo import Hamiltonian, VQE
        rng = np.random.default_rng(SEED)
        coeffs = rng.uniform(-1, 1, n_qubits).tolist()
        terms = [[('Z', q)] for q in range(n_qubits)]
        H = Hamiltonian(coeffs, terms)
        n_p = VQE.n_params_hardware_efficient(n_qubits, n_layers)
        vqe = VQE(n_qubits=n_qubits, hamiltonian=H, n_layers=n_layers, backend="cpu")
        params = np.zeros(n_p)
        vqe._evaluate(params); vqe.gradient(params)
        def run():
            return vqe.gradient(params)
        return bench(run, n_warmup=1, n_runs=self.config.n_runs)

    def _grad_pennylane(self, n_qubits, n_layers):
        qml = get_pennylane()
        if qml is None: return None
        pnp = qml.numpy
        dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(SEED)
        coeffs = rng.uniform(-1, 1, n_qubits).tolist()
        obs_list = [qml.PauliZ(q) for q in range(n_qubits)]
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
                    for q in range(n_qubits - 1): qml.CNOT(wires=[q, q + 1])
            return qml.expval(H_pl)

        grad_fn = qml.grad(circuit)
        def run():
            return grad_fn(pnp.zeros(n_p, requires_grad=True))
        return bench(run, n_warmup=2, n_runs=self.config.n_runs)

    def _grad_qiskit(self, n_qubits, n_layers):
        if not get_qiskit_available(): return None
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
        rng = np.random.default_rng(SEED)
        coeffs = rng.uniform(-1, 1, n_qubits).tolist()
        pauli_list = []
        for q in range(n_qubits):
            label = ["I"] * n_qubits; label[n_qubits - 1 - q] = "Z"
            pauli_list.append(("".join(label), coeffs[q]))
        H_qk = SparsePauliOp.from_list(pauli_list)
        ansatz = TwoLocal(n_qubits, rotation_blocks=["ry"], entanglement_blocks="cx",
                          reps=n_layers, entanglement="linear")
        estimator = StatevectorEstimator()

        def evaluate(params):
            return float(estimator.run([(ansatz, H_qk, params)]).result()[0].data.evs)

        def run():
            params = np.zeros(ansatz.num_parameters)
            g = np.zeros_like(params)
            denom = 2.0 * np.sin(SHIFT)
            for i in range(len(params)):
                pp = params.copy(); pp[i] += SHIFT
                pm = params.copy(); pm[i] -= SHIFT
                g[i] = (evaluate(pp) - evaluate(pm)) / denom
            return g
        return bench(run, n_warmup=2, n_runs=self.config.n_runs)
