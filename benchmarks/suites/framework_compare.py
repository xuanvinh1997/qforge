# -*- coding: utf-8 -*-
"""Fair cross-framework comparison: Qforge vs Qiskit vs PennyLane.

Methodology:
  - Warm-up runs before measurement to eliminate JIT/compilation artifacts
  - Separate phases: compile, simulate, gradient
  - Fixed hardware-efficient ansatz: RY rotations + CNOT ladder
  - Multiple gradient methods per framework
  - Statistical reporting: median + IQR over N trials
  - Circuit depth scales with qubit count: L = ceil(n/2)
"""
from __future__ import annotations

import time
import math
import numpy as np


N_WARMUP = 3
N_TRIALS = 15


def _stats(times: list[float]) -> dict:
    """Compute median, IQR, and coefficient of variation."""
    a = np.array(times)
    med = float(np.median(a))
    q25, q75 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
    mean = float(np.mean(a))
    cv = float(np.std(a) / mean) if mean > 0 else 0.0
    return {"median": med, "q25": q25, "q75": q75, "iqr": q75 - q25,
            "mean": mean, "cv": cv, "n_trials": len(times)}


def _hw_efficient_depth(n_qubits: int) -> int:
    """Fixed circuit depth: ceil(n/2)."""
    return max(math.ceil(n_qubits / 2), 1)


class FrameworkCompareSuite:
    name = "framework"
    description = "Qforge vs Qiskit vs PennyLane (fair: warm-up, phase-separated)"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        qubit_range = self.config.get("qubit_range",
                                      [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        n_trials = self.config.get("fair_trials", N_TRIALS)
        n_warmup = self.config.get("fair_warmup", N_WARMUP)

        results = {
            "methodology": {
                "n_warmup": n_warmup,
                "n_trials": n_trials,
                "circuit": "hardware-efficient ansatz: RY + CNOT ladder",
                "depth_rule": "ceil(n_qubits / 2)",
                "metrics": "median + IQR",
            },
            "simulate": {},
            "gradient": {},
        }

        for n in qubit_range:
            depth = _hw_efficient_depth(n)
            n_params = n * (depth + 1)
            np.random.seed(42)
            params = np.random.uniform(-np.pi, np.pi, n_params)

            sim_entry = {"n_qubits": n, "depth": depth, "n_params": n_params}
            grad_entry = {"n_qubits": n, "depth": depth, "n_params": n_params}

            # ---- Qforge ----
            sim_entry["qforge"] = self._bench_qforge_sim(n, depth, params,
                                                          n_warmup, n_trials)
            grad_entry["qforge"] = self._bench_qforge_grad(n, depth, params,
                                                            n_warmup, n_trials)

            # ---- Qiskit ----
            qi_sim = self._bench_qiskit_sim(n, depth, params, n_warmup, n_trials)
            if qi_sim:
                sim_entry["qiskit"] = qi_sim
            qi_grad = self._bench_qiskit_grad(n, depth, params, n_warmup, n_trials)
            if qi_grad:
                grad_entry["qiskit"] = qi_grad

            # ---- PennyLane ----
            pl_sim = self._bench_pennylane_sim(n, depth, params, n_warmup, n_trials)
            if pl_sim:
                sim_entry["pennylane"] = pl_sim
            pl_grad = self._bench_pennylane_grad(n, depth, params, n_warmup, n_trials)
            if pl_grad:
                grad_entry["pennylane"] = pl_grad

            results["simulate"][str(n)] = sim_entry
            results["gradient"][str(n)] = grad_entry

            # Progress
            fws = [k for k in sim_entry if k not in ("n_qubits", "depth", "n_params")]
            times_str = "  ".join(
                f"{fw}={sim_entry[fw]['median']:.5f}s" for fw in fws
                if isinstance(sim_entry[fw], dict)
            )
            print(f"    {n:2d}q (L={depth}): {times_str}")

        return results

    # ------------------------------------------------------------------ #
    #  Qforge benchmarks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _qforge_circuit(n, depth, params):
        """Build and run hardware-efficient ansatz in Qforge."""
        from qforge.circuit import Qubit
        from qforge.gates import RY, CNOT
        wf = Qubit(n, backend='cpu')
        idx = 0
        for layer in range(depth + 1):
            for q in range(n):
                RY(wf, q, params[idx]); idx += 1
            if layer < depth:
                for q in range(n - 1):
                    CNOT(wf, q, q + 1)
        return wf

    def _bench_qforge_sim(self, n, depth, params, n_warmup, n_trials):
        for _ in range(n_warmup):
            self._qforge_circuit(n, depth, params)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            self._qforge_circuit(n, depth, params)
            times.append(time.perf_counter() - t0)
        return _stats(times)

    def _bench_qforge_grad(self, n, depth, params, n_warmup, n_trials):
        from qforge.algo import Hamiltonian
        from qforge.algo.gradient import parameter_shift

        ham = Hamiltonian([-1.0] * n, [[('Z', i)] for i in range(n)])

        def cost(p):
            wf = self._qforge_circuit(n, depth, p)
            return ham.expectation(wf)

        # Warm-up
        for _ in range(n_warmup):
            parameter_shift(cost, params.copy())

        # Parameter-shift
        ps_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            parameter_shift(cost, params.copy())
            ps_times.append(time.perf_counter() - t0)

        result = {"parameter_shift": _stats(ps_times)}

        # Adjoint diff (if available)
        try:
            from qforge.ir import Circuit as IRCircuit
            from qforge.algo.adjoint_diff import adjoint_differentiation

            ir = self._build_ir_circuit(n, depth)
            for _ in range(n_warmup):
                adjoint_differentiation(ir, ham, params.copy(), backend='cpu')
            adj_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                adjoint_differentiation(ir, ham, params.copy(), backend='cpu')
                adj_times.append(time.perf_counter() - t0)
            result["adjoint"] = _stats(adj_times)
        except Exception:
            pass

        return result

    @staticmethod
    def _build_ir_circuit(n, depth):
        """Build qforge IR Circuit for adjoint diff."""
        from qforge.ir import Circuit as IRCircuit
        ir = IRCircuit(n)
        idx = 0
        for layer in range(depth + 1):
            for q in range(n):
                ir.ry(q, f"p{idx}"); idx += 1
            if layer < depth:
                for q in range(n - 1):
                    ir.cnot(q, q + 1)
        return ir

    # ------------------------------------------------------------------ #
    #  Qiskit benchmarks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _qiskit_circuit(n, depth, params):
        """Build hardware-efficient ansatz as a Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        idx = 0
        for layer in range(depth + 1):
            for q in range(n):
                qc.ry(params[idx], q); idx += 1
            if layer < depth:
                for q in range(n - 1):
                    qc.cx(q, q + 1)
        return qc

    def _bench_qiskit_sim(self, n, depth, params, n_warmup, n_trials):
        try:
            from qiskit.quantum_info import Statevector
        except ImportError:
            return None

        def run():
            qc = self._qiskit_circuit(n, depth, params)
            Statevector(qc)

        for _ in range(n_warmup):
            run()
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        return _stats(times)

    def _bench_qiskit_grad(self, n, depth, params, n_warmup, n_trials):
        try:
            from qiskit.quantum_info import Statevector, SparsePauliOp
        except ImportError:
            return None

        # Build observable
        pauli_list = []
        for i in range(n):
            label = ['I'] * n
            label[i] = 'Z'
            pauli_list.append((''.join(label), -1.0))
        obs = SparsePauliOp.from_list(pauli_list)

        def cost(p):
            qc = self._qiskit_circuit(n, depth, p)
            sv = Statevector(qc)
            return float(np.real(sv.expectation_value(obs)))

        def param_shift_qiskit(p):
            shift = np.pi / 2
            grad = np.zeros(len(p))
            for i in range(len(p)):
                pp = p.copy(); pp[i] += shift
                pm = p.copy(); pm[i] -= shift
                grad[i] = (cost(pp) - cost(pm)) / 2
            return grad

        # Warm-up
        for _ in range(n_warmup):
            param_shift_qiskit(params.copy())

        ps_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            param_shift_qiskit(params.copy())
            ps_times.append(time.perf_counter() - t0)

        return {"parameter_shift": _stats(ps_times)}

    # ------------------------------------------------------------------ #
    #  PennyLane benchmarks
    # ------------------------------------------------------------------ #

    def _bench_pennylane_sim(self, n, depth, params, n_warmup, n_trials):
        try:
            import pennylane as qml
        except ImportError:
            return None

        dev = qml.device('default.qubit', wires=n)

        @qml.qnode(dev)
        def circuit(p):
            idx = 0
            for layer in range(depth + 1):
                for q in range(n):
                    qml.RY(p[idx], wires=q); idx += 1
                if layer < depth:
                    for q in range(n - 1):
                        qml.CNOT(wires=[q, q + 1])
            return qml.state()

        pl_params = qml.numpy.array(params.copy(), requires_grad=False)

        # Warm-up (triggers JIT/compilation)
        for _ in range(n_warmup):
            circuit(pl_params)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            circuit(pl_params)
            times.append(time.perf_counter() - t0)
        return _stats(times)

    def _bench_pennylane_grad(self, n, depth, params, n_warmup, n_trials):
        try:
            import pennylane as qml
        except ImportError:
            return None

        result = {}

        # --- Parameter-shift ---
        dev_ps = qml.device('default.qubit', wires=n)

        @qml.qnode(dev_ps, diff_method='parameter-shift')
        def cost_ps(p):
            idx = 0
            for layer in range(depth + 1):
                for q in range(n):
                    qml.RY(p[idx], wires=q); idx += 1
                if layer < depth:
                    for q in range(n - 1):
                        qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.sum(*[-1.0 * qml.PauliZ(i) for i in range(n)]))

        grad_ps = qml.grad(cost_ps)
        pl_params = qml.numpy.array(params.copy(), requires_grad=True)

        for _ in range(n_warmup):
            grad_ps(pl_params)

        ps_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            grad_ps(pl_params)
            ps_times.append(time.perf_counter() - t0)
        result["parameter_shift"] = _stats(ps_times)

        # --- Adjoint diff (PennyLane) ---
        try:
            dev_adj = qml.device('default.qubit', wires=n)

            @qml.qnode(dev_adj, diff_method='adjoint')
            def cost_adj(p):
                idx = 0
                for layer in range(depth + 1):
                    for q in range(n):
                        qml.RY(p[idx], wires=q); idx += 1
                    if layer < depth:
                        for q in range(n - 1):
                            qml.CNOT(wires=[q, q + 1])
                return qml.expval(qml.sum(*[-1.0 * qml.PauliZ(i) for i in range(n)]))

            grad_adj = qml.grad(cost_adj)

            for _ in range(n_warmup):
                grad_adj(pl_params)

            adj_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                grad_adj(pl_params)
                adj_times.append(time.perf_counter() - t0)
            result["adjoint"] = _stats(adj_times)
        except Exception:
            pass

        # --- Backprop (PennyLane default.qubit supports it) ---
        try:
            dev_bp = qml.device('default.qubit', wires=n)

            @qml.qnode(dev_bp, diff_method='backprop')
            def cost_bp(p):
                idx = 0
                for layer in range(depth + 1):
                    for q in range(n):
                        qml.RY(p[idx], wires=q); idx += 1
                    if layer < depth:
                        for q in range(n - 1):
                            qml.CNOT(wires=[q, q + 1])
                return qml.expval(qml.sum(*[-1.0 * qml.PauliZ(i) for i in range(n)]))

            grad_bp = qml.grad(cost_bp)

            for _ in range(n_warmup):
                grad_bp(pl_params)

            bp_times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                grad_bp(pl_params)
                bp_times.append(time.perf_counter() - t0)
            result["backprop"] = _stats(bp_times)
        except Exception:
            pass

        return result
