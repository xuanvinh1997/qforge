# -*- coding: utf-8 -*-
"""Category 7: Scalability benchmark — HEA forward pass vs qubit count."""
from __future__ import annotations

import os
import time
import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42
_BENCH_CUDA = os.environ.get("QFORGE_BENCH_CUDA", "") == "1"


class ScalingBenchmarkSuite(BaseBenchmarkSuite):
    name = "scaling"
    description = "Scalability (HEA 1-layer forward pass)"

    def run(self):
        from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
        section(f"CATEGORY 7: {self.description}",
                "Time (ms) vs qubit count")

        qubit_list = list(range(2, min(self.config.max_qubits, 24) + 1, 2))
        bench_cuda = _HAS_CUDA and _BENCH_CUDA
        timeout = 60.0

        headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)"]
        if bench_cuda:
            headers.insert(2, "QF-CUDA(ms)")
        rows = []
        skip_pl = skip_qk = False

        for nq in qubit_list:
            n_p = nq * 2
            rng = np.random.RandomState(SEED)
            params = rng.uniform(0, 2 * np.pi, n_p)
            row = [nq]
            t_qf = t_cuda = t_pl = t_qk = None

            # Qforge CPU
            if _HAS_CPP:
                from qforge.circuit import Qubit
                from qforge.algo.ansatz import hardware_efficient_ansatz
                def run_qf(nq=nq, params=params):
                    wf = Qubit(nq, backend="cpu")
                    hardware_efficient_ansatz(wf, params, 1)
                    _ = wf.amplitude[0]
                try:
                    t_qf = bench(run_qf, n_warmup=2, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass
            row.append(f"{t_qf:.3f}" if t_qf else "N/A")

            # Qforge CUDA
            if bench_cuda:
                def run_cuda(nq=nq, params=params):
                    wf = Qubit(nq, backend="cuda")
                    hardware_efficient_ansatz(wf, params, 1)
                    _ = wf.amplitude[0]
                try:
                    t_cuda = bench(run_cuda, n_warmup=2, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass
                row.append(f"{t_cuda:.3f}" if t_cuda is not None else "N/A")

            # PennyLane
            if not skip_pl:
                qml = get_pennylane()
                if qml is not None:
                    dev = qml.device("default.qubit", wires=nq)
                    @qml.qnode(dev)
                    def pl_hea(params=params, nq=nq):
                        idx = 0
                        for layer in range(2):
                            for q in range(nq):
                                qml.RY(params[idx], wires=q); idx += 1
                            if layer < 1:
                                for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
                        return qml.state()
                    try:
                        t0 = time.perf_counter()
                        r = bench(pl_hea, n_warmup=1, n_runs=3)
                        if time.perf_counter() - t0 > timeout: skip_pl = True
                        t_pl = r.median * 1000
                    except Exception:
                        pass
            row.append(f"{t_pl:.3f}" if t_pl else ("TIMEOUT" if skip_pl else "N/A"))

            # Qiskit
            if not skip_qk and get_qiskit_available():
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                qc = QuantumCircuit(nq)
                idx = 0
                for layer in range(2):
                    for q in range(nq):
                        qc.ry(params[idx], q); idx += 1
                    if layer < 1:
                        for q in range(nq - 1): qc.cx(q, q + 1)
                def run_qk(qc=qc, nq=nq):
                    return Statevector.from_label("0" * nq).evolve(qc)
                try:
                    t0 = time.perf_counter()
                    r = bench(run_qk, n_warmup=1, n_runs=3)
                    if time.perf_counter() - t0 > timeout: skip_qk = True
                    t_qk = r.median * 1000
                except Exception:
                    pass
            row.append(f"{t_qk:.3f}" if t_qk else ("TIMEOUT" if skip_qk else "N/A"))

            rows.append(row)
            self._store(f"{nq}q", {
                "qubits": nq, "qforge": t_qf, "qforge_cuda": t_cuda,
                "pennylane": t_pl, "qiskit": t_qk,
            })

        table(headers, rows)
        return self._results
