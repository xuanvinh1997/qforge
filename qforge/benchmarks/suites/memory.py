# -*- coding: utf-8 -*-
"""Category 9: Memory Usage benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, measure_memory, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42


class MemoryBenchmarkSuite(BaseBenchmarkSuite):
    name = "memory"
    description = "Memory Usage"

    QUBIT_OPTIONS = [4, 8, 12, 16, 20, 24]

    def run(self):
        section(f"CATEGORY 9: {self.description}",
                "Peak Python-side memory (MB) for HEA 1-layer")

        qubit_list = [n for n in self.QUBIT_OPTIONS if n <= self.config.max_qubits]
        headers = ["Qubits", "Theoretical(MB)", "Qforge(MB)", "PennyLane(MB)", "Qiskit(MB)"]
        rows = []

        for nq in qubit_list:
            theoretical = 2**nq * 16 / (1024 * 1024)
            n_p = nq * 2
            rng = np.random.RandomState(SEED)
            params = rng.uniform(0, 2 * np.pi, n_p)
            m_qf = m_pl = m_qk = None

            from qforge import _HAS_CPP
            if _HAS_CPP:
                from qforge.circuit import Qubit
                from qforge.algo.ansatz import hardware_efficient_ansatz
                def fn_qf(nq=nq, params=params):
                    wf = Qubit(nq, backend="cpu")
                    hardware_efficient_ansatz(wf, params, 1)
                    _ = wf.amplitude[0]
                try:
                    m_qf = measure_memory(fn_qf)
                except Exception:
                    pass

            qml = get_pennylane()
            if qml is not None:
                dev = qml.device("default.qubit", wires=nq)
                @qml.qnode(dev)
                def pl_fn(params=params, nq=nq):
                    idx = 0
                    for layer in range(2):
                        for q in range(nq):
                            qml.RY(params[idx], wires=q); idx += 1
                        if layer < 1:
                            for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
                    return qml.state()
                try:
                    m_pl = measure_memory(pl_fn)
                except Exception:
                    pass

            if get_qiskit_available():
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                qc = QuantumCircuit(nq)
                idx = 0
                for layer in range(2):
                    for q in range(nq):
                        qc.ry(params[idx], q); idx += 1
                    if layer < 1:
                        for q in range(nq - 1): qc.cx(q, q + 1)
                def fn_qk(qc=qc, nq=nq):
                    return Statevector.from_label("0" * nq).evolve(qc)
                try:
                    m_qk = measure_memory(fn_qk)
                except Exception:
                    pass

            rows.append([
                nq, f"{theoretical:.3f}",
                f"{m_qf:.3f}" if m_qf is not None else "N/A",
                f"{m_pl:.3f}" if m_pl is not None else "N/A",
                f"{m_qk:.3f}" if m_qk is not None else "N/A",
            ])
            self._store(f"{nq}q", {
                "qubits": nq, "theoretical": theoretical,
                "qforge": m_qf, "pennylane": m_pl, "qiskit": m_qk,
            })

        table(headers, rows)
        print("\n  Note: Qforge C++ allocations are not tracked by tracemalloc.")
        return self._results
