# -*- coding: utf-8 -*-
"""Category 6: Measurement Operations benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)


class MeasurementBenchmarkSuite(BaseBenchmarkSuite):
    name = "measurement"
    description = "Measurement Operations"

    QUBIT_OPTIONS = [4, 8, 12, 16]

    def run(self):
        section(f"CATEGORY 6: {self.description}", "Time per operation")
        qubit_list = [n for n in self.QUBIT_OPTIONS if n <= self.config.max_qubits]
        self._bench_pauli_z(qubit_list)
        self._bench_sampling(qubit_list)
        return self._results

    def _prepare_qforge(self, nq):
        from qforge.circuit import Qubit
        from qforge.gates import H, CNOT
        wf = Qubit(nq, backend="cpu")
        for q in range(nq): H(wf, q)
        for q in range(nq - 1): CNOT(wf, q, q + 1)
        return wf

    def _bench_pauli_z(self, qubit_list):
        print("\n  --- Single Pauli-Z expectation (us) ---")
        headers = ["Qubits", "Qforge(us)", "PennyLane(us)", "Qiskit(us)", "Speedup"]
        rows = []

        for nq in qubit_list:
            t_qf = t_pl = t_qk = None

            from qforge import _HAS_CPP
            if _HAS_CPP:
                from qforge.measurement import pauli_expectation
                wf = self._prepare_qforge(nq)
                def run_qf(wf=wf):
                    return pauli_expectation(wf, 0, 'Z')
                t_qf = bench(run_qf, n_warmup=3, n_runs=self.config.n_runs).median * 1e6

            qml = get_pennylane()
            if qml is not None:
                dev = qml.device("default.qubit", wires=nq)
                @qml.qnode(dev)
                def pl_circuit(nq=nq):
                    for q in range(nq): qml.Hadamard(wires=q)
                    for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
                    return qml.expval(qml.PauliZ(0))
                t_pl = bench(pl_circuit, n_warmup=3, n_runs=self.config.n_runs).median * 1e6

            if get_qiskit_available():
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import SparsePauliOp, Statevector
                qc = QuantumCircuit(nq)
                qc.h(range(nq))
                for q in range(nq - 1): qc.cx(q, q + 1)
                z_label = ["I"] * nq; z_label[nq - 1] = "Z"
                z_op = SparsePauliOp.from_list([("".join(z_label), 1.0)])
                def run_qk(qc=qc, z_op=z_op, nq=nq):
                    sv = Statevector.from_label("0" * nq).evolve(qc)
                    return sv.expectation_value(z_op).real
                t_qk = bench(run_qk, n_warmup=3, n_runs=self.config.n_runs).median * 1e6

            others = [t for t in [t_pl, t_qk] if t is not None]
            speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"
            rows.append([nq, f"{t_qf:.1f}" if t_qf else "N/A", f"{t_pl:.1f}" if t_pl else "N/A",
                         f"{t_qk:.1f}" if t_qk else "N/A", speedup])
            self._store(f"pauliZ_{nq}q", {"qubits": nq, "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})
        table(headers, rows)

    def _bench_sampling(self, qubit_list):
        n_shots = 10000
        print(f"\n  --- Sampling ({n_shots:,} shots, ms) ---")
        headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "Speedup"]
        rows = []

        for nq in qubit_list:
            t_qf = t_pl = t_qk = None

            from qforge import _HAS_CPP
            if _HAS_CPP:
                from qforge.measurement import measure_all
                wf = self._prepare_qforge(nq)
                def run_qf_s(wf=wf):
                    return measure_all(wf, n_shots)
                t_qf = bench(run_qf_s, n_warmup=2, n_runs=self.config.n_runs).median * 1000

            qml = get_pennylane()
            if qml is not None:
                dev = qml.device("default.qubit", wires=nq, shots=n_shots)
                @qml.qnode(dev)
                def pl_sample(nq=nq):
                    for q in range(nq): qml.Hadamard(wires=q)
                    for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
                    return qml.sample()
                t_pl = bench(pl_sample, n_warmup=2, n_runs=self.config.n_runs).median * 1000

            if get_qiskit_available():
                from qiskit import QuantumCircuit
                from qiskit.primitives import StatevectorSampler
                qc = QuantumCircuit(nq); qc.h(range(nq))
                for q in range(nq - 1): qc.cx(q, q + 1)
                qc.measure_all()
                sampler = StatevectorSampler()
                def run_qk_s(qc=qc):
                    return sampler.run([qc], shots=n_shots).result()
                t_qk = bench(run_qk_s, n_warmup=2, n_runs=self.config.n_runs).median * 1000

            others = [t for t in [t_pl, t_qk] if t is not None]
            speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"
            rows.append([nq, f"{t_qf:.2f}" if t_qf else "N/A", f"{t_pl:.2f}" if t_pl else "N/A",
                         f"{t_qk:.2f}" if t_qk else "N/A", speedup])
            self._store(f"sampling_{nq}q", {"qubits": nq, "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})
        table(headers, rows)
