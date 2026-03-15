# -*- coding: utf-8 -*-
"""Category 1: Primitive Gate Operations benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, BenchConfig, bench, section, table,
    get_pennylane, get_qiskit_available,
)


class GatesBenchmarkSuite(BaseBenchmarkSuite):
    name = "gates"
    description = "Primitive Gate Operations (us/gate)"

    GATES = ["H", "X", "RX", "RZ", "CNOT", "SWAP", "CCNOT"]
    QUBIT_OPTIONS = [4, 8, 12, 16, 20]
    N_REPS = 1000

    def run(self):
        qubit_list = [n for n in self.QUBIT_OPTIONS if n <= self.config.max_qubits]
        section(f"CATEGORY 1: {self.description}",
                f"Microseconds per gate (median of {self.config.n_runs} runs, {self.N_REPS} reps each)")

        for nq in qubit_list:
            print(f"\n  --- {nq} qubits ---")
            headers = ["Gate", "Qforge(us)", "PennyLane(us)", "Qiskit(us)", "Speedup"]
            rows = []
            for g in self.GATES:
                if g == "CCNOT" and nq < 3:
                    continue
                t_qf = self._bench_qforge(g, nq)
                t_pl = self._bench_pennylane(g, nq)
                t_qk = self._bench_qiskit(g, nq)

                times = [t for t in [t_qf, t_pl, t_qk] if t is not None]
                slowest = max(times) if times else 1
                speedup = f"{slowest / t_qf:.1f}x" if t_qf else "N/A"

                rows.append([
                    g,
                    f"{t_qf:.2f}" if t_qf else "N/A",
                    f"{t_pl:.2f}" if t_pl else "N/A",
                    f"{t_qk:.2f}" if t_qk else "N/A",
                    speedup,
                ])
                self._store(f"{g}_{nq}q", {
                    "gate": g, "qubits": nq,
                    "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk,
                })
            table(headers, rows)
        return self._results

    # -- Qforge ----------------------------------------------------------

    def _bench_qforge(self, gate_name, n_qubits):
        from qforge import _HAS_CPP
        if not _HAS_CPP:
            return None
        from qforge.circuit import Qubit
        from qforge import gates as G

        gate_fn, args = {
            "H":     (G.H,     lambda wf: (wf, 0)),
            "X":     (G.X,     lambda wf: (wf, 0)),
            "RX":    (G.RX,    lambda wf: (wf, 0, np.pi / 3)),
            "RZ":    (G.RZ,    lambda wf: (wf, 0, np.pi / 6)),
            "CNOT":  (G.CNOT,  lambda wf: (wf, 0, 1)),
            "SWAP":  (G.SWAP,  lambda wf: (wf, 0, 1)),
            "CCNOT": (G.CCNOT, lambda wf: (wf, 0, 1, 2)),
        }[gate_name]
        n_reps = self.N_REPS

        def run():
            wf = Qubit(n_qubits, backend="cpu")
            a = args(wf)
            for _ in range(n_reps):
                gate_fn(*a)
            _ = wf.amplitude[0]

        r = bench(run, n_warmup=2, n_runs=self.config.n_runs)
        return r.median / n_reps * 1e6

    # -- PennyLane -------------------------------------------------------

    def _bench_pennylane(self, gate_name, n_qubits):
        qml = get_pennylane()
        if qml is None:
            return None

        dev = qml.device("default.qubit", wires=n_qubits)
        gate_map = {
            "H":     lambda: qml.Hadamard(wires=0),
            "X":     lambda: qml.PauliX(wires=0),
            "RX":    lambda: qml.RX(np.pi / 3, wires=0),
            "RZ":    lambda: qml.RZ(np.pi / 6, wires=0),
            "CNOT":  lambda: qml.CNOT(wires=[0, 1]),
            "SWAP":  lambda: qml.SWAP(wires=[0, 1]),
            "CCNOT": lambda: qml.Toffoli(wires=[0, 1, 2]),
        }
        gate_fn = gate_map[gate_name]
        n_reps = self.N_REPS

        @qml.qnode(dev)
        def circuit():
            for _ in range(n_reps):
                gate_fn()
            return qml.state()

        r = bench(circuit, n_warmup=2, n_runs=self.config.n_runs)
        return r.median / n_reps * 1e6

    # -- Qiskit ----------------------------------------------------------

    def _bench_qiskit(self, gate_name, n_qubits):
        if not get_qiskit_available():
            return None
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        qc = QuantumCircuit(n_qubits)
        gate_map = {
            "H":     lambda: qc.h(0),
            "X":     lambda: qc.x(0),
            "RX":    lambda: qc.rx(np.pi / 3, 0),
            "RZ":    lambda: qc.rz(np.pi / 6, 0),
            "CNOT":  lambda: qc.cx(0, 1),
            "SWAP":  lambda: qc.swap(0, 1),
            "CCNOT": lambda: qc.ccx(0, 1, 2),
        }
        gate_map[gate_name]()
        big_qc = QuantumCircuit(n_qubits)
        for _ in range(self.N_REPS):
            big_qc.compose(qc, inplace=True)

        def run():
            sv = Statevector.from_label("0" * n_qubits)
            return sv.evolve(big_qc)

        r = bench(run, n_warmup=2, n_runs=self.config.n_runs)
        return r.median / self.N_REPS * 1e6
