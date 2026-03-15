# -*- coding: utf-8 -*-
"""Category 2: Circuit Execution Patterns benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42


class CircuitsBenchmarkSuite(BaseBenchmarkSuite):
    name = "circuits"
    description = "Circuit Execution Patterns"

    QUBIT_OPTIONS = [4, 8, 12, 16, 20]

    def run(self):
        qubit_list = [n for n in self.QUBIT_OPTIONS if n <= self.config.max_qubits]
        section(f"CATEGORY 2: {self.description}",
                f"Total execution time in ms (median of {self.config.n_runs} runs)")

        patterns = [
            ("H+CNOT (d=10)", self._hcnot_qf, self._hcnot_pl, self._hcnot_qk),
            ("QFT-like",       self._qft_qf,   self._qft_pl,   self._qft_qk),
            ("Random (50g)",   self._rand_qf,   self._rand_pl,   self._rand_qk),
            ("HEA (3L)",       self._hea_qf,    self._hea_pl,    self._hea_qk),
        ]

        for pat_name, build_qf, build_pl, build_qk in patterns:
            print(f"\n  --- {pat_name} ---")
            headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "QF speedup"]
            rows = []
            for nq in qubit_list:
                t_qf = t_pl = t_qk = None
                try:
                    fn = build_qf(nq)
                    if fn:
                        t_qf = bench(fn, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass
                try:
                    fn = build_pl(nq)
                    if fn:
                        t_pl = bench(fn, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass
                try:
                    fn = build_qk(nq)
                    if fn:
                        t_qk = bench(fn, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass

                others = [t for t in [t_pl, t_qk] if t is not None]
                speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"
                rows.append([
                    nq,
                    f"{t_qf:.2f}" if t_qf else "N/A",
                    f"{t_pl:.2f}" if t_pl else "N/A",
                    f"{t_qk:.2f}" if t_qk else "N/A",
                    speedup,
                ])
                self._store(f"{pat_name}_{nq}q", {
                    "pattern": pat_name, "qubits": nq,
                    "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk,
                })
            table(headers, rows)
        return self._results

    # ── H+CNOT ──────────────────────────────────────────────────────────

    def _hcnot_qf(self, nq, depth=10):
        from qforge.circuit import Qubit
        from qforge.gates import H, CNOT
        def run():
            wf = Qubit(nq, backend="cpu")
            for _ in range(depth):
                for q in range(nq): H(wf, q)
                for q in range(nq - 1): CNOT(wf, q, q + 1)
            _ = wf.amplitude[0]
        return run

    def _hcnot_pl(self, nq, depth=10):
        qml = get_pennylane()
        if qml is None: return None
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def circuit():
            for _ in range(depth):
                for q in range(nq): qml.Hadamard(wires=q)
                for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return circuit

    def _hcnot_qk(self, nq, depth=10):
        if not get_qiskit_available(): return None
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(nq)
        for _ in range(depth):
            qc.h(range(nq))
            for q in range(nq - 1): qc.cx(q, q + 1)
        def run():
            return Statevector.from_label("0" * nq).evolve(qc)
        return run

    # ── QFT ─────────────────────────────────────────────────────────────

    def _qft_qf(self, nq):
        from qforge.circuit import Qubit
        from qforge.gates import H, CPhase
        def run():
            wf = Qubit(nq, backend="cpu")
            for i in range(nq):
                H(wf, i)
                for j in range(i + 1, nq):
                    CPhase(wf, j, i, np.pi / (2 ** (j - i)))
            _ = wf.amplitude[0]
        return run

    def _qft_pl(self, nq):
        qml = get_pennylane()
        if qml is None: return None
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def circuit():
            for i in range(nq):
                qml.Hadamard(wires=i)
                for j in range(i + 1, nq):
                    qml.ControlledPhaseShift(np.pi / (2 ** (j - i)), wires=[j, i])
            return qml.state()
        return circuit

    def _qft_qk(self, nq):
        if not get_qiskit_available(): return None
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(nq)
        for i in range(nq):
            qc.h(i)
            for j in range(i + 1, nq):
                qc.cp(np.pi / (2 ** (j - i)), j, i)
        def run():
            return Statevector.from_label("0" * nq).evolve(qc)
        return run

    # ── Random ──────────────────────────────────────────────────────────

    def _rand_qf(self, nq, n_gates=50):
        from qforge.circuit import Qubit
        from qforge.gates import H, CNOT, RX, RY, RZ
        ops = self._build_random_ops(nq, n_gates)
        gate_map = {"H": H, "CNOT": CNOT, "RX": RX, "RY": RY, "RZ": RZ}
        def run():
            wf = Qubit(nq, backend="cpu")
            for op_name, a, b in ops:
                if op_name == "CNOT": gate_map[op_name](wf, a, b)
                elif b is not None: gate_map[op_name](wf, a, b)
                else: gate_map[op_name](wf, a)
            _ = wf.amplitude[0]
        return run

    def _rand_pl(self, nq, n_gates=50):
        qml = get_pennylane()
        if qml is None: return None
        ops = self._build_random_ops(nq, n_gates)
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def circuit():
            for op_name, a, b in ops:
                if op_name == "H": qml.Hadamard(wires=a)
                elif op_name == "CNOT": qml.CNOT(wires=[a, b])
                elif op_name == "RX": qml.RX(b, wires=a)
                elif op_name == "RY": qml.RY(b, wires=a)
                elif op_name == "RZ": qml.RZ(b, wires=a)
            return qml.state()
        return circuit

    def _rand_qk(self, nq, n_gates=50):
        if not get_qiskit_available(): return None
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        ops = self._build_random_ops(nq, n_gates)
        qc = QuantumCircuit(nq)
        for op_name, a, b in ops:
            if op_name == "H": qc.h(a)
            elif op_name == "CNOT": qc.cx(a, b)
            elif op_name == "RX": qc.rx(b, a)
            elif op_name == "RY": qc.ry(b, a)
            elif op_name == "RZ": qc.rz(b, a)
        def run():
            return Statevector.from_label("0" * nq).evolve(qc)
        return run

    # ── HEA ─────────────────────────────────────────────────────────────

    def _hea_qf(self, nq, n_layers=3):
        from qforge.circuit import Qubit
        from qforge.algo.ansatz import hardware_efficient_ansatz
        rng = np.random.RandomState(SEED)
        params = rng.uniform(0, 2 * np.pi, nq * (n_layers + 1))
        def run():
            wf = Qubit(nq, backend="cpu")
            hardware_efficient_ansatz(wf, params, n_layers)
            _ = wf.amplitude[0]
        return run

    def _hea_pl(self, nq, n_layers=3):
        qml = get_pennylane()
        if qml is None: return None
        dev = qml.device("default.qubit", wires=nq)
        rng = np.random.RandomState(SEED)
        params = rng.uniform(0, 2 * np.pi, nq * (n_layers + 1))
        @qml.qnode(dev)
        def circuit():
            idx = 0
            for layer in range(n_layers + 1):
                for q in range(nq):
                    qml.RY(params[idx], wires=q); idx += 1
                if layer < n_layers:
                    for q in range(nq - 1): qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return circuit

    def _hea_qk(self, nq, n_layers=3):
        if not get_qiskit_available(): return None
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        rng = np.random.RandomState(SEED)
        params = rng.uniform(0, 2 * np.pi, nq * (n_layers + 1))
        qc = QuantumCircuit(nq)
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(nq):
                qc.ry(params[idx], q); idx += 1
            if layer < n_layers:
                for q in range(nq - 1): qc.cx(q, q + 1)
        def run():
            return Statevector.from_label("0" * nq).evolve(qc)
        return run

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_random_ops(nq, n_gates=50):
        rng = np.random.RandomState(SEED)
        ops = []
        for _ in range(n_gates):
            g = rng.choice(["H", "CNOT", "RX", "RY", "RZ"])
            if g == "H":
                ops.append(("H", rng.randint(nq), None))
            elif g == "CNOT" and nq >= 2:
                c, t = rng.choice(nq, 2, replace=False)
                ops.append(("CNOT", int(c), int(t)))
            elif g in ("RX", "RY", "RZ"):
                ops.append((g, rng.randint(nq), rng.uniform(0, 2 * np.pi)))
            else:
                ops.append(("H", rng.randint(nq), None))
        return ops
