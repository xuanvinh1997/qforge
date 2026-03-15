# -*- coding: utf-8 -*-
"""Category 8: Accuracy & Correctness benchmark."""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, section, table,
    get_pennylane, get_qiskit_available,
)

SEED = 42


class AccuracyBenchmarkSuite(BaseBenchmarkSuite):
    name = "accuracy"
    description = "Accuracy & Correctness"

    def run(self):
        section(f"CATEGORY 8: {self.description}",
                "Max |amplitude difference| between frameworks")
        headers = ["Test", "QF vs PL", "QF vs QK", "PL vs QK"]
        rows = []

        # Bell state
        sv_qf, sv_pl, sv_qk = self._bell_states()
        rows.append(["Bell (2q)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])
        self._store("bell", self._diffs_dict(sv_qf, sv_pl, sv_qk))

        # GHZ states
        for nq in [4, 8, 12]:
            if nq > self.config.max_qubits:
                break
            sv_qf, sv_pl, sv_qk = self._ghz_states(nq)
            rows.append([f"GHZ ({nq}q)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])
            self._store(f"ghz_{nq}q", self._diffs_dict(sv_qf, sv_pl, sv_qk))

        # Random circuit
        nq_rand = min(8, self.config.max_qubits)
        if nq_rand >= 4:
            sv_qf, sv_pl, sv_qk = self._random_circuit(nq_rand)
            rows.append([f"Random ({nq_rand}q, 50g)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])
            self._store(f"random_{nq_rand}q", self._diffs_dict(sv_qf, sv_pl, sv_qk))

        table(headers, rows)
        return self._results

    @staticmethod
    def _diffs_dict(sv_qf, sv_pl, sv_qk):
        return {
            "qf_pl": float(np.max(np.abs(sv_qf - sv_pl))) if sv_qf is not None and sv_pl is not None else None,
            "qf_qk": float(np.max(np.abs(sv_qf - sv_qk))) if sv_qf is not None and sv_qk is not None else None,
            "pl_qk": float(np.max(np.abs(sv_pl - sv_qk))) if sv_pl is not None and sv_qk is not None else None,
        }

    def _bell_states(self):
        return self._build_circuit(2, lambda wf, H, CNOT: (H(wf, 0), CNOT(wf, 0, 1)))

    def _ghz_states(self, nq):
        def build(wf, H, CNOT):
            H(wf, 0)
            for q in range(nq - 1): CNOT(wf, q, q + 1)
        return self._build_circuit(nq, build)

    def _random_circuit(self, nq):
        ops = self._build_random_ops(nq)
        def build_qf(wf, H, CNOT):
            from qforge.gates import RX, RY, RZ
            gmap = {"H": H, "CNOT": CNOT, "RX": RX, "RY": RY, "RZ": RZ}
            for op, a, b in ops:
                if op == "CNOT": gmap[op](wf, a, b)
                elif b is not None: gmap[op](wf, a, b)
                else: gmap[op](wf, a)

        # Qforge
        sv_qf = None
        from qforge import _HAS_CPP
        if _HAS_CPP:
            from qforge.circuit import Qubit
            from qforge.gates import H, CNOT
            wf = Qubit(nq, backend="cpu")
            build_qf(wf, H, CNOT)
            sv_qf = wf.amplitude.copy()

        # PennyLane
        sv_pl = None
        qml = get_pennylane()
        if qml is not None:
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def rand_pl():
                for op, a, b in ops:
                    if op == "H": qml.Hadamard(wires=a)
                    elif op == "CNOT": qml.CNOT(wires=[a, b])
                    elif op == "RX": qml.RX(b, wires=a)
                    elif op == "RY": qml.RY(b, wires=a)
                    elif op == "RZ": qml.RZ(b, wires=a)
                return qml.state()
            sv_pl = np.array(rand_pl())

        # Qiskit
        sv_qk = None
        if get_qiskit_available():
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(nq)
            for op, a, b in ops:
                if op == "H": qc.h(a)
                elif op == "CNOT": qc.cx(a, b)
                elif op == "RX": qc.rx(b, a)
                elif op == "RY": qc.ry(b, a)
                elif op == "RZ": qc.rz(b, a)
            sv_qk = _get_statevector_qiskit(qc, nq)

        return sv_qf, sv_pl, sv_qk

    def _build_circuit(self, nq, build_fn):
        """Build same circuit in all 3 frameworks, return statevectors."""
        sv_qf = sv_pl = sv_qk = None

        from qforge import _HAS_CPP
        if _HAS_CPP:
            from qforge.circuit import Qubit
            from qforge.gates import H, CNOT
            wf = Qubit(nq, backend="cpu")
            build_fn(wf, H, CNOT)
            sv_qf = wf.amplitude.copy()

        qml = get_pennylane()
        if qml is not None:
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def pl_circuit():
                class _W:
                    @staticmethod
                    def h(wf, q): qml.Hadamard(wires=q)
                    @staticmethod
                    def cnot(wf, c, t): qml.CNOT(wires=[c, t])
                build_fn(None, _W.h, _W.cnot)
                return qml.state()
            sv_pl = np.array(pl_circuit())

        if get_qiskit_available():
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(nq)
            class _QK:
                @staticmethod
                def h(wf, q): qc.h(q)
                @staticmethod
                def cnot(wf, c, t): qc.cx(c, t)
            build_fn(None, _QK.h, _QK.cnot)
            sv_qk = _get_statevector_qiskit(qc, nq)

        return sv_qf, sv_pl, sv_qk

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


def _max_diff(a, b):
    if a is None or b is None:
        return "N/A"
    return f"{np.max(np.abs(a - b)):.2e}"


def _get_statevector_qiskit(qc, nq):
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_label("0" * nq).evolve(qc)
    data = np.array(sv.data)
    reordered = np.zeros_like(data)
    for i in range(len(data)):
        rev = int(f"{i:0{nq}b}"[::-1], 2)
        reordered[rev] = data[i]
    return reordered
