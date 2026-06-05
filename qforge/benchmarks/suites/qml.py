# -*- coding: utf-8 -*-
"""QML algorithm benchmark suite.

Small, deterministic benchmarks for QML circuits inspired by:

* Havlicek et al., arXiv:1804.11326 -- fidelity quantum kernels.
* Perez-Salinas et al., arXiv:1907.02085 -- data re-uploading classifiers.
* Cong, Choi, Lukin, arXiv:1810.03787 -- QCNN-style pooling/readout.
"""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
    get_pennylane, get_qiskit_available,
)


class QMLBenchmarkSuite(BaseBenchmarkSuite):
    name = "qml"
    description = "QML Algorithms"

    def run(self):
        section(
            f"CATEGORY QML: {self.description}",
            "Ansatz forward pass, fidelity kernel, and tiny classifier training",
        )

        rows = []
        qubits = self._qubit_grid()

        for nq in qubits:
            ansatz = self._benchmark_ansatz(nq)
            rows.append([
                f"Strongly entangling ansatz {nq}q",
                f"{ansatz['qforge_ms']:.3f}",
                _fmt_ms(ansatz.get("pennylane_ms")),
                _fmt_ms(ansatz.get("qiskit_ms")),
                f"<Z0>={ansatz['observable']:.6f}",
            ])
            self._store(f"ansatz_{nq}q", ansatz)

        for nq in qubits:
            kernel = self._benchmark_qsvm_kernel(nq)
            rows.append([
                f"QSVM fidelity kernel {nq}q",
                f"{kernel['qforge_ms']:.3f}",
                _fmt_ms(kernel.get("pennylane_ms")),
                "N/A",
                f"min_eig={kernel['min_eig']:.2e}",
            ])
            self._store(f"kernel_{nq}q", kernel)

        for nq in self._training_qubits(qubits):
            training = self._benchmark_classifiers(nq)
            for name, data in training.items():
                rows.append([
                    f"{name} {nq}q",
                    f"{data['qforge_ms']:.3f}",
                    "N/A",
                    "N/A",
                    f"loss={data['final_loss']:.6f}",
                ])
                self._store(f"train_{data['model']}_{nq}q", data)

        table(
            ["Task", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "Result"],
            rows,
        )
        return self._results

    def _qubit_grid(self):
        if self.config.qml_qubits is not None:
            return [q for q in self.config.qml_qubits if q <= self.config.max_qubits]
        candidates = [2, 4, 6, 8, 10, 12]
        return [q for q in candidates if q <= self.config.max_qubits]

    @staticmethod
    def _training_qubits(qubits):
        # Parameter-shift training is intentionally capped here; the forward
        # and kernel benchmarks above cover the larger statevector paths.
        return [q for q in qubits if q in (2, 4, 8)]

    def _benchmark_ansatz(self, n_qubits):
        from qforge.circuit import Qubit
        from qforge.algo.ansatz import strongly_entangling_ansatz
        from qforge.measurement import pauli_expectation

        n_layers = 2
        params = np.linspace(-0.3, 0.7, n_layers * n_qubits * 3)

        def qforge_run():
            wf = Qubit(n_qubits, backend="auto")
            strongly_entangling_ansatz(wf, params, n_layers=n_layers)
            return pauli_expectation(wf, 0, "Z")

        result = bench(
            qforge_run,
            n_warmup=self.config.n_warmup,
            n_runs=self.config.n_runs,
        )
        data = {
            "qforge_ms": result.median * 1000,
            "observable": float(result.result),
            "qubits": n_qubits,
            "n_layers": n_layers,
        }

        qml = get_pennylane()
        if qml is not None:
            data["pennylane_ms"] = self._ansatz_pennylane_ms(qml, params, n_qubits, n_layers)

        if get_qiskit_available():
            try:
                data["qiskit_ms"] = self._ansatz_qiskit_ms(params, n_qubits, n_layers)
            except Exception:
                data["qiskit_ms"] = None

        return data

    def _ansatz_pennylane_ms(self, qml, params, n_qubits, n_layers):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(p):
            idx = 0
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RZ(p[idx], wires=q); idx += 1
                    qml.RY(p[idx], wires=q); idx += 1
                    qml.RZ(p[idx], wires=q); idx += 1
                stride = (layer % (n_qubits - 1)) + 1
                for q in range(n_qubits):
                    target = (q + stride) % n_qubits
                    if target != q:
                        qml.CNOT(wires=[q, target])
            return qml.expval(qml.PauliZ(0))

        return bench(
            lambda: float(circuit(params)),
            n_warmup=self.config.n_warmup,
            n_runs=self.config.n_runs,
        ).median * 1000

    def _ansatz_qiskit_ms(self, params, n_qubits, n_layers):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        qc = QuantumCircuit(n_qubits)
        idx = 0
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rz(float(params[idx]), q); idx += 1
                qc.ry(float(params[idx]), q); idx += 1
                qc.rz(float(params[idx]), q); idx += 1
            stride = (layer % (n_qubits - 1)) + 1
            for q in range(n_qubits):
                target = (q + stride) % n_qubits
                if target != q:
                    qc.cx(q, target)

        op = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])
        estimator = StatevectorEstimator()

        def run():
            return float(estimator.run([(qc, op)]).result()[0].data.evs)

        return bench(
            run,
            n_warmup=self.config.n_warmup,
            n_runs=self.config.n_runs,
        ).median * 1000

    def _benchmark_qsvm_kernel(self, n_qubits):
        from qforge.algo import QSVM

        rng = np.random.default_rng(self.config.seed + n_qubits)
        X = rng.uniform(-0.75, 0.75, size=(4, n_qubits))
        qsvm = QSVM(n_qubits=n_qubits, n_layers=1, backend="auto")

        result = bench(
            lambda: qsvm.kernel_matrix(X),
            n_warmup=self.config.n_warmup,
            n_runs=self.config.n_runs,
        )
        eigs = np.linalg.eigvalsh((result.result + result.result.T) / 2)
        data = {
            "qforge_ms": result.median * 1000,
            "kernel_trace": float(np.trace(result.result)),
            "min_eig": float(eigs.min()),
            "n_samples": len(X),
            "qubits": n_qubits,
        }

        qml = get_pennylane()
        if qml is not None:
            data["pennylane_ms"] = self._qsvm_kernel_pennylane_ms(qml, X, n_qubits)

        return data

    def _qsvm_kernel_pennylane_ms(self, qml, X, n_qubits):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def feature_state(x):
            feats = np.zeros(n_qubits)
            feats[:len(x)] = x[:n_qubits]
            for q in range(n_qubits):
                qml.Hadamard(wires=q)
            for q in range(n_qubits):
                qml.PhaseShift(2.0 * feats[q], wires=q)
            for q in range(n_qubits - 1):
                qml.ControlledPhaseShift(
                    2.0 * (np.pi - feats[q]) * (np.pi - feats[q + 1]),
                    wires=[q, q + 1],
                )
            return qml.state()

        def run():
            states = [np.asarray(feature_state(x)) for x in X]
            return np.array([
                [abs(np.vdot(a, b)) ** 2 for b in states]
                for a in states
            ])

        return bench(
            run,
            n_warmup=self.config.n_warmup,
            n_runs=self.config.n_runs,
        ).median * 1000

    def _benchmark_classifiers(self, n_qubits):
        from qforge.algo import VQC, QCNN, DataReuploadingClassifier, Adam

        rng = np.random.default_rng(self.config.seed + 100 + n_qubits)
        X = rng.uniform(-1.0, 1.0, size=(4, n_qubits))
        y = np.array([0, 1, 1, 0])
        steps = min(self.config.steps, 3)
        qcnn_qubits = 1 << int(np.ceil(np.log2(max(n_qubits, 2))))

        specs = {
            "VQC training": VQC(n_qubits=n_qubits, n_layers=1, n_classes=2, backend="auto"),
            "QCNN training": QCNN(n_qubits=qcnn_qubits, n_classes=2, backend="auto"),
            "Data reuploading training": DataReuploadingClassifier(
                n_qubits=min(n_qubits, 4), n_layers=2, n_classes=2, backend="auto"
            ),
        }
        results = {}
        for name, model in specs.items():
            params0 = np.linspace(-0.2, 0.2, model.n_params)

            def run(model=model, params0=params0):
                _, history = model.fit(
                    X,
                    y,
                    params=params0,
                    optimizer=Adam(lr=0.02),
                    steps=steps,
                )
                return history[-1]

            result = bench(run, n_warmup=0, n_runs=1)
            results[name] = {
                "qforge_ms": result.median * 1000,
                "final_loss": float(result.result),
                "steps": steps,
                "qubits": n_qubits,
                "model": _model_key(name),
            }
        return results


def _fmt_ms(value):
    return f"{value:.3f}" if value is not None else "N/A"


def _model_key(name):
    if name.startswith("VQC"):
        return "vqc"
    if name.startswith("QCNN"):
        return "qcnn"
    return "reuploading"
