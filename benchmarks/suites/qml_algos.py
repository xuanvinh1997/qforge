# -*- coding: utf-8 -*-
"""QML algorithm benchmark suite."""
from __future__ import annotations

import time
import numpy as np
from qforge.circuit import Qubit
from qforge import gates as G
from qforge.algo import VQC, QCNN, DataReuploadingClassifier, QuantumReservoir, QSVM


class QMLAlgosSuite:
    name = "qml"
    description = "QML classifiers: accuracy, training time"

    def __init__(self, config):
        self.config = config

    def run(self, backends=None):
        qubit_range = self.config.get("qml_qubits", [2, 4, 8])
        steps = self.config.get("steps", 15)
        n_train, n_test = 40, 20

        results = {"accuracy": {}, "training_time": {}}

        for nq in qubit_range:
            X, y = self._make_data(nq, n_train + n_test)
            X_tr, y_tr = X[:n_train], y[:n_train]
            X_te, y_te = X[n_train:], y[n_train:]

            acc_entry = {}
            time_entry = {}

            # VQC
            vqc = VQC(n_qubits=nq, n_layers=2, n_classes=2)
            t0 = time.perf_counter()
            vqc.fit(X_tr, y_tr, steps=steps)
            time_entry["vqc"] = time.perf_counter() - t0
            acc_entry["vqc"] = float(vqc.score(X_te, y_te))

            # QCNN
            nq_cnn = 2 ** int(np.ceil(np.log2(max(nq, 2))))
            qcnn = QCNN(n_qubits=nq_cnn, n_classes=2)
            t0 = time.perf_counter()
            qcnn.fit(X_tr, y_tr, steps=steps)
            time_entry["qcnn"] = time.perf_counter() - t0
            acc_entry["qcnn"] = float(qcnn.score(X_te, y_te))

            # DataReuploading
            drc = DataReuploadingClassifier(n_qubits=max(nq, 1), n_layers=3, n_classes=2)
            t0 = time.perf_counter()
            drc.fit(X_tr, y_tr, steps=steps)
            time_entry["reuploading"] = time.perf_counter() - t0
            acc_entry["reuploading"] = float(drc.score(X_te, y_te))

            # Reservoir
            qr = QuantumReservoir(n_qubits=nq, n_layers=2)
            t0 = time.perf_counter()
            qr.fit(X_tr, y_tr)
            time_entry["reservoir"] = time.perf_counter() - t0
            acc_entry["reservoir"] = float(qr.score(X_te, y_te))

            # QSVM
            qsvm = QSVM(n_qubits=min(nq, 4), n_layers=1, C=1.0)
            t0 = time.perf_counter()
            qsvm.fit(X_tr, y_tr)
            time_entry["qsvm"] = time.perf_counter() - t0
            acc_entry["qsvm"] = float(qsvm.score(X_te, y_te))

            results["accuracy"][str(nq)] = acc_entry
            results["training_time"][str(nq)] = time_entry

        return results

    @staticmethod
    def _make_data(n_features, n_samples, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-np.pi, np.pi, (n_samples, n_features))
        y = (np.sin(X[:, 0]) * np.cos(X[:, 1 % n_features]) > 0).astype(int)
        return X, y
