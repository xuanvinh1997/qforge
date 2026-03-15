# -*- coding: utf-8 -*-
# author: vinhpx
"""qforge.algo — QML algorithm support.

Modules
-------
hamiltonian  Hamiltonian and Pauli-string expectations
gradient     parameter_shift gradient rule
optimizers   GradientDescent, Adam, SPSA, LBFGS
ansatz       hardware_efficient_ansatz, strongly_entangling_ansatz
vqa          VQA (general variational framework)
vqe          VQE (Variational Quantum Eigensolver)
qaoa         QAOA (Quantum Approximate Optimization Algorithm)
qsvm         QSVM (Quantum Support Vector Machine)
vqc          VQC (Variational Quantum Classifier)
qcnn         QCNN (Quantum Convolutional Neural Network)
data_reuploading  DataReuploadingClassifier (data re-uploading)
reservoir    QuantumReservoir (quantum reservoir computing)
qgan         QGAN (Quantum Generative Adversarial Network)

Quick start::

    from qforge.algo import Hamiltonian, VQE, QAOA, Adam
    import numpy as np

    # VQE
    H = Hamiltonian([-1.0, 0.5], [[('Z', 0)], [('X', 0), ('X', 1)]])
    vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=1)
    params, history = vqe.optimize(np.zeros(4), steps=100)

    # QAOA (Max-Cut)
    qaoa = QAOA(n_qubits=4, edges=[(0,1),(1,2),(2,3),(3,0)], p_layers=2)
    p0 = np.random.uniform(0, np.pi, qaoa.n_params)
    params, history = qaoa.optimize(p0, steps=100)
    sol = qaoa.get_solution(params)

    # QSVM
    qsvm = QSVM(n_qubits=2, n_layers=2)
    qsvm.fit(X_train, y_train)
    preds = qsvm.predict(X_test)

    # VQC
    vqc = VQC(n_qubits=4, n_layers=3, n_classes=2)
    params, history = vqc.fit(X_train, y_train, steps=80)

    # Quantum Reservoir
    qr = QuantumReservoir(n_qubits=4, n_layers=3)
    qr.fit(X_train, y_train)
    preds = qr.predict(X_test)
"""

from qforge.algo.hamiltonian import Hamiltonian
from qforge.algo.gradient import parameter_shift, parallel_parameter_shift
from qforge.algo.optimizers import GradientDescent, Adam, SPSA, LBFGS
from qforge.algo.ansatz import hardware_efficient_ansatz, strongly_entangling_ansatz
from qforge.algo.vqa import VQA
from qforge.algo.vqe import VQE
from qforge.algo.qaoa import QAOA
from qforge.algo.qsvm import QSVM
from qforge.algo.vqc import VQC
from qforge.algo.qcnn import QCNN
from qforge.algo.data_reuploading import DataReuploadingClassifier
from qforge.algo.reservoir import QuantumReservoir
from qforge.algo.qgan import QGAN
from qforge.mps import MatrixProductState
from qforge.dmrg import DMRG

__all__ = [
    "Hamiltonian",
    "parameter_shift",
    "parallel_parameter_shift",
    "GradientDescent",
    "Adam",
    "SPSA",
    "LBFGS",
    "hardware_efficient_ansatz",
    "strongly_entangling_ansatz",
    "VQA",
    "VQE",
    "QAOA",
    "QSVM",
    "VQC",
    "QCNN",
    "DataReuploadingClassifier",
    "QuantumReservoir",
    "QGAN",
    "MatrixProductState",
    "DMRG",
]
