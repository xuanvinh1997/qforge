# Qforge.algo

Algorithm sub-package for variational quantum algorithms, gradient computation,
classical optimizers, parameterized ansatze, and standard quantum algorithms.

## Quick Start

```python
from qforge.algo import Hamiltonian, VQE, QAOA, Adam
import numpy as np

# VQE for a toy Hamiltonian
H = Hamiltonian([-1.0, 0.5], [[('Z', 0)], [('X', 0), ('X', 1)]])
vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=1)
params, history = vqe.optimize(np.zeros(4), steps=100)

# QAOA for Max-Cut
qaoa = QAOA(n_qubits=4, edges=[(0,1),(1,2),(2,3),(3,0)], p_layers=2)
p0 = np.random.uniform(0, np.pi, qaoa.n_params)
params, history = qaoa.optimize(p0, steps=100)
sol = qaoa.get_solution(params)
```

## Sub-modules

| Module | Description |
|--------|-------------|
| [`hamiltonian`](hamiltonian.md) | Hamiltonian representation and Pauli-string expectations |
| [`gradient`](gradient.md) | Parameter-shift gradient rule |
| [`optimizers`](optimizers.md) | GradientDescent, Adam, SPSA, LBFGS |
| [`ansatz`](ansatz.md) | Hardware-efficient and strongly-entangling ansatze |
| [`vqe`](vqe.md) | Variational Quantum Eigensolver |
| [`qaoa`](qaoa.md) | Quantum Approximate Optimization Algorithm |
| [`standard`](standard.md) | QFT, QPE, Grover, HHL |

## Exported Names

All key symbols are re-exported from `qforge.algo`:

```python
from qforge.algo import (
    Hamiltonian,
    parameter_shift, parallel_parameter_shift,
    GradientDescent, Adam, SPSA, LBFGS,
    hardware_efficient_ansatz, strongly_entangling_ansatz,
    VQA, VQE, QAOA,
)
```
