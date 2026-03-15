# Qforge.kernels

Quantum kernel methods for computing similarity between quantum states. Includes
analytic inner products, the SWAP test, Hadamard test, and a projected quantum
kernel suitable for use with classical SVMs and kernel ridge regression.

## Usage

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, RY
from Qforge.kernels import state_product, swap_test, ProjectedQuantumKernel
import numpy as np

wf1 = Qubit(2); H(wf1, 0); RY(wf1, 1, 0.5)
wf2 = Qubit(2); H(wf2, 0); RY(wf2, 1, 1.0)

# Analytic overlap
k = state_product(wf1, wf2)

# SWAP test (circuit-based)
k_swap = swap_test(wf1, wf2)

# Projected quantum kernel for ML
pqk = ProjectedQuantumKernel(
    circuit_fn=my_ansatz, n_qubits=4, n_layers=3, gamma=1.0
)
K = pqk.kernel_matrix(X_train)
```

## Functions

### `state_product(wavefunction_1, wavefunction_2)`

Compute the fidelity `|<psi_1|psi_2>|` as the absolute value of the inner product
of the two amplitude vectors.

**Returns:** `float` in [0, 1].

---

### `swap_test(wavefunction_1, wavefunction_2)`

Compute the state overlap using the quantum SWAP test circuit. Requires both
states to have the same number of qubits. Constructs an ancilla-based circuit
with CSWAP gates.

**Returns:** `float` -- `sqrt(2 * P(|0>) - 1)`.

---

### `hadamard_test(wavefunction_1, wavefunction_2)`

Compute the real part of the inner product using the Hadamard test circuit.

**Returns:** `float` -- `Re(<psi_1|psi_2>)`.

## Classes

### `ProjectedQuantumKernel`

Projected quantum kernel (Huang et al., 2021) for quantum-enhanced machine learning.

```python
ProjectedQuantumKernel(
    circuit_fn,
    n_qubits=4,
    n_layers=3,
    params=None,
    gamma=1.0,
    random_state=42,
)
```

**Parameters:**

- `circuit_fn` -- Quantum circuit function `(wf, x, params) -> None`.
- `n_qubits` -- Number of qubits.
- `n_layers` -- Number of circuit layers.
- `params` -- Optional fixed circuit parameters.
- `gamma` -- RBF kernel bandwidth.
- `random_state` -- Random seed.

**Methods:**

- `kernel_matrix(X)` -- Compute the kernel Gram matrix for dataset X.
- `kernel_entry(x1, x2)` -- Compute a single kernel entry.

## Full API

::: Qforge.kernels
    options:
      show_source: false
