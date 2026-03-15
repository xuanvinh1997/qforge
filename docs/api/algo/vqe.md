# Qforge.algo.vqe

Variational Quantum Eigensolver (VQE) for finding ground-state energies of
quantum Hamiltonians using a parameterized ansatz.

## Usage

```python
from Qforge.algo import Hamiltonian, VQE
import numpy as np

# Toy H2 Hamiltonian (2 qubits)
H = Hamiltonian(
    coeffs=[-1.0523, 0.3979, -0.3979, -0.0112, 0.1809],
    terms=[
        [],
        [('Z', 0)],
        [('Z', 1)],
        [('Z', 0), ('Z', 1)],
        [('X', 0), ('X', 1)],
    ],
)

n_p = VQE.n_params_hardware_efficient(n_qubits=2, n_layers=2)
vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=2)
params, history = vqe.optimize(np.zeros(n_p), steps=150)
print("Ground-state energy:", history[-1])
```

## Class

### `VQE`

Inherits from `VQA` (general variational framework).

```python
VQE(
    n_qubits: int,
    hamiltonian: Hamiltonian,
    circuit_fn=None,
    n_layers: int = 1,
    backend: str = 'auto',
)
```

**Parameters:**

- `n_qubits` -- Number of qubits.
- `hamiltonian` -- `Hamiltonian` object defining the cost function.
- `circuit_fn` -- Custom ansatz `callable(wf, params) -> None`. Defaults to `hardware_efficient_ansatz`.
- `n_layers` -- Layers for the default ansatz (ignored if `circuit_fn` is provided).
- `backend` -- Qforge backend (`'auto'`, `'cpu'`, `'cuda'`, `'metal'`).

**Key methods (inherited from VQA):**

- `optimize(params0, steps=100, optimizer=None)` -- Run optimization. Returns `(params, history)`.
- `cost(params)` -- Evaluate the cost function at given parameters.
- `n_params_hardware_efficient(n_qubits, n_layers)` -- Class method returning parameter count.

## Full API

::: Qforge.algo.vqe
    options:
      show_source: false
