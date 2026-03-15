# Qforge.algo.ansatz

Pre-built parameterized ansatz circuits for variational quantum algorithms.

## Usage

```python
from qforge.circuit import Qubit
from qforge.algo.ansatz import hardware_efficient_ansatz, strongly_entangling_ansatz
import numpy as np

wf = Qubit(4)
n_params = 4 * (2 + 1)  # n_qubits * (n_layers + 1)
params = np.random.randn(n_params)
hardware_efficient_ansatz(wf, params, n_layers=2)
```

## Functions

### `hardware_efficient_ansatz(wf, params, n_layers=1)`

Hardware-efficient ansatz with alternating RY rotation blocks and nearest-neighbour
CNOT ladders.

**Structure:** `(n_layers + 1)` RY blocks separated by `n_layers` CNOT ladders.

**Number of parameters:** `n_qubits * (n_layers + 1)`

**Parameters:**

- `wf` -- Wavefunction (modified in-place).
- `params` -- 1-D rotation-angle array.
- `n_layers` -- Number of entangling layers (default 1).

---

### `strongly_entangling_ansatz(wf, params, n_layers=1)`

Strongly entangling ansatz with RZ/RY/RZ rotations and long-range CNOT rings
with stride that varies by layer.

**Number of parameters:** `n_layers * n_qubits * 3`

**Parameters:**

- `wf` -- Wavefunction (modified in-place).
- `params` -- 1-D rotation-angle array.
- `n_layers` -- Number of entangling layers (default 1).

## Full API

::: qforge.algo.ansatz
    options:
      show_source: false
