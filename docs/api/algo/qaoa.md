# Qforge.algo.qaoa

Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization,
with built-in support for the Max-Cut problem.

## Usage

```python
from qforge.algo import QAOA
import numpy as np

# 4-cycle graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
qaoa = QAOA(n_qubits=4, edges=edges, p_layers=2)

params0 = np.random.uniform(0, np.pi, qaoa.n_params)
params, history = qaoa.optimize(params0, steps=120)
sol = qaoa.get_solution(params)
print("Best bitstring:", sol['bitstring'], "| Cut:", sol['cut_value'])
```

## Class

### `QAOA`

Inherits from `VQA` (general variational framework).

```python
QAOA(
    n_qubits: int,
    edges: list[tuple[int, int]],
    p_layers: int = 1,
    backend: str = 'auto',
)
```

**Parameters:**

- `n_qubits` -- Number of qubits (= number of graph nodes).
- `edges` -- List of `(i, j)` edge tuples (0-indexed).
- `p_layers` -- Number of QAOA layers.
- `backend` -- Qforge backend.

**Parameter layout:**

```
params[:p_layers]  = gamma  (problem Hamiltonian angles)
params[p_layers:]  = beta   (mixer Hamiltonian angles)
```

**Key methods:**

- `optimize(params0, steps=100, optimizer=None)` -- Run optimization. Returns `(params, history)`.
- `cost(params)` -- Evaluate the Max-Cut cost.
- `get_solution(params)` -- Return dict with `'bitstring'` and `'cut_value'`.
- `n_params` -- Total number of parameters (property): `2 * p_layers`.

## Full API

::: qforge.algo.qaoa
    options:
      show_source: false
