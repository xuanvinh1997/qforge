# Qforge.mps

Matrix Product State (MPS) simulation backend. MPS represents quantum states as
a chain of rank-3 tensors, enabling efficient simulation of low-entanglement
states with tunable bond dimension. Supports the same gate interface as the
standard `Wavefunction` backend.

## Usage

```python
from qforge.mps import MatrixProductState
from qforge.gates import H, CNOT

mps = MatrixProductState(n_qubits=8, chi_max=32)
H(mps, 0)
CNOT(mps, 0, 1)

# Measure
from qforge.measurement import measure_one
probs = measure_one(mps, 0)  # [P(|0>), P(|1>)]
```

## Class

### `MatrixProductState`

```python
MatrixProductState(n_qubits: int, chi_max: int = 64)
```

**Parameters:**

- `n_qubits` -- Number of qubits.
- `chi_max` -- Maximum bond dimension. Controls the accuracy/memory trade-off.

**Key attributes:**

- `amplitude` -- Full amplitude vector (computed via contraction; expensive for large n).
- `state` -- Basis state labels.
- `_tensors` -- List of MPS site tensors.

**Key methods:**

- `probabilities()` -- Compute probability vector.
- `inner(other)` -- Compute inner product with another MPS.
- `entropy(bond)` -- Entanglement entropy across bond `bond`.

## Full API

::: qforge.mps
    options:
      show_source: false
      members:
        - MatrixProductState
