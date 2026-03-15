# How-To: Define Custom Gates

Qforge provides several ways to extend the gate library with custom unitary
operations.

## QubitUnitary: Apply an Arbitrary Matrix

Apply any unitary matrix as a gate:

```python
from Qforge.circuit import Qubit
from Qforge.gates import QubitUnitary
import numpy as np

# Define a custom single-qubit gate (must be 2x2 unitary)
sqrt_x = (1 / 2) * np.array([
    [1 + 1j, 1 - 1j],
    [1 - 1j, 1 + 1j],
])

qc = Qubit(n_qubits=2)
QubitUnitary(qc, target=0, matrix=sqrt_x)
```

For two-qubit gates, pass a 4x4 matrix:

```python
# Custom two-qubit gate (must be 4x4 unitary)
# Example: sqrt(SWAP)
sqrt_swap = np.array([
    [1, 0, 0, 0],
    [0, 0.5*(1+1j), 0.5*(1-1j), 0],
    [0, 0.5*(1-1j), 0.5*(1+1j), 0],
    [0, 0, 0, 1],
])

QubitUnitary(qc, target=[0, 1], matrix=sqrt_swap)
```

> **Warning:** `QubitUnitary` does not verify unitarity by default for
> performance. Pass `validate=True` to enable the check:
> `QubitUnitary(qc, target=0, matrix=my_matrix, validate=True)`.

## CustomGate: Reusable Named Gates

For gates you use repeatedly, register them with a name:

```python
from Qforge.ir import CustomGate, register_gate
import numpy as np

# Define the gate
sqrt_x_gate = CustomGate(
    name='SqrtX',
    matrix=np.array([
        [0.5*(1+1j), 0.5*(1-1j)],
        [0.5*(1-1j), 0.5*(1+1j)],
    ]),
    n_qubits=1,
    label='sqrt(X)',
)

# Register globally
register_gate(sqrt_x_gate)
```

Once registered, use the gate by name:

```python
from Qforge.circuit import Qubit
from Qforge.ir import CustomGate

qc = Qubit(n_qubits=2)
CustomGate.apply(qc, 'SqrtX', target=0)
CustomGate.apply(qc, 'SqrtX', target=1)
```

## Parameterized Custom Gates

Define gates that accept rotation angles:

```python
from Qforge.ir import CustomGate, register_gate
import numpy as np

def rxx_matrix(theta):
    """RXX gate: exp(-i * theta/2 * X x X)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c,    0,    0,    -1j*s],
        [0,    c,    -1j*s, 0],
        [0,    -1j*s, c,    0],
        [-1j*s, 0,    0,    c],
    ])

rxx_gate = CustomGate(
    name='RXX',
    matrix_fn=rxx_matrix,   # Function instead of fixed matrix
    n_qubits=2,
    n_params=1,
    label='RXX(theta)',
)

register_gate(rxx_gate)

# Use with a parameter
qc = Qubit(n_qubits=2)
CustomGate.apply(qc, 'RXX', target=[0, 1], params=[np.pi / 4])
```

## Multi-Controlled Gates

Qforge provides multi-controlled versions of standard gates:

```python
from Qforge.circuit import Qubit
from Qforge.gates import mcx, mcz, mcp
import numpy as np

qc = Qubit(n_qubits=5)

# Multi-controlled X (Toffoli generalization)
mcx(qc, controls=[0, 1, 2], target=3)

# Multi-controlled Z
mcz(qc, controls=[0, 1, 2], target=3)

# Multi-controlled Phase
mcp(qc, controls=[0, 1], target=2, theta=np.pi / 4)
```

> **Note:** Multi-controlled gates are decomposed into a sequence of
> single-qubit and CNOT gates internally. The decomposition adds ancilla
> operations but preserves the overall unitary.

## Gate Composition

Compose multiple gates into a single reusable unit:

```python
from Qforge.ir import CustomGate, register_gate
from Qforge.gates import H, CNOT, RZ
import numpy as np

def bell_basis_matrix():
    """Unitary that maps computational basis to Bell basis."""
    return (1 / np.sqrt(2)) * np.array([
        [1, 0, 0,  1],
        [0, 1, 1,  0],
        [0, 1, -1, 0],
        [1, 0, 0, -1],
    ])

bell_gate = CustomGate(
    name='BellBasis',
    matrix=bell_basis_matrix(),
    n_qubits=2,
    label='Bell',
)

register_gate(bell_gate)

# Use in a circuit
qc = Qubit(n_qubits=4)
CustomGate.apply(qc, 'BellBasis', target=[0, 1])
CustomGate.apply(qc, 'BellBasis', target=[2, 3])
```

## Custom Gate with C++ Acceleration

Registered custom gates automatically use the C++ engine when available.
The matrix is passed to the C++ `apply_single_qubit_gate` or
`apply_controlled_gate` kernel:

```python
# Check if C++ acceleration is active
import Qforge
print(f"C++ engine: {Qforge._HAS_CPP}")

# Custom gates use C++ when available -- no extra setup needed
qc = Qubit(n_qubits=10)
CustomGate.apply(qc, 'SqrtX', target=5)  # C++ accelerated
```

## Listing Registered Gates

```python
from Qforge.ir import list_registered_gates

for gate in list_registered_gates():
    print(f"{gate.name}: {gate.n_qubits}-qubit, "
          f"params={gate.n_params}, label='{gate.label}'")
```

## Summary

| Method | Use Case |
|--------|----------|
| `QubitUnitary(qc, target, matrix)` | One-off arbitrary unitary |
| `CustomGate` + `register_gate` | Reusable named gate |
| `CustomGate` with `matrix_fn` | Parameterized gate |
| `mcx`, `mcz`, `mcp` | Multi-controlled gates |
