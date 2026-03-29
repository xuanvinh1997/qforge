# Qforge.circuit

Circuit initialization functions that create quantum states ready for gate application.

`Qubit` is the primary entry point for creating a qubit circuit. `Qudit` creates
circuits with arbitrary local dimension (qutrits, ququarts, etc.). Both return a
[`Wavefunction`](wavefunction.md) initialized to the |00...0> state, optionally backed
by a C++ `StateVector` for accelerated simulation.

## Usage

```python
from qforge.circuit import Qubit, Qudit, Walk_Qubit

# Standard qubit circuit (auto-selects best backend)
wf = Qubit(4)

# Force pure-Python backend
wf = Qubit(4, backend='python')

# Qutrit circuit (d=3)
wf3 = Qudit(4, dimension=3)

# Ququart circuit (d=4)
wf4 = Qudit(3, dimension=4)

# Quantum walk state (1D lattice)
walk = Walk_Qubit(qubit_num=3, dim=1)
```

## API

### `Qubit(qubit_num, backend='auto')`

Create a qubit (d=2) quantum circuit with `qubit_num` qubits.

**Parameters:**

- `qubit_num` -- Number of qubits.
- `backend` -- `'auto'`, `'cpu'`, `'cuda'`, `'metal'`, or `'python'`.

### `Qudit(n_qudits, dimension=3, backend='auto')`

Create a qudit quantum circuit with arbitrary local Hilbert space dimension.

**Parameters:**

- `n_qudits` -- Number of qudits.
- `dimension` -- Local dimension (d=2 for qubits, d=3 for qutrits, d=4 for ququarts, etc.).
- `backend` -- `'auto'`, `'cpu'`, or `'python'`.

**Note:** The state space grows as d^n, so the maximum number of qudits decreases
as dimension increases. The C++ backend limits total dimension to 2^30.

### `Walk_Qubit(qubit_num=1, dim=1)`

Create an initial quantum state for Hadamard coin quantum walk.

::: qforge.circuit
    options:
      show_source: false
      members:
        - Qubit
        - Qudit
        - Walk_Qubit
