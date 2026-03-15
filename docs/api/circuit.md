# Qforge.circuit

Circuit initialization functions that create quantum states ready for gate application.

`Qubit` is the primary entry point for creating a quantum circuit. It returns a
[`Wavefunction`](wavefunction.md) initialized to the |00...0> state, optionally backed
by a C++ `StateVector` for accelerated simulation.

## Usage

```python
from Qforge.circuit import Qubit, Walk_Qubit

# Standard circuit (auto-selects best backend)
wf = Qubit(4)

# Force pure-Python backend
wf = Qubit(4, backend='python')

# Quantum walk state (1D lattice)
walk = Walk_Qubit(qubit_num=3, dim=1)
```

## API

::: Qforge.circuit
    options:
      show_source: false
      members:
        - Qubit
        - Walk_Qubit
