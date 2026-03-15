# Qforge.wavefunction

The `Wavefunction` class represents a quantum state as a complex amplitude vector over
computational basis states. It transparently wraps either a pure-Python numpy backend or
a C++ `StateVector` for 100-200x speedup.

## Usage

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT

wf = Qubit(2)          # |00>
H(wf, 0)               # (|00> + |10>) / sqrt(2)
CNOT(wf, 0, 1)         # Bell state (|00> + |11>) / sqrt(2)

# Inspect the state
print(wf.amplitude)       # complex amplitude vector
print(wf.probabilities()) # array([0.5, 0.0, 0.0, 0.5])
print(wf.visual)          # list of gate operations applied
```

## API

::: qforge.wavefunction
    options:
      show_source: false
      members:
        - Wavefunction
