# Qforge.measurement

Quantum measurement operations including sampling, single-qubit measurement,
projective collapse, and Pauli expectation values. All functions dispatch to
C++ kernels when available.

## Usage

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT
from Qforge.measurement import measure_all, measure_one, collapse_one, pauli_expectation

wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)   # Bell state

# Sample 1000 measurements
states, counts = measure_all(wf, 1000)

# Single-qubit probabilities
probs = measure_one(wf, 0)   # array([0.5, 0.5])

# Projective collapse
collapse_one(wf, 0)  # collapses qubit 0 into |0> or |1>

# Pauli expectation value
exp_val = pauli_expectation(wf, [('Z', 0), ('Z', 1)])
```

## API

::: Qforge.measurement
    options:
      show_source: false
