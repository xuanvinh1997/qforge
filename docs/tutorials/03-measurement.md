# Tutorial 3: Measurement

This tutorial covers all measurement operations in Qforge: sampling, single-qubit probabilities, projective collapse, and expectation values.

## Setup

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT, X, RY
from qforge.measurement import measure_all, measure_one, collapse_one, pauli_expectation
import numpy as np
```

## measure_all -- Sample from the Probability Distribution

`measure_all(wf, n_samples)` draws `n_samples` independent measurements from the state's probability distribution. It returns two arrays: the unique measured states and their counts.

```python
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)

states, counts = measure_all(wf, 10000)
for state, count in zip(states, counts):
    print(f"|{state}>: {count}")
# |00>: ~5000
# |11>: ~5000
```

> **Note:** `measure_all` does **not** collapse the wavefunction. It samples from `|amplitude|^2` without modifying the state. You can call it multiple times and the wavefunction remains unchanged.

### Accessing Results as a Dictionary

```python
results = dict(zip(states, counts))
print(results)
# {'00': 5012, '11': 4988}
```

### Computing Relative Frequencies

```python
n_samples = 10000
states, counts = measure_all(wf, n_samples)
frequencies = {s: c / n_samples for s, c in zip(states, counts)}
print(frequencies)
# {'00': 0.5012, '11': 0.4988}
```

## measure_one -- Single-Qubit Probabilities

`measure_one(wf, qubit)` returns `[P(|0>), P(|1>)]` for the specified qubit. This is computed analytically (no sampling noise), and the state is **not** modified.

```python
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)

print(measure_one(wf, 0))  # [0.5, 0.5]
print(measure_one(wf, 1))  # [0.5, 0.5]
```

### Example: Non-Uniform Probabilities

```python
wf = Qubit(1)
RY(wf, 0, phi=np.pi/3)   # Creates an asymmetric superposition

probs = measure_one(wf, 0)
print(f"P(|0>) = {probs[0]:.4f}")  # 0.75
print(f"P(|1>) = {probs[1]:.4f}")  # 0.25
```

## collapse_one -- Projective Measurement

`collapse_one(wf, qubit)` performs a projective measurement on a single qubit. It:

1. Computes `P(|0>)` and `P(|1>)` for the qubit
2. Randomly collapses to `|0>` or `|1>` according to those probabilities
3. Renormalizes the remaining amplitudes

The wavefunction is **modified in place**.

```python
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)

print("Before collapse:", wf.probabilities())
# [0.5 0.  0.  0.5]

collapse_one(wf, 0)

print("After collapse:", wf.probabilities())
# Either [1. 0. 0. 0.] (collapsed to |00>)
# or     [0. 0. 0. 1.] (collapsed to |11>)
```

> **Warning:** `collapse_one` is non-deterministic and destructive. The wavefunction is irreversibly modified. If you need the original state, make a copy first.

### Sequential Collapse

```python
wf = Qubit(3)
H(wf, 0)
CNOT(wf, 0, 1)
CNOT(wf, 1, 2)  # GHZ state

# Collapse qubit 0 -- this determines the entire GHZ state
collapse_one(wf, 0)
print(measure_one(wf, 1))  # [1., 0.] or [0., 1.] -- now deterministic
print(measure_one(wf, 2))  # Same as qubit 1
```

## pauli_expectation -- Expectation Values

`pauli_expectation(wf, qubit, pauli_type)` computes the expectation value `<psi|O|psi>` where `O` is a Pauli operator (`'X'`, `'Y'`, or `'Z'`) on the specified qubit. Returns a float in `[-1, +1]`.

### Pauli-Z Expectation

`<Z>` equals `P(|0>) - P(|1>)`.

```python
# |0> state: <Z> = +1
wf = Qubit(1)
print(pauli_expectation(wf, 0, 'Z'))  # 1.0

# |1> state: <Z> = -1
wf = Qubit(1)
X(wf, 0)
print(pauli_expectation(wf, 0, 'Z'))  # -1.0

# Equal superposition: <Z> = 0
wf = Qubit(1)
H(wf, 0)
print(pauli_expectation(wf, 0, 'Z'))  # 0.0
```

### Pauli-X Expectation

`<X>` measures coherence in the X basis.

```python
# |+> state = (|0>+|1>)/sqrt(2): <X> = +1
wf = Qubit(1)
H(wf, 0)
print(pauli_expectation(wf, 0, 'X'))  # 1.0

# |-> state = (|0>-|1>)/sqrt(2): <X> = -1
wf = Qubit(1)
X(wf, 0)
H(wf, 0)
print(pauli_expectation(wf, 0, 'X'))  # -1.0

# |0> state: <X> = 0
wf = Qubit(1)
print(pauli_expectation(wf, 0, 'X'))  # 0.0
```

### Pauli-Y Expectation

```python
# |0> state: <Y> = 0
wf = Qubit(1)
print(pauli_expectation(wf, 0, 'Y'))  # 0.0

# Eigenstate of Y:  S H |0> = (|0> + i|1>)/sqrt(2)
wf = Qubit(1)
H(wf, 0)
S(wf, 0)
print(round(pauli_expectation(wf, 0, 'Y'), 4))  # 1.0
```

### Expectation Values in Multi-Qubit Systems

```python
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)  # Bell state

# Each qubit individually is maximally mixed
print(pauli_expectation(wf, 0, 'Z'))  # 0.0
print(pauli_expectation(wf, 1, 'Z'))  # 0.0
print(pauli_expectation(wf, 0, 'X'))  # 0.0
print(pauli_expectation(wf, 1, 'X'))  # 0.0
```

> **Tip:** For multi-body correlators like `<Z_0 Z_1>`, use `PauliZExpectation` from `qforge.data` (covered in the [Entanglement tutorial](04-entanglement.md)).

## Comparing `probabilities()` with Measurement Functions

| Method | Modifies state? | Output | Noise? |
|--------|----------------|--------|--------|
| `wf.probabilities()` | No | Full probability array (all 2^n states) | Exact |
| `measure_one(wf, q)` | No | `[P(0), P(1)]` for one qubit | Exact |
| `measure_all(wf, N)` | No | Sampled states + counts | Statistical |
| `collapse_one(wf, q)` | **Yes** | None (state collapses) | Stochastic |
| `pauli_expectation(wf, q, op)` | No | Single float in [-1, 1] | Exact |

## Full Example: Variational State Analysis

```python
from qforge.circuit import Qubit
from qforge.gates import RY, CNOT
from qforge.measurement import measure_all, measure_one, pauli_expectation
import numpy as np

# Parameterized 2-qubit state
wf = Qubit(2)
RY(wf, 0, phi=np.pi/3)
RY(wf, 1, phi=np.pi/6)
CNOT(wf, 0, 1)

# Full probability distribution
print("Probabilities:", np.round(wf.probabilities(), 4))

# Per-qubit marginals
for q in range(2):
    p = measure_one(wf, q)
    print(f"Qubit {q}: P(0)={p[0]:.4f}, P(1)={p[1]:.4f}")

# Expectation values
for q in range(2):
    for op in ['X', 'Y', 'Z']:
        val = pauli_expectation(wf, q, op)
        print(f"<{op}_{q}> = {val:.4f}")

# Sample
states, counts = measure_all(wf, 10000)
print("Samples:", dict(zip(states, counts)))
```
