# Tutorial 1: Your First Circuit

This tutorial walks through building a simple quantum circuit using both the functional API and the Circuit IR.

## Goal

Create a 3-qubit GHZ state: (|000> + |111>) / sqrt(2).

## Functional API

### Step 1: Initialize

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT

wf = Qubit(3)
```

This creates a 3-qubit wavefunction initialized to |000>. The amplitude vector has 2^3 = 8 entries.

### Step 2: Apply Gates

The GHZ state is built by applying a Hadamard to the first qubit, then cascading CNOT gates.

```python
H(wf, 0)        # |000> -> (|000> + |100>) / sqrt(2)
CNOT(wf, 0, 1)  # -> (|000> + |110>) / sqrt(2)
CNOT(wf, 1, 2)  # -> (|000> + |111>) / sqrt(2)
```

### Step 3: Verify

```python
import numpy as np

probs = wf.probabilities()
print(probs)
# [0.5 0.  0.  0.  0.  0.  0.  0.5]

# Confirm only |000> (index 0) and |111> (index 7) have amplitude
assert np.isclose(probs[0], 0.5)
assert np.isclose(probs[7], 0.5)
assert np.isclose(probs.sum(), 1.0)
```

### Step 4: Measure and Visualize

```python
from Qforge.measurement import measure_all

states, counts = measure_all(wf, 1000)
print(dict(zip(states, counts)))
# {'000': ~500, '111': ~500}

wf.visual_circuit()
```

## Circuit IR API

The same circuit built with the IR. Instead of calling gate functions directly, you record operations on a `Circuit` object.

### Step 1: Build the Circuit

```python
from Qforge.ir import Circuit

qc = Circuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)
```

Methods return `self`, so you can chain them:

```python
qc = Circuit(3)
qc.h(0).cnot(0, 1).cnot(1, 2)
```

### Step 2: Inspect

```python
print(qc)
# Circuit(n_qubits=3, depth=3)

for op in qc:
    print(op)
# GateOp(name='H', qubits=(0,), params=(), ...)
# GateOp(name='CNOT', qubits=(0, 1), params=(), ...)
# GateOp(name='CNOT', qubits=(1, 2), params=(), ...)
```

### Step 3: Execute

```python
wf = qc.run()
print(wf.probabilities())
# [0.5 0.  0.  0.  0.  0.  0.  0.5]
```

`run()` creates a fresh `Wavefunction`, applies all operations, and returns the final state. You can pass `backend='cpu'` or any other backend string.

### Step 4: Apply to an Existing Wavefunction

If you already have a wavefunction, use the circuit as a callable:

```python
from Qforge.circuit import Qubit

wf2 = Qubit(3)
qc(wf2)  # Apply circuit in-place
print(wf2.probabilities())
# [0.5 0.  0.  0.  0.  0.  0.  0.5]
```

## Recording Functional Calls into a Circuit

You can capture functional API calls into a Circuit using the `record` context manager:

```python
from Qforge.ir import record
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT

with record(3) as qc:
    wf = Qubit(3)
    H(wf, 0)
    CNOT(wf, 0, 1)
    CNOT(wf, 1, 2)

# qc is now a Circuit with the same 3 operations
print(qc)
# Circuit(n_qubits=3, depth=3)

# Replay it
wf2 = qc.run()
```

This is useful when you want to use the functional API's syntax but still need the IR's transformation capabilities.

## Full Example

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT
from Qforge.measurement import measure_all
from Qforge.ir import Circuit

# --- Functional API ---
wf = Qubit(3)
H(wf, 0)
CNOT(wf, 0, 1)
CNOT(wf, 1, 2)
print("Functional:", wf.probabilities())
wf.visual_circuit()

# --- Circuit IR ---
qc = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
wf2 = qc.run()
print("Circuit IR:", wf2.probabilities())

# --- Measure both ---
for label, w in [("Functional", wf), ("Circuit IR", wf2)]:
    states, counts = measure_all(w, 1000)
    print(f"{label}: {dict(zip(states, counts))}")
```
