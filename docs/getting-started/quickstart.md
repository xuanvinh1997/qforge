# Quickstart: Your First Bell State

This 5-minute tutorial walks you through creating a Bell state, inspecting the quantum state, measuring it, and visualizing the circuit.

## Step 1: Create a Quantum Circuit

```python
from Qforge.circuit import Qubit

# Create a 2-qubit system initialized to |00>
wf = Qubit(2)

print(wf.print_state())
# (1+0j)|00> + 0j|01> + 0j|10> + 0j|11>
```

`Qubit(n)` creates an n-qubit wavefunction in the all-zeros state `|00...0>`. The returned object is a `Wavefunction` that holds a complex amplitude vector of length 2^n.

## Step 2: Apply Gates

```python
from Qforge.gates import H, CNOT

# Hadamard on qubit 0: |00> -> (|00> + |10>) / sqrt(2)
H(wf, 0)

# CNOT with control=0, target=1: -> (|00> + |11>) / sqrt(2)
CNOT(wf, 0, 1)
```

Gates are functions that modify the wavefunction in place. The first argument is always the wavefunction, followed by qubit indices and optional parameters.

## Step 3: Inspect the State

```python
# Bra-ket representation
print(wf.print_state())
# (0.7071067811865476+0j)|00> + 0j|01> + 0j|10> + (0.7071067811865476+0j)|11>

# Probability distribution
print(wf.probabilities())
# [0.5 0.  0.  0.5]

# Direct amplitude access
print(wf.amplitude)
# [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
```

The probabilities confirm an equal 50/50 split between `|00>` and `|11>` -- this is the Bell state (|00> + |11>) / sqrt(2).

## Step 4: Measure

```python
from Qforge.measurement import measure_all, measure_one

# Sample 1000 measurements
states, counts = measure_all(wf, 1000)
for state, count in zip(states, counts):
    print(f"|{state}>: {count} times")
# |00>: ~500 times
# |11>: ~500 times

# Check single-qubit probabilities
print(measure_one(wf, 0))  # [0.5, 0.5]
print(measure_one(wf, 1))  # [0.5, 0.5]
```

> **Note:** `measure_all` samples from the probability distribution without collapsing the state. Use `collapse_one` for projective measurement that modifies the wavefunction.

## Step 5: Visualize the Circuit

```python
wf.visual_circuit()
```

Output:

```
|Q_0> : H--o---M
            |
|Q_1> : ---x---M
```

The diagram shows the Hadamard gate on qubit 0, followed by a CNOT (control `o` on qubit 0, target `x` on qubit 1).

## Putting It All Together

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT
from Qforge.measurement import measure_all

# Build
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)

# Analyze
print("State:", wf.print_state())
print("Probabilities:", wf.probabilities())

# Measure
states, counts = measure_all(wf, 1000)
print("Samples:", dict(zip(states, counts)))

# Visualize
wf.visual_circuit()
```

## Next Steps

- [Core Concepts](concepts.md) -- understand the wavefunction model, qubit indexing, and backends
- [Gates and Rotations](../tutorials/02-gates-and-rotations.md) -- explore the full gate catalog
- [Circuit IR](../tutorials/05-circuit-ir.md) -- build reusable, composable circuits
