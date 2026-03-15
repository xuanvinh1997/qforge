# Tutorial 5: Circuit IR

The Circuit Intermediate Representation (IR) lets you build quantum programs as data structures that can be inspected, transformed, composed, parameterized, and executed on any backend.

## Setup

```python
from qforge.ir import Circuit, GateOp, record, CustomGate, register_gate
from qforge.circuit import Qubit
import numpy as np
```

## Building a Circuit

Create a circuit by specifying the number of qubits and chaining gate methods.

```python
qc = Circuit(3)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)

print(qc)
# Circuit(n_qubits=3, depth=3)
```

All gate methods return `self`, so you can chain them:

```python
qc = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
```

### Available Gate Methods

| Method | Gate |
|--------|------|
| `qc.h(q)` | Hadamard |
| `qc.x(q)`, `qc.y(q)`, `qc.z(q)` | Pauli gates |
| `qc.s(q)`, `qc.t(q)` | S and T gates |
| `qc.xsquare(q)` | sqrt(X) |
| `qc.rx(q, phi)`, `qc.ry(q, phi)`, `qc.rz(q, phi)` | Rotation gates |
| `qc.phase(q, phi)` | Phase gate |
| `qc.cnot(c, t)` / `qc.cx(c, t)` | CNOT |
| `qc.crx(c, t, phi)`, `qc.cry(c, t, phi)`, `qc.crz(c, t, phi)` | Controlled rotations |
| `qc.cphase(c, t, phi)` / `qc.cp(c, t, phi)` | Controlled phase |
| `qc.ccnot(c1, c2, t)` / `qc.toffoli(c1, c2, t)` | Toffoli |
| `qc.or_gate(c1, c2, t)` | OR gate |
| `qc.swap(q1, q2)` | SWAP |
| `qc.iswap(q1, q2)` | iSWAP |
| `qc.siswap(q1, q2)` | sqrt(iSWAP) |
| `qc.cswap(c, t1, t2)` | Fredkin |
| `qc.mcx(ctrls, t)` | Multi-controlled X |
| `qc.mcz(ctrls, t)` | Multi-controlled Z |
| `qc.mcp(ctrls, t, phi)` | Multi-controlled Phase |
| `qc.unitary(matrix, qubits)` | Arbitrary unitary |
| `qc.depolarize(q, p)` | Depolarizing noise |

## Executing a Circuit

### run() -- Fresh Execution

`run()` creates a new `|00...0>` wavefunction, applies all operations, and returns the result.

```python
qc = Circuit(2).h(0).cnot(0, 1)
wf = qc.run()

print(wf.probabilities())
# [0.5 0.  0.  0.5]
```

You can specify a backend:

```python
wf = qc.run(backend='cpu')
wf = qc.run(backend='python')
```

### Calling a Circuit on an Existing Wavefunction

Use the circuit as a callable to apply it to an existing wavefunction.

```python
wf = Qubit(2)
qc = Circuit(2).h(0).cnot(0, 1)
qc(wf)  # Modifies wf in-place

print(wf.probabilities())
# [0.5 0.  0.  0.5]
```

## Inspecting a Circuit

A `Circuit` is iterable and indexable.

```python
qc = Circuit(3).h(0).rx(1, 0.5).cnot(0, 2)

# Length
print(len(qc))  # 3

# Iterate
for op in qc:
    print(f"{op.name} on qubits {op.qubits}, params={op.params}")
# H on qubits (0,), params=()
# RX on qubits (1,), params=(0.5,)
# CNOT on qubits (0, 2), params=()

# Index
print(qc[0])  # GateOp(name='H', qubits=(0,), ...)
print(qc[-1]) # GateOp(name='CNOT', qubits=(0, 2), ...)
```

### Parameter Count

```python
qc = Circuit(2).ry(0, 0.5).ry(1, 1.2).cnot(0, 1)
print(qc.num_parameters)     # 2
print(qc.parameter_indices)  # [(0, 0), (1, 0)]
```

## Circuit Composition

### compose() -- Sequential Composition

Concatenate two circuits into a new one.

```python
# Preparation circuit
prep = Circuit(2).h(0).cnot(0, 1)

# Measurement rotation
meas = Circuit(2).ry(0, np.pi/4).ry(1, np.pi/4)

# Combined
full = prep.compose(meas)
print(full)  # Circuit(n_qubits=2, depth=4)

wf = full.run()
```

> **Note:** Both circuits must have the same number of qubits.

### copy() -- Deep Copy

```python
original = Circuit(2).h(0).cnot(0, 1)
duplicate = original.copy()
duplicate.rx(0, 0.5)  # Does not modify original

print(len(original))   # 2
print(len(duplicate))  # 3
```

## Adjoint (Inverse) Circuit

`adjoint()` returns a new circuit with all operations reversed and each gate replaced by its adjoint.

```python
qc = Circuit(2).h(0).rx(1, 0.5).cnot(0, 1)
qc_inv = qc.adjoint()

# Forward then inverse should return to |00>
full = qc.compose(qc_inv)
wf = full.run()

print(np.round(wf.probabilities(), 6))
# [1. 0. 0. 0.]  -- back to |00>
```

The adjoint rules are:

| Gate | Adjoint |
|------|---------|
| H, X, Y, Z | Self (Hermitian) |
| CNOT, CCNOT, SWAP | Self |
| RX(phi), RY(phi), RZ(phi) | RX(-phi), RY(-phi), RZ(-phi) |
| Phase(phi) | Phase(-phi) |
| CRX(phi), CRY(phi), CRZ(phi) | CRX(-phi), CRY(-phi), CRZ(-phi) |
| S | Phase(-pi/2) |
| T | Phase(-pi/4) |

## Parameter Binding

### bind_parameters()

Replace all numeric parameters in a circuit sequentially from a flat array.

```python
# Parameterized ansatz
ansatz = Circuit(2).ry(0, 0.0).ry(1, 0.0).cnot(0, 1)

# Bind specific values
bound = ansatz.bind_parameters(np.array([np.pi/4, np.pi/3]))
wf = bound.run()
print(wf.probabilities())
```

### Parameters via run()

You can also pass parameters directly to `run()`:

```python
ansatz = Circuit(2).ry(0, 0.0).ry(1, 0.0).cnot(0, 1)
wf = ansatz.run(params=np.array([np.pi/4, np.pi/3]))
```

### Variational Circuit Example

```python
def cost_function(params):
    """Evaluate the cost function for given parameters."""
    ansatz = Circuit(2).ry(0, 0.0).ry(1, 0.0).cnot(0, 1).ry(0, 0.0).ry(1, 0.0)
    wf = ansatz.run(params=params)
    from qforge.measurement import pauli_expectation
    return pauli_expectation(wf, 0, 'Z')

# Evaluate at different parameter values
for theta in np.linspace(0, np.pi, 5):
    params = np.array([theta, 0.5, theta/2, 0.3])
    print(f"theta={theta:.2f}: cost={cost_function(params):.4f}")
```

### Applying to Existing Wavefunctions with Parameters

```python
ansatz = Circuit(2).ry(0, 0.0).ry(1, 0.0).cnot(0, 1)

wf = Qubit(2)
ansatz(wf, params=np.array([np.pi/4, np.pi/3]))
print(wf.probabilities())
```

## Custom Gates

Define reusable gates from matrices or sub-circuits.

### Matrix-Defined Gate

```python
from qforge.ir import CustomGate, register_gate

# Define a sqrt(Z) gate
sqrt_z_matrix = np.array([[1, 0], [0, 1j]])
register_gate(CustomGate(
    name='SqrtZ',
    n_qubits=1,
    matrix=sqrt_z_matrix,
))

# Use in a circuit via unitary
qc = Circuit(1)
qc.unitary(sqrt_z_matrix, [0])
wf = qc.run()
```

### Sub-Circuit-Defined Gate

```python
# Define a Bell preparation gate as a sub-circuit
bell_prep = Circuit(2).h(0).cnot(0, 1)
register_gate(CustomGate(
    name='BellPrep',
    n_qubits=2,
    subcircuit=bell_prep,
))
```

### Unregistering Custom Gates

```python
from qforge.ir import unregister_gate
unregister_gate('SqrtZ')
```

## Mid-Circuit Measurement

The Circuit IR supports measuring qubits in the middle of a circuit and conditioning subsequent gates on the results.

### Basic Mid-Circuit Measurement

```python
qc = Circuit(2)
qc.h(0)
qc.measure(0, classical_bit=0)  # Measure qubit 0, store in classical bit 0
```

After execution, the measurement result is stored in the wavefunction's classical register:

```python
wf = qc.run()
print(wf.classical_register)
# ClassicalRegister([0 or 1, 0])
```

### Conditional Gates

Execute a gate only if a classical bit has a specific value.

```python
qc = Circuit(2)
qc.h(0)
qc.measure(0, classical_bit=0)

# Apply X to qubit 1 only if qubit 0 measured |1>
qc.c_if(
    classical_bit=0,
    expected_value=1,
    op=GateOp(name='X', qubits=(1,))
)
```

### Quantum Teleportation Example

```python
qc = Circuit(3)

# Prepare state to teleport on qubit 0
qc.rx(0, np.pi/3)

# Create Bell pair between qubits 1 and 2
qc.h(1)
qc.cnot(1, 2)

# Bell measurement on qubits 0 and 1
qc.cnot(0, 1)
qc.h(0)
qc.measure(0, classical_bit=0)
qc.measure(1, classical_bit=1)

# Conditional corrections on qubit 2
qc.c_if(1, 1, GateOp(name='X', qubits=(2,)))
qc.c_if(0, 1, GateOp(name='Z', qubits=(2,)))

wf = qc.run()
print("Classical register:", wf.classical_register)
print("Final probabilities:", np.round(wf.probabilities(), 4))
```

> **Note:** Since mid-circuit measurement is stochastic, each `run()` call may produce different classical register values and different final states. The teleported state on qubit 2 is always correct after the conditional corrections.

## Recording Functional Calls

The `record` context manager captures functional API calls into a Circuit.

```python
from qforge.ir import record
from qforge.gates import H, CNOT, RY

with record(3) as qc:
    wf = Qubit(3)
    H(wf, 0)
    RY(wf, 1, phi=0.5)
    CNOT(wf, 0, 2)

# qc captured all 3 operations
print(len(qc))  # 3
for op in qc:
    print(op.name, op.qubits, op.params)
# H (0,) ()
# RY (1,) (0.5,)
# CNOT (0, 2) ()

# Replay
wf2 = qc.run()
```

## Full Example: Parameterized Ansatz with Adjoint Verification

```python
from qforge.ir import Circuit
from qforge.measurement import pauli_expectation
import numpy as np

# Build a parameterized ansatz
def make_ansatz(n_qubits, n_layers):
    qc = Circuit(n_qubits)
    param_count = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(q, 0.0)  # Placeholder parameter
            param_count += 1
        for q in range(n_qubits - 1):
            qc.cnot(q, q + 1)
    return qc

ansatz = make_ansatz(4, 2)
print(f"Ansatz: {ansatz}")
print(f"Parameters: {ansatz.num_parameters}")

# Verify adjoint is correct: U U^dag = I
params = np.random.randn(ansatz.num_parameters)
forward = ansatz.bind_parameters(params)
backward = forward.adjoint()
roundtrip = forward.compose(backward)

wf = roundtrip.run()
print("Round-trip fidelity:", np.round(wf.probabilities()[0], 8))
# 1.0 -- returned to |0000>

# Evaluate cost function
wf = ansatz.run(params=params)
cost = sum(pauli_expectation(wf, q, 'Z') for q in range(4))
print(f"Cost = sum(<Z_i>) = {cost:.4f}")
```
