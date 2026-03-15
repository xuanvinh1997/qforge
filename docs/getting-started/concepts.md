# Core Concepts

## Wavefunction Model

Qforge simulates quantum circuits using the **statevector** (wavefunction) model. An n-qubit system is represented as a complex amplitude vector of length 2^n over the computational basis states.

```python
from qforge.circuit import Qubit

wf = Qubit(3)  # 2^3 = 8 amplitudes, initialized to |000>
print(len(wf.amplitude))  # 8
print(wf.amplitude[0])    # (1+0j) -- all probability in |000>
```

Gates operate directly on this amplitude vector in place, with O(2^n) time complexity per gate. This is exponentially more efficient than density matrix simulation (O(4^n)) and allows Qforge to handle 20+ qubit systems with the C++ engine.

The `Wavefunction` object exposes:

| Attribute/Method | Description |
|-----------------|-------------|
| `wf.amplitude` | Complex amplitude array (numpy). Zero-copy view when using C++ backend. |
| `wf.state` | Array of basis state label strings (`['000', '001', ...]`) |
| `wf.probabilities()` | Returns `|amplitude|^2` as a numpy array |
| `wf.print_state()` | Human-readable bra-ket string |
| `wf.visual` | List of gate operations (used by `visual_circuit()`) |
| `wf.visual_circuit()` | Prints an ASCII circuit diagram |

## Qubit Indexing

Qforge uses **big-endian** qubit ordering: qubit 0 is the **leftmost** (most-significant) bit.

```
Qubit:    0   1   2
State:   |q0  q1  q2>
Binary:   MSB       LSB
```

For example, in a 3-qubit system:

| Index | Binary | Qubit 0 | Qubit 1 | Qubit 2 |
|-------|--------|---------|---------|---------|
| 0 | 000 | 0 | 0 | 0 |
| 1 | 001 | 0 | 0 | 1 |
| 2 | 010 | 0 | 1 | 0 |
| 3 | 011 | 0 | 1 | 1 |
| 4 | 100 | 1 | 0 | 0 |
| 5 | 101 | 1 | 0 | 1 |
| 6 | 110 | 1 | 1 | 0 |
| 7 | 111 | 1 | 1 | 1 |

This means `wf.amplitude[4]` corresponds to state `|100>` where qubit 0 is `|1>` and qubits 1, 2 are `|0>`.

```python
from qforge.circuit import Qubit
from qforge.gates import X

wf = Qubit(3)
X(wf, 0)          # Flip qubit 0 (MSB)
print(wf.probabilities())
# [0. 0. 0. 0. 1. 0. 0. 0.]  -- index 4 = |100>
```

> **Warning:** This is the opposite convention from some frameworks (e.g., Qiskit uses little-endian by default). Be careful when comparing results across frameworks.

## Two APIs: Functional and Circuit IR

Qforge provides two ways to build quantum programs.

### Functional API

Gates are standalone functions that modify a wavefunction in place. This is the most direct way to work with Qforge.

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT, RY

wf = Qubit(2)
H(wf, 0)
RY(wf, 1, phi=0.5)
CNOT(wf, 0, 1)
```

Advantages: simple, immediate execution, full numpy access at every step.

### Circuit IR API

The `Circuit` class records gate operations as an instruction list that can be inspected, transformed, composed, and replayed.

```python
from qforge.ir import Circuit

qc = Circuit(2)
qc.h(0).ry(1, 0.5).cnot(0, 1)

# Inspect
print(qc)               # Circuit(n_qubits=2, depth=3)
print(qc.ops)           # [GateOp(name='H', ...), ...]

# Transform
qc_inv = qc.adjoint()   # Inverse circuit
qc_big = qc.compose(qc) # Self-composed

# Execute
wf = qc.run(backend='auto')
```

Advantages: deferred execution, composition, adjoint, parameter binding, mid-circuit measurement.

> **Tip:** Both APIs produce identical results. Use the functional API for quick experiments and the Circuit IR for variational algorithms, circuit transformations, or when you need to replay the same circuit with different parameters.

## Backends

Qforge automatically selects the fastest available backend.

| Backend | Flag | Description |
|---------|------|-------------|
| `'cuda'` | `_HAS_CUDA` | NVIDIA GPU via CUDA |
| `'metal'` | `_HAS_METAL` | Apple Silicon GPU via Metal |
| `'cpu'` | `_HAS_CPP` | C++ engine with OpenMP, 64-byte aligned memory |
| `'python'` | always | Pure numpy fallback |

The default `backend='auto'` selects in priority order: CUDA > Metal > C++ > Python.

```python
from qforge.circuit import Qubit

wf_auto   = Qubit(10)                    # Best available
wf_cpu    = Qubit(10, backend='cpu')     # Force C++
wf_python = Qubit(10, backend='python') # Force pure Python
```

You can check which backends are compiled:

```python
from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
print(f"C++: {_HAS_CPP}, CUDA: {_HAS_CUDA}, Metal: {_HAS_METAL}")
```

All gate functions automatically dispatch to the correct backend based on the wavefunction's internal state. You never need to change gate code when switching backends.

## Gate Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Gate functions | UPPERCASE | `H`, `X`, `CNOT`, `RX`, `CRZ`, `CCNOT` |
| Measurement functions | snake_case | `measure_all`, `measure_one`, `pauli_expectation` |
| Analysis classes | PascalCase | `PauliZExpectation`, `EntanglementEntropy` |
| Circuit IR methods | lowercase | `qc.h()`, `qc.cnot()`, `qc.rx()` |

## Angles and Phase Convention

All rotation angles are in **radians**. The phase convention is `exp(1j * phi)`.

```python
import numpy as np
from qforge.gates import RX, Phase

RX(wf, 0, phi=np.pi)       # pi rotation around X-axis
Phase(wf, 0, phi=np.pi/2)  # Apply exp(i*pi/2) = i to |1> component
```
