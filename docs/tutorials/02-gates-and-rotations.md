# Tutorial 2: Gates and Rotations

A complete walkthrough of every gate in Qforge with examples.

## Setup

All examples below start from:

```python
from Qforge.circuit import Qubit
from Qforge.gates import *
import numpy as np
```

## Single-Qubit Fixed Gates

These gates take a wavefunction and a qubit index.

### Hadamard (H)

Creates an equal superposition. Maps `|0> -> (|0>+|1>)/sqrt(2)`, `|1> -> (|0>-|1>)/sqrt(2)`.

```python
wf = Qubit(1)
H(wf, 0)
print(wf.probabilities())  # [0.5, 0.5]
```

### Pauli-X (X)

Bit flip. Maps `|0> -> |1>`, `|1> -> |0>`.

```python
wf = Qubit(1)
X(wf, 0)
print(wf.probabilities())  # [0., 1.]
```

### Pauli-Y (Y)

Maps `|0> -> i|1>`, `|1> -> -i|0>`.

```python
wf = Qubit(1)
Y(wf, 0)
print(wf.amplitude)  # [0.+0.j, 0.+1.j]
```

### Pauli-Z (Z)

Phase flip. Maps `|0> -> |0>`, `|1> -> -|1>`.

```python
wf = Qubit(1)
H(wf, 0)   # (|0>+|1>)/sqrt(2)
Z(wf, 0)   # (|0>-|1>)/sqrt(2)
print(wf.amplitude)  # [ 0.707+0.j, -0.707+0.j]
```

### S Gate

Phase gate with angle pi/2. Equivalent to `Phase(wf, n, pi/2)`. Maps `|1> -> i|1>`.

```python
wf = Qubit(1)
X(wf, 0)
S(wf, 0)
print(wf.amplitude)  # [0.+0.j, 0.+1.j]
```

### T Gate

Phase gate with angle pi/4. Maps `|1> -> exp(i*pi/4)|1>`.

```python
wf = Qubit(1)
X(wf, 0)
T(wf, 0)
print(wf.amplitude)  # [0.+0.j, 0.707+0.707j]
```

### Xsquare (sqrt(X))

Square root of NOT gate. Applying it twice equals X.

```python
wf = Qubit(1)
Xsquare(wf, 0)
Xsquare(wf, 0)
print(np.round(wf.probabilities(), 4))  # [0., 1.] -- same as X
```

## Rotation Gates

Parameterized single-qubit gates. All angles are in radians.

### RX -- Rotation around X

Matrix: `[[cos(phi/2), -i*sin(phi/2)], [-i*sin(phi/2), cos(phi/2)]]`

```python
wf = Qubit(1)
RX(wf, 0, phi=np.pi)     # Same as X (up to global phase)
print(np.round(wf.probabilities(), 4))  # [0., 1.]

wf = Qubit(1)
RX(wf, 0, phi=np.pi/2)   # Partial rotation
print(np.round(wf.probabilities(), 4))  # [0.5, 0.5]
```

### RY -- Rotation around Y

Matrix: `[[cos(phi/2), -sin(phi/2)], [sin(phi/2), cos(phi/2)]]`

```python
wf = Qubit(1)
RY(wf, 0, phi=np.pi)     # Same as Y (up to global phase)
print(np.round(wf.probabilities(), 4))  # [0., 1.]
```

### RZ -- Rotation around Z

Matrix: `[[exp(-i*phi/2), 0], [0, exp(i*phi/2)]]`

```python
wf = Qubit(1)
H(wf, 0)
RZ(wf, 0, phi=np.pi)
print(np.round(np.abs(wf.amplitude)**2, 4))  # [0.5, 0.5] -- phase change only
```

### Phase Gate

Matrix: `[[1, 0], [0, exp(i*phi)]]`. Applies a phase to the `|1>` component only.

```python
wf = Qubit(1)
X(wf, 0)
Phase(wf, 0, phi=np.pi/4)
print(wf.amplitude)  # [0.+0.j, 0.707+0.707j]
```

> **Tip:** `S` is equivalent to `Phase(wf, n, pi/2)` and `T` is equivalent to `Phase(wf, n, pi/4)`.

## Controlled Gates

Two-qubit gates where the operation on the target qubit is conditioned on the control qubit being `|1>`.

### CNOT (Controlled-X)

Flips the target when the control is `|1>`.

```python
wf = Qubit(2)
X(wf, 0)          # |10>
CNOT(wf, 0, 1)    # |11>
print(wf.probabilities())  # [0. 0. 0. 1.]
```

### CRX -- Controlled RX

```python
wf = Qubit(2)
X(wf, 0)
CRX(wf, 0, 1, phi=np.pi)   # Apply RX(pi) to qubit 1, controlled by qubit 0
print(np.round(wf.probabilities(), 4))  # [0. 0. 0. 1.]
```

### CRY -- Controlled RY

```python
wf = Qubit(2)
X(wf, 0)
CRY(wf, 0, 1, phi=np.pi/2)
print(np.round(wf.probabilities(), 4))  # [0.  0.  0.5 0.5]
```

### CRZ -- Controlled RZ

```python
wf = Qubit(2)
X(wf, 0)
H(wf, 1)
CRZ(wf, 0, 1, phi=np.pi)   # Phase rotation on qubit 1
print(np.round(wf.probabilities(), 4))  # [0.  0.  0.5 0.5]
```

### CPhase / CP -- Controlled Phase

Applies `exp(i*phi)` to the `|11>` component.

```python
wf = Qubit(2)
H(wf, 0)
H(wf, 1)
CPhase(wf, 0, 1, phi=np.pi)
# Equivalent: CP(wf, 0, 1, phi=np.pi)
```

## Multi-Controlled Gates

### CCNOT (Toffoli)

Flips target only when both controls are `|1>`.

```python
wf = Qubit(3)
X(wf, 0)
X(wf, 1)
CCNOT(wf, 0, 1, 2)   # Both controls are |1>, so qubit 2 flips
print(wf.probabilities())
# [0. 0. 0. 0. 0. 0. 0. 1.]  -- state |111>
```

### OR Gate

Flips target if control_1 OR control_2 is `|1>`.

```python
wf = Qubit(3)
X(wf, 0)             # Only qubit 0 is |1>
OR(wf, 0, 1, 2)      # qubit 0 is |1>, so qubit 2 flips
print(wf.probabilities())
# State: |101>
```

### mcx -- Generalized Multi-Controlled X

Supports any number of control qubits.

```python
wf = Qubit(4)
X(wf, 0)
X(wf, 1)
X(wf, 2)
mcx(wf, [0, 1, 2], 3)  # Flip qubit 3 if qubits 0,1,2 are all |1>
print(wf.probabilities())
# State: |1111>
```

### mcz -- Multi-Controlled Z

```python
wf = Qubit(3)
X(wf, 0)
X(wf, 1)
X(wf, 2)
mcz(wf, [0, 1], 2)   # Apply Z to qubit 2 if qubits 0,1 are |1>
# Adds phase -1 to |111>
```

### mcp -- Multi-Controlled Phase

```python
wf = Qubit(3)
X(wf, 0)
X(wf, 1)
X(wf, 2)
mcp(wf, [0, 1], 2, np.pi/4)  # Apply Phase(pi/4) to qubit 2 if controls are |1>
```

## Swap Gates

### SWAP

Exchanges the states of two qubits.

```python
wf = Qubit(2)
X(wf, 0)             # |10>
SWAP(wf, 0, 1)       # |01>
print(wf.probabilities())  # [0. 1. 0. 0.]
```

### CSWAP (Fredkin Gate)

Conditional swap -- swaps two target qubits when the control is `|1>`.

```python
wf = Qubit(3)
X(wf, 0)             # Control = |1>
X(wf, 1)             # Target 1 = |1>
# State: |110>
CSWAP(wf, 0, 1, 2)   # Swap qubits 1 and 2 because control is |1>
# State: |101>
print(wf.probabilities())
```

### ISWAP

Swaps two qubits and applies a phase factor of `i`.

```python
wf = Qubit(2)
X(wf, 0)             # |10>
ISWAP(wf, 0, 1)      # i|01>
print(wf.amplitude)   # [0, i, 0, 0]
```

### SISWAP (sqrt(iSWAP))

Square root of iSWAP. Applying twice equals ISWAP.

```python
wf = Qubit(2)
X(wf, 0)
SISWAP(wf, 0, 1)
print(np.round(wf.probabilities(), 4))  # [0.  0.5 0.5 0. ]
```

## Arbitrary Unitary (QubitUnitary)

Apply any unitary matrix to one or two qubits.

### Single-Qubit Unitary

```python
# Custom sqrt(Z) gate
sqrt_z = np.array([[1, 0], [0, 1j]])
wf = Qubit(1)
X(wf, 0)
QubitUnitary(wf, sqrt_z, [0])
print(wf.amplitude)  # [0, 1j]
```

### Two-Qubit Unitary

```python
# Custom CNOT matrix (for verification)
cnot_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])
wf = Qubit(2)
H(wf, 0)
QubitUnitary(wf, cnot_matrix, [0, 1])
print(np.round(wf.probabilities(), 4))  # [0.5 0.  0.  0.5]
```

## Noise Channels

### E -- Single-Qubit Depolarizing Channel

Applies depolarizing noise with probability `p` to a single qubit.

```python
wf = Qubit(1)
X(wf, 0)
E(wf, 0.1, 0)   # 10% depolarizing noise on qubit 0
print(wf.probabilities())  # ~[0.05, 0.95]
```

> **Note:** The function signature is `E(wf, p, qubit)` -- the probability comes before the qubit index.

### E_all -- Depolarizing Channel on All Qubits

```python
wf = Qubit(3)
X(wf, 0)
H(wf, 1)
E_all(wf, 0.01, 3)  # 1% noise on all 3 qubits
```

## Gate Summary Table

| Gate | Signature | Description |
|------|-----------|-------------|
| `H` | `H(wf, n)` | Hadamard |
| `X` | `X(wf, n)` | Pauli-X (bit flip) |
| `Y` | `Y(wf, n)` | Pauli-Y |
| `Z` | `Z(wf, n)` | Pauli-Z (phase flip) |
| `S` | `S(wf, n)` | Phase(pi/2) |
| `T` | `T(wf, n)` | Phase(pi/4) |
| `Xsquare` | `Xsquare(wf, n)` | sqrt(X) |
| `RX` | `RX(wf, n, phi=angle)` | X-rotation |
| `RY` | `RY(wf, n, phi=angle)` | Y-rotation |
| `RZ` | `RZ(wf, n, phi=angle)` | Z-rotation |
| `Phase` | `Phase(wf, n, phi=angle)` | Phase gate |
| `CNOT` | `CNOT(wf, ctrl, tgt)` | Controlled-X |
| `CRX` | `CRX(wf, ctrl, tgt, phi=angle)` | Controlled-RX |
| `CRY` | `CRY(wf, ctrl, tgt, phi=angle)` | Controlled-RY |
| `CRZ` | `CRZ(wf, ctrl, tgt, phi=angle)` | Controlled-RZ |
| `CPhase`/`CP` | `CPhase(wf, ctrl, tgt, phi=angle)` | Controlled-Phase |
| `CCNOT` | `CCNOT(wf, c1, c2, tgt)` | Toffoli |
| `OR` | `OR(wf, c1, c2, tgt)` | OR gate |
| `mcx` | `mcx(wf, [ctrls], tgt)` | Multi-controlled X |
| `mcz` | `mcz(wf, [ctrls], tgt)` | Multi-controlled Z |
| `mcp` | `mcp(wf, [ctrls], tgt, phi)` | Multi-controlled Phase |
| `SWAP` | `SWAP(wf, t1, t2)` | Swap |
| `CSWAP` | `CSWAP(wf, ctrl, t1, t2)` | Fredkin |
| `ISWAP` | `ISWAP(wf, t1, t2)` | iSWAP |
| `SISWAP` | `SISWAP(wf, t1, t2)` | sqrt(iSWAP) |
| `QubitUnitary` | `QubitUnitary(wf, matrix, [qubits])` | Arbitrary unitary |
| `E` | `E(wf, p, n)` | Depolarizing noise |
| `E_all` | `E_all(wf, p, n_qubits)` | Depolarizing noise (all) |
