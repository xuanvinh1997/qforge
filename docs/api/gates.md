# Qforge.gates

Quantum gate library with 30+ gates spanning single-qubit, controlled, entangling,
multi-controlled, and noise operations. Every gate function uses a dual-dispatch
pattern: the C++ kernel is called when available, falling back to a pure-Python
implementation otherwise.

All gate functions modify the wavefunction **in-place** and append to `wf.visual`
for circuit visualization.

## Usage

```python
from qforge.circuit import Qubit
from qforge.gates import H, X, CNOT, RY, SWAP, E

wf = Qubit(3)
H(wf, 0)                  # Hadamard on qubit 0
X(wf, 1)                  # Pauli-X on qubit 1
CNOT(wf, 0, 2)            # CNOT: control=0, target=2
RY(wf, 1, 0.5)            # RY rotation by 0.5 rad
SWAP(wf, 0, 1)            # Swap qubits 0 and 1
E(wf, 0, 0.01)            # Depolarizing noise on qubit 0
```

## Single-Qubit Gates

### `H(wavefunction, n)`

Hadamard gate on qubit `n`.

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

### `X(wavefunction, n)`

Pauli-X (NOT) gate on qubit `n`.

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### `Y(wavefunction, n)`

Pauli-Y gate on qubit `n`.

$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

### `Z(wavefunction, n)`

Pauli-Z gate on qubit `n`.

$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### `S(wavefunction, n)`

S (phase) gate on qubit `n`. Equivalent to `Phase(wf, n, pi/2)`.

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

### `T(wavefunction, n)`

T gate on qubit `n`. Equivalent to `Phase(wf, n, pi/4)`.

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

### `Xsquare(wavefunction, n)`

Square root of X gate on qubit `n`.

$$\sqrt{X} = \frac{1}{2} \begin{pmatrix} 1+i & 1-i \\ 1-i & 1+i \end{pmatrix}$$

## Rotation Gates

### `RX(wavefunction, n, theta)`

Rotation about X-axis by angle `theta` (radians).

$$R_X(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

### `RY(wavefunction, n, theta)`

Rotation about Y-axis by angle `theta` (radians).

$$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

### `RZ(wavefunction, n, theta)`

Rotation about Z-axis by angle `theta` (radians).

$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### `Phase(wavefunction, n, theta)`

Phase gate: applies `exp(i*theta)` to the |1> amplitude of qubit `n`.

$$P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$$

## Controlled Gates

### `CNOT(wavefunction, control, target)`

Controlled-NOT (CX) gate.

### `CCNOT(wavefunction, control1, control2, target)`

Toffoli (double-controlled NOT) gate.

### `CRX(wavefunction, control, target, theta)`

Controlled-RX rotation.

### `CRY(wavefunction, control, target, theta)`

Controlled-RY rotation.

### `CRZ(wavefunction, control, target, theta)`

Controlled-RZ rotation.

### `CPhase(wavefunction, control, target, theta)` / `CP`

Controlled-Phase gate. `CP` is an alias for `CPhase`.

### `OR(wavefunction, control, target)`

Controlled-OR gate: flips the target if the control is |0>.

## Entangling Gates

### `SWAP(wavefunction, n1, n2)`

Swap the states of qubits `n1` and `n2`.

### `CSWAP(wavefunction, control, n1, n2)`

Fredkin (controlled-SWAP) gate.

### `ISWAP(wavefunction, n1, n2)`

iSWAP gate: swaps and applies a phase of `i` to the swapped components.

### `SISWAP(wavefunction, n1, n2)`

Square root of iSWAP gate.

## Multi-Controlled Gates

### `mcx(wavefunction, controls, target)`

Multi-controlled X gate. `controls` is a list of control qubit indices.

### `mcz(wavefunction, controls, target)`

Multi-controlled Z gate. `controls` is a list of control qubit indices.

### `mcp(wavefunction, controls, target, theta)`

Multi-controlled Phase gate. `controls` is a list of control qubit indices.

## Custom Unitary

### `QubitUnitary(wavefunction, qubits, matrix)`

Apply an arbitrary unitary matrix to the specified qubits.

**Parameters:**

- `wavefunction` -- Target wavefunction.
- `qubits` -- Qubit index (int) or list of qubit indices.
- `matrix` -- Unitary matrix (numpy array).

## Noise Gates

### `E(wavefunction, n, p)`

Single-qubit depolarizing channel with probability `p` on qubit `n`.

### `E_all(wavefunction, p)`

Apply the depolarizing channel with probability `p` to all qubits.

## Full API

::: qforge.gates
    options:
      show_source: false
