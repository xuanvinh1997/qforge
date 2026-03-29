# Qforge.qudit_gates

Qudit gate library for quantum systems with local dimension d > 2.
Includes gates for qutrits (d=3), general qudits, parameterized rotations
via Gell-Mann generators, and entangling gates. All gates use
C++ backend dispatch with Python fallback.

## Quick Start

```python
from qforge.circuit import Qudit
from qforge.qudit_gates import Hd, X01, CSUM, R01, measure_qudit, GELL_MANN

# Create a 3-qutrit circuit
qc = Qudit(3, dimension=3)

# Apply qutrit Hadamard
Hd(qc, 0)

# Entangle with CSUM (qutrit analog of CNOT)
CSUM(qc, 0, 1)

# Measure
probs = measure_qudit(qc, 0)   # [P(0), P(1), P(2)]
```

## Single-Qudit Gates

### `Hd(wavefunction, target)`

Qudit Hadamard gate (Discrete Fourier Transform / sqrt(d)).

$$H_d[j,k] = \frac{\omega^{jk}}{\sqrt{d}}, \quad \omega = e^{2\pi i / d}$$

### `X01(wavefunction, target)`

Swap levels |0> and |1>. Qutrit Pauli-X analog for the 0-1 subspace.

### `X02(wavefunction, target)`

Swap levels |0> and |2>.

### `X12(wavefunction, target)`

Swap levels |1> and |2>.

### `CLOCK(wavefunction, target)`

Cyclic shift (increment) gate: |k> -> |k+1 mod d>.

### `ZPHASE(wavefunction, target)`

Ternary phase gate: diag(1, omega, omega^2, ...) where omega = exp(2*pi*i/d).

## Parameterized Rotation Gates

### `R01(wavefunction, target, theta)`

Givens rotation in the |0>-|1> subspace by angle `theta`.

### `R02(wavefunction, target, theta)`

Givens rotation in the |0>-|2> subspace by angle `theta`.

### `R12(wavefunction, target, theta)`

Givens rotation in the |1>-|2> subspace by angle `theta`.

### `RGM(wavefunction, target, generator, angle)`

Rotation by Gell-Mann generator: exp(-i * angle * lambda_k / 2).

**Parameters:**

- `target` -- Qudit index.
- `generator` -- Gell-Mann matrix index (1-8 for qutrits).
- `angle` -- Rotation angle in radians.

## Entangling Gates

### `CSUM(wavefunction, control, target)`

CSUM gate (qutrit analog of CNOT): |c, t> -> |c, (t + c) mod d>.

### `QUDIT_SWAP(wavefunction, t1, t2)`

Swap two qudits (works for any dimension d).

### `apply_qudit_gate(wavefunction, target, gate)`

Apply an arbitrary d x d unitary matrix to a single qudit.

### `apply_controlled_qudit_gate(wavefunction, control, ctrl_val, target, gate)`

Apply a d x d gate to `target` qudit, conditioned on `control` qudit having value `ctrl_val`.

## Measurement

### `measure_qudit(wavefunction, qudit)`

Return probabilities [P(0), P(1), ..., P(d-1)] for a single qudit.

### `collapse_qudit(wavefunction, qudit)`

Collapse a qudit via projective measurement. Returns the measured value (int in [0, d)).

### `qudit_expectation(wavefunction, qudit, operator)`

Expectation value of a d x d Hermitian operator on a single qudit.

## Constants

### `GELL_MANN`

List of 8 Gell-Mann matrices (3x3 numpy arrays), the SU(3) generators for qutrits.

| Index | Name      | Type                     |
|-------|-----------|--------------------------|
| 0     | lambda_1  | Off-diagonal real (0-1)  |
| 1     | lambda_2  | Off-diagonal imag (0-1)  |
| 2     | lambda_3  | Diagonal (0-1)           |
| 3     | lambda_4  | Off-diagonal real (0-2)  |
| 4     | lambda_5  | Off-diagonal imag (0-2)  |
| 5     | lambda_6  | Off-diagonal real (1-2)  |
| 6     | lambda_7  | Off-diagonal imag (1-2)  |
| 7     | lambda_8  | Diagonal (normalized)    |

## Full API

::: qforge.qudit_gates
    options:
      show_source: false
