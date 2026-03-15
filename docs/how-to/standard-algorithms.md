# How-To: Use Standard Quantum Algorithms

Qforge includes implementations of foundational quantum algorithms. This guide
shows how to use each one.

## Quantum Fourier Transform (QFT)

The QFT transforms the computational basis into the Fourier basis. It is a
key subroutine in phase estimation, Shor's algorithm, and more.

```python
from Qforge.circuit import Qubit
from Qforge.gates import X
from Qforge.algo.standard import qft, inverse_qft

# Prepare state |5> = |101>
qc = Qubit(n_qubits=3)
X(qc, target=0)
X(qc, target=2)

# Apply QFT
qft(qc, qubits=[0, 1, 2])

# Inspect amplitudes (should be Fourier coefficients of |5>)
import numpy as np
print("Amplitudes after QFT:")
for i, amp in enumerate(qc.wavefunction.amplitude):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}>: {amp:.4f}")
```

### Inverse QFT

```python
qc2 = Qubit(n_qubits=3)
X(qc2, target=0)
X(qc2, target=2)

# QFT then inverse QFT should return to the original state
qft(qc2, qubits=[0, 1, 2])
inverse_qft(qc2, qubits=[0, 1, 2])

# Should be back to |101>
print("Amplitudes after QFT + iQFT:")
for i, amp in enumerate(qc2.wavefunction.amplitude):
    if abs(amp) > 1e-10:
        print(f"  |{i:03b}>: {amp:.4f}")
```

> **Note:** `qft` and `inverse_qft` accept a `qubits` parameter to apply the
> transform on a subset of qubits. If omitted, it applies to all qubits.

## Quantum Phase Estimation (QPE)

QPE estimates the eigenvalue phase of a unitary operator. Given U|psi> =
e^(2*pi*i*phi)|psi>, QPE outputs phi in a register of counting qubits.

```python
from Qforge.circuit import Qubit
from Qforge.gates import X, Phase
from Qforge.algo.standard import qpe
from Qforge.measurement import measure_all
import numpy as np

# Estimate the phase of a T gate (phase = pi/4, so phi = 1/8)
# T|1> = e^(i*pi/4)|1>

n_counting = 4  # Precision qubits

qc = Qubit(n_qubits=n_counting + 1)

# Prepare eigenstate |1> on the last qubit
X(qc, target=n_counting)

# Run QPE
qpe(
    qc,
    unitary_gate='T',
    counting_qubits=list(range(n_counting)),
    target_qubits=[n_counting],
)

# Measure the counting register
result = measure_all(qc)
counting_bits = result[:n_counting]
phase_estimate = int(counting_bits, 2) / (2 ** n_counting)

print(f"Measured: {counting_bits}")
print(f"Estimated phase: {phase_estimate:.4f}")
print(f"Exact phase:     {1/8:.4f}")
print(f"Estimated angle: {2 * np.pi * phase_estimate:.4f}")
print(f"Exact angle:     {np.pi / 4:.4f}")
```

### QPE with a Custom Unitary

```python
from Qforge.gates import QubitUnitary

# Phase gate with theta = 2*pi * 0.3
theta = 2 * np.pi * 0.3
custom_unitary = np.array([
    [1, 0],
    [0, np.exp(1j * theta)],
])

n_counting = 6  # More qubits = higher precision

qc = Qubit(n_qubits=n_counting + 1)
X(qc, target=n_counting)

qpe(
    qc,
    unitary_gate=custom_unitary,
    counting_qubits=list(range(n_counting)),
    target_qubits=[n_counting],
)

result = measure_all(qc)
counting_bits = result[:n_counting]
phase_estimate = int(counting_bits, 2) / (2 ** n_counting)
print(f"Estimated phase: {phase_estimate:.4f} (exact: 0.3)")
```

> **Tip:** QPE precision scales as 1/2^n where n is the number of counting
> qubits. For most applications, 8-10 counting qubits provide sufficient
> precision.

## Grover's Search

Grover's algorithm finds a marked item in an unstructured database with
quadratic speedup.

```python
from Qforge.algo.standard import grover_search
from Qforge.measurement import measure_all

# Search for |101> in a 3-qubit space
def oracle(qc, qubits):
    """Marks the state |101>."""
    from Qforge.gates import X, CCNOT
    # Flip qubit 1 (so |101> becomes |111>)
    X(qc, target=qubits[1])
    # Apply multi-controlled Z via CCNOT + H
    CCNOT(qc, control1=qubits[0], control2=qubits[1], target=qubits[2])
    # Unflip qubit 1
    X(qc, target=qubits[1])

qc = grover_search(
    n_qubits=3,
    oracle=oracle,
    n_iterations=1,  # Optimal for N=8 is floor(pi/4 * sqrt(8)) ~ 2
)

# Sample the result
counts = {}
for _ in range(100):
    qc = grover_search(n_qubits=3, oracle=oracle, n_iterations=2)
    result = measure_all(qc)
    counts[result] = counts.get(result, 0) + 1

print("Search results:")
for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  |{bitstring}>: {count}")
```

### Grover with Multiple Solutions

```python
def multi_oracle(qc, qubits):
    """Marks |01> and |10> in a 2-qubit space."""
    from Qforge.gates import CNOT, Z
    # This oracle flips the phase of states where qubits differ
    CNOT(qc, control=qubits[0], target=qubits[1])
    Z(qc, target=qubits[1])
    CNOT(qc, control=qubits[0], target=qubits[1])

qc = grover_search(
    n_qubits=2,
    oracle=multi_oracle,
    n_iterations=1,  # Optimal for 2 solutions in N=4
)

result = measure_all(qc)
print(f"Found: |{result}>")  # Should be |01> or |10>
```

> **Note:** The optimal number of Grover iterations is approximately
> (pi/4) * sqrt(N/M) where N is the search space size and M is the number of
> solutions. Too many iterations causes the amplitude to rotate past the
> target.

## HHL Algorithm

The Harrow-Hassidim-Lloyd (HHL) algorithm solves linear systems Ax = b with
exponential speedup for sparse, well-conditioned matrices.

```python
from Qforge.algo.standard import hhl
import numpy as np

# Solve Ax = b where A is a 2x2 Hermitian matrix
A = np.array([
    [1.5, 0.5],
    [0.5, 1.5],
])
b = np.array([1.0, 0.0])

# HHL returns the quantum state proportional to A^{-1}b
qc, solution_qubits = hhl(
    matrix=A,
    vector=b,
    n_precision=4,  # Precision qubits for eigenvalue inversion
)

# Extract the solution from the quantum state
# (Post-select on the ancilla qubit being |1>)
print(f"Quantum solution state prepared on qubits {solution_qubits}")

# Classical verification
x_classical = np.linalg.solve(A, b)
print(f"Classical solution: {x_classical}")
```

> **Warning:** HHL has significant caveats for practical use:
> 1. The matrix A must be Hermitian (use (A^dagger A) trick for non-Hermitian).
> 2. The solution is a quantum state -- extracting all amplitudes defeats the
>    speedup.
> 3. The circuit depth scales with the condition number of A.
> 4. Practical quantum advantage requires very large, sparse systems.

## Combining Algorithms

Use QPE as a subroutine in a larger computation:

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, X
from Qforge.algo.standard import qft, inverse_qft, qpe

# Example: use QFT in a custom algorithm
qc = Qubit(n_qubits=4)

# Prepare some state
for i in range(4):
    H(qc, target=i)

# Apply QFT to first 3 qubits
qft(qc, qubits=[0, 1, 2])

# Do something in the Fourier domain...

# Apply inverse QFT
inverse_qft(qc, qubits=[0, 1, 2])
```

## Summary

| Algorithm | API | Key Parameters |
|-----------|-----|----------------|
| QFT | `qft(qc, qubits)` | `qubits`: which qubits to transform |
| Inverse QFT | `inverse_qft(qc, qubits)` | Same as QFT |
| QPE | `qpe(qc, unitary_gate, counting_qubits, target_qubits)` | Precision from counting qubit count |
| Grover | `grover_search(n_qubits, oracle, n_iterations)` | `oracle(qc, qubits)` function |
| HHL | `hhl(matrix, vector, n_precision)` | `matrix` must be Hermitian |
