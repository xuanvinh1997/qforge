# Qforge.algo.standard

Standard quantum algorithms: Quantum Fourier Transform, Quantum Phase Estimation,
Grover's search, and the HHL algorithm for linear systems.

## Usage

```python
from Qforge.algo.standard import qft, inverse_qft, qpe, grover_search, hhl
from Qforge.circuit import Qubit

# QFT
wf = Qubit(4)
qft(wf)

# Grover's search
wf, result = grover_search(n_qubits=4, oracle_fn=my_oracle)
```

## Functions

### `qft(wavefunction)`

Apply the Quantum Fourier Transform to the wavefunction in-place.

---

### `inverse_qft(wavefunction)`

Apply the inverse QFT to the wavefunction in-place.

---

### `qpe(unitary, eigenvector, n_counting_qubits)`

Quantum Phase Estimation. Estimates the eigenvalue phase of a unitary operator
given an approximate eigenvector.

**Parameters:**

- `unitary` -- Unitary matrix (numpy array).
- `eigenvector` -- Initial state Wavefunction approximating an eigenvector.
- `n_counting_qubits` -- Number of ancilla qubits for phase precision.

**Returns:** Estimated phase as a float.

---

### `grover_search(n_qubits, oracle_fn, n_iterations=None)`

Run Grover's search algorithm.

**Parameters:**

- `n_qubits` -- Number of qubits.
- `oracle_fn` -- Oracle function `callable(wf) -> None` that marks target states.
- `n_iterations` -- Number of Grover iterations (default: optimal `floor(pi/4 * sqrt(N))`).

**Returns:** `(wavefunction, measurement_result)`.

---

### `grover_oracle(wavefunction, target_state)`

Apply a phase oracle that flips the sign of the target computational basis state.

---

### `grover_diffusion(wavefunction)`

Apply the Grover diffusion operator (inversion about the mean).

---

### `hhl(A, b, n_qubits=None)`

HHL algorithm for solving the linear system `Ax = b`.

**Parameters:**

- `A` -- Hermitian matrix.
- `b` -- Right-hand side vector.
- `n_qubits` -- Number of qubits (inferred from `A` if not given).

**Returns:** Solution state as a Wavefunction.

## Full API

::: Qforge.algo.standard
    options:
      show_source: false
