# Qforge.algo.hamiltonian

Hamiltonian representation as a weighted sum of Pauli strings, with efficient
expectation value computation.

## Usage

```python
from Qforge.algo.hamiltonian import Hamiltonian
from Qforge.circuit import Qubit
from Qforge.gates import H

# H = -1.0 * Z_0 + 0.5 * X_0 X_1
ham = Hamiltonian(
    coeffs=[-1.0, 0.5],
    terms=[[('Z', 0)], [('X', 0), ('X', 1)]],
)

wf = Qubit(2)
H(wf, 0)
energy = ham.expectation(wf)
print(f"<H> = {energy:.4f}")
```

## Class

### `Hamiltonian`

```python
Hamiltonian(coeffs, terms)
```

**Parameters:**

- `coeffs` -- List of real coefficients `[c_0, c_1, ...]`.
- `terms` -- List of Pauli strings, where each string is a list of `(pauli, qubit)` tuples. An empty list `[]` represents the identity operator.

**Methods:**

- `expectation(wavefunction)` -- Compute `<psi|H|psi>` analytically from the amplitude vector.
- `to_matrix(n_qubits)` -- Return the full 2^n x 2^n Hamiltonian matrix (for small systems).

## Full API

::: Qforge.algo.hamiltonian
    options:
      show_source: false
