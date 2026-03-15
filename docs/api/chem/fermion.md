# Qforge.chem.fermion

Fermionic operators and fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev).

## Usage

```python
from qforge.chem import FermionicOperator, jordan_wigner, bravyi_kitaev

# Create a fermionic operator from one- and two-body integrals
ferm_op = FermionicOperator(h1=one_body, h2=two_body)

# Map to qubit Hamiltonian
qubit_ham_jw = jordan_wigner(ferm_op)
qubit_ham_bk = bravyi_kitaev(ferm_op)
```

## Classes

### `FermionicOperator`

```python
FermionicOperator(h1, h2, constant=0.0)
```

**Parameters:**

- `h1` -- One-body integrals (2D numpy array, n_orbitals x n_orbitals).
- `h2` -- Two-body integrals (4D numpy array, chemist notation).
- `constant` -- Constant energy offset (e.g. nuclear repulsion).

**Properties:**

- `n_modes` -- Number of fermionic modes (orbitals).

## Functions

### `jordan_wigner(fermionic_op) -> Hamiltonian`

Transform a `FermionicOperator` into a qubit `Hamiltonian` using the
Jordan-Wigner mapping. Each fermionic mode maps to one qubit with
Pauli-Z strings encoding the parity.

**Returns:** `qforge.algo.Hamiltonian`.

---

### `bravyi_kitaev(fermionic_op) -> Hamiltonian`

Transform a `FermionicOperator` into a qubit `Hamiltonian` using the
Bravyi-Kitaev mapping. Provides a balance between locality and qubit
overhead compared to Jordan-Wigner.

**Returns:** `qforge.algo.Hamiltonian`.

## Full API

::: qforge.chem.fermion
    options:
      show_source: false
