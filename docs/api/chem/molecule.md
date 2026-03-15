# Qforge.chem.molecule

Molecular Hamiltonian construction from atomic geometry and basis set.

## Usage

```python
from Qforge.chem import Molecule

mol = Molecule(
    atoms=[('H', (0, 0, 0)), ('H', (0, 0, 0.74))],
    basis='sto-3g',
    charge=0,
    multiplicity=1,
)

# Get the fermionic Hamiltonian
ferm_ham = mol.fermionic_hamiltonian()

# Get one- and two-electron integrals
h1, h2 = mol.integrals()
print(f"Number of orbitals: {mol.n_orbitals}")
print(f"Number of electrons: {mol.n_electrons}")
```

## Class

### `Molecule`

```python
Molecule(atoms, basis='sto-3g', charge=0, multiplicity=1)
```

**Parameters:**

- `atoms` -- List of `(symbol, (x, y, z))` tuples specifying the molecular geometry (Angstroms).
- `basis` -- Basis set name (default `'sto-3g'`).
- `charge` -- Net molecular charge (default 0).
- `multiplicity` -- Spin multiplicity (default 1 = singlet).

**Properties:**

- `n_orbitals` -- Number of molecular orbitals.
- `n_electrons` -- Number of electrons.

**Methods:**

- `fermionic_hamiltonian()` -- Return a `FermionicOperator` for the molecular Hamiltonian.
- `integrals()` -- Return `(h1, h2)` one- and two-electron integrals.
- `nuclear_repulsion()` -- Nuclear repulsion energy (float).

## Full API

::: Qforge.chem.molecule
    options:
      show_source: false
