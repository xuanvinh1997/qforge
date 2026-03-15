# Qforge.chem

Quantum chemistry module providing molecular Hamiltonian construction,
fermionic-to-qubit operator mappings, and variational ansatz building blocks
for quantum chemistry simulations.

## Usage

```python
from qforge.chem import (
    Molecule, FermionicOperator,
    jordan_wigner, bravyi_kitaev,
    uccsd_ansatz, uccsd_n_params,
)

# Build a molecular Hamiltonian
mol = Molecule(atoms=[('H', (0, 0, 0)), ('H', (0, 0, 0.74))], basis='sto-3g')
ferm_op = mol.fermionic_hamiltonian()

# Map to qubit operator
qubit_ham = jordan_wigner(ferm_op)

# UCCSD ansatz
n_params = uccsd_n_params(n_electrons=2, n_orbitals=4)
```

## Sub-modules

| Module | Description |
|--------|-------------|
| [`molecule`](molecule.md) | `Molecule` class and Hamiltonian construction |
| [`fermion`](fermion.md) | `FermionicOperator`, Jordan-Wigner, Bravyi-Kitaev |
| [`ucc`](ucc.md) | UCCSD ansatz |
