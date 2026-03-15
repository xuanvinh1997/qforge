# Tutorial 11: Quantum Chemistry with Qforge

This tutorial demonstrates how to simulate molecular systems using Qforge's
chemistry module. We compute the ground-state energy of H2 using the
Jordan-Wigner transformation and a UCCSD ansatz optimized with VQE.

## 1. Define a Molecule

```python
from Qforge.chem import Molecule

# Hydrogen molecule at equilibrium bond length
h2 = Molecule(
    atoms=[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.735))],
    basis='sto-3g',
    charge=0,
    multiplicity=1,
)

print(f"Number of electrons:    {h2.n_electrons}")
print(f"Number of spin-orbitals: {h2.n_spin_orbitals}")
print(f"Nuclear repulsion:      {h2.nuclear_repulsion:.6f} Ha")
```

## 2. Jordan-Wigner Transformation

The Jordan-Wigner transformation maps the fermionic Hamiltonian to a qubit
Hamiltonian:

```python
from Qforge.chem import jordan_wigner

qubit_hamiltonian = jordan_wigner(h2)

print(f"Number of Pauli terms: {len(qubit_hamiltonian.terms)}")
print(f"Number of qubits:     {qubit_hamiltonian.n_qubits}")
print("\nHamiltonian terms:")
for coeff, term in zip(qubit_hamiltonian.coeffs, qubit_hamiltonian.terms):
    if abs(coeff) > 1e-8:
        term_str = ' '.join(f'{op}{q}' for op, q in term) if term else 'I'
        print(f"  {coeff:+.6f} * {term_str}")
```

> **Note:** The Jordan-Wigner transformation maps n spin-orbitals to n qubits.
> For H2 in STO-3G, this gives 4 qubits (2 spatial orbitals x 2 spins).
> Symmetry reduction can bring this down to 2 qubits.

## 3. UCCSD Ansatz

The Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz is a
chemically motivated variational circuit:

```python
from Qforge.chem import uccsd_ansatz

# Generate the UCCSD circuit function
ansatz_fn, n_params = uccsd_ansatz(
    n_qubits=h2.n_spin_orbitals,
    n_electrons=h2.n_electrons,
)

print(f"UCCSD parameters: {n_params}")
```

The UCCSD ansatz:
1. Prepares the Hartree-Fock reference state (computational basis state with
   the first n_electrons qubits set to |1>).
2. Applies parameterized single and double excitation operators.

## 4. VQE for H2

```python
from Qforge.algo import VQE, Adam
import numpy as np

vqe = VQE(
    n_qubits=h2.n_spin_orbitals,
    hamiltonian=qubit_hamiltonian,
    circuit_fn=ansatz_fn,
    n_layers=1,
)

result = vqe.optimize(
    optimizer=Adam(lr=0.05),
    n_iterations=200,
    initial_params=np.zeros(n_params),  # Start from HF solution
)

print(f"\nVQE Results for H2:")
print(f"  VQE energy:   {result.energy:.6f} Ha")
print(f"  Exact energy: -1.137276 Ha")
print(f"  Error:        {abs(result.energy - (-1.137276)):.6f} Ha")
print(f"  Chemical accuracy (1.6 mHa): "
      f"{'ACHIEVED' if abs(result.energy - (-1.137276)) < 0.0016 else 'NOT achieved'}")
```

> **Tip:** Starting from zero parameters corresponds to the Hartree-Fock
> solution, which is a good initial point for UCCSD. This avoids the barren
> plateau problem that affects random initialization.

## 5. Potential Energy Surface

Compute the dissociation curve of H2:

```python
import matplotlib.pyplot as plt

bond_lengths = np.linspace(0.3, 3.0, 25)
vqe_energies = []
hf_energies = []

for r in bond_lengths:
    mol = Molecule(
        atoms=[('H', (0, 0, 0)), ('H', (0, 0, r))],
        basis='sto-3g',
    )
    ham = jordan_wigner(mol)
    ansatz, n_p = uccsd_ansatz(
        n_qubits=mol.n_spin_orbitals,
        n_electrons=mol.n_electrons,
    )

    # Hartree-Fock energy (zero parameters)
    from Qforge.measurement import pauli_expectation
    qc_hf = ansatz(np.zeros(n_p))
    hf_energies.append(pauli_expectation(qc_hf, ham))

    # VQE energy
    vqe = VQE(
        n_qubits=mol.n_spin_orbitals,
        hamiltonian=ham,
        circuit_fn=ansatz,
        n_layers=1,
    )
    res = vqe.optimize(
        optimizer=Adam(lr=0.05),
        n_iterations=150,
        initial_params=np.zeros(n_p),
    )
    vqe_energies.append(res.energy)
    print(f"r={r:.2f} A: E_HF={hf_energies[-1]:.4f}, E_VQE={res.energy:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(bond_lengths, hf_energies, '--', label='Hartree-Fock', color='gray')
plt.plot(bond_lengths, vqe_energies, 'o-', label='VQE-UCCSD', markersize=4)
plt.xlabel('Bond length (Angstrom)')
plt.ylabel('Energy (Hartree)')
plt.title('H2 Dissociation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('h2_dissociation.png', dpi=150)
plt.show()
```

> **Note:** UCCSD correctly describes the dissociation limit of H2, unlike
> restricted Hartree-Fock which fails qualitatively at large bond lengths.
> This is because UCCSD can capture static correlation through the double
> excitation operator.

## 6. Larger Molecules: LiH

```python
# Lithium hydride (requires more qubits)
lih = Molecule(
    atoms=[('Li', (0, 0, 0)), ('H', (0, 0, 1.6))],
    basis='sto-3g',
)

print(f"LiH: {lih.n_spin_orbitals} qubits, {lih.n_electrons} electrons")

ham_lih = jordan_wigner(lih)
ansatz_lih, n_params_lih = uccsd_ansatz(
    n_qubits=lih.n_spin_orbitals,
    n_electrons=lih.n_electrons,
)

print(f"UCCSD parameters: {n_params_lih}")
print(f"Hamiltonian terms: {len(ham_lih.terms)}")

vqe_lih = VQE(
    n_qubits=lih.n_spin_orbitals,
    hamiltonian=ham_lih,
    circuit_fn=ansatz_lih,
    n_layers=1,
)

result_lih = vqe_lih.optimize(
    optimizer=Adam(lr=0.02),
    n_iterations=300,
    initial_params=np.zeros(n_params_lih),
)

print(f"\nLiH VQE energy: {result_lih.energy:.6f} Ha")
```

> **Warning:** LiH in STO-3G requires 12 spin-orbitals (12 qubits). The
> wavefunction backend handles this easily. For larger molecules, consider
> active space reduction or the MPS backend.

## 7. Active Space Reduction

For larger molecules, select an active space to reduce qubit count:

```python
# BeH2 with active space
beh2 = Molecule(
    atoms=[('Be', (0, 0, 0)), ('H', (0, 0, 1.3)), ('H', (0, 0, -1.3))],
    basis='sto-3g',
)

# Full space would require 14 qubits
# Select active space: 2 electrons in 3 spatial orbitals = 6 qubits
ham_beh2 = jordan_wigner(beh2, active_electrons=2, active_orbitals=3)

print(f"Active space: {ham_beh2.n_qubits} qubits")
print(f"Hamiltonian terms: {len(ham_beh2.terms)}")
```

## 8. Complete Workflow Summary

Here is the complete workflow for a molecular simulation:

```python
from Qforge.chem import Molecule, jordan_wigner, uccsd_ansatz
from Qforge.algo import VQE, Adam
import numpy as np

# 1. Define molecule
mol = Molecule(
    atoms=[('H', (0, 0, 0)), ('H', (0, 0, 0.735))],
    basis='sto-3g',
)

# 2. Get qubit Hamiltonian
hamiltonian = jordan_wigner(mol)

# 3. Build ansatz
ansatz, n_params = uccsd_ansatz(
    n_qubits=mol.n_spin_orbitals,
    n_electrons=mol.n_electrons,
)

# 4. Run VQE
vqe = VQE(
    n_qubits=mol.n_spin_orbitals,
    hamiltonian=hamiltonian,
    circuit_fn=ansatz,
    n_layers=1,
)

result = vqe.optimize(
    optimizer=Adam(lr=0.05),
    n_iterations=200,
    initial_params=np.zeros(n_params),
)

# 5. Report
print(f"Ground-state energy: {result.energy:.6f} Ha")
```

## Summary

| Concept | API |
|---------|-----|
| Define molecule | `Molecule(atoms, basis, charge, multiplicity)` |
| Qubit mapping | `jordan_wigner(molecule)` |
| Active space | `jordan_wigner(mol, active_electrons, active_orbitals)` |
| UCCSD ansatz | `uccsd_ansatz(n_qubits, n_electrons)` |
| VQE optimization | `VQE(n_qubits, hamiltonian, circuit_fn, n_layers)` |

---

Back to [Tutorials Index](../index.md)
