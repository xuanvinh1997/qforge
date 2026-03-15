# Tutorial 10: Matrix Product States for Large-Scale Simulation

The wavefunction backend stores all 2^n amplitudes, limiting simulations to
roughly 25 qubits. The Matrix Product State (MPS) backend decomposes the state
into a chain of tensors, enabling simulation of 50+ qubits for circuits with
bounded entanglement.

## 1. Creating an MPS

```python
from qforge.mps import MatrixProductState

# 50-qubit MPS with bond dimension 64
mps = MatrixProductState(n_qubits=50, max_bond_dim=64)
```

The `max_bond_dim` parameter (often called chi) controls the trade-off between
accuracy and memory:

| Bond Dim | Memory per site | Max entanglement (ebits) | Use case |
|----------|----------------|--------------------------|----------|
| 1        | O(d)           | 0 (product states)       | Classical states |
| 16       | O(16^2 * d)    | 4                        | Low-entanglement circuits |
| 64       | O(64^2 * d)    | 6                        | Moderate entanglement |
| 256      | O(256^2 * d)   | 8                        | High-accuracy simulation |

> **Note:** Memory scales as O(n * chi^2 * d) where d=2 for qubits, compared
> to O(2^n) for the wavefunction backend. For n=50 and chi=64, this is roughly
> 400 KB vs 8 PB.

## 2. Applying Gates

MPS uses the same gate API as the wavefunction backend:

```python
from qforge.gates import H, CNOT, RX, RY, RZ, SWAP

# GHZ-like state preparation
H(mps, target=0)
for i in range(49):
    CNOT(mps, control=i, target=i + 1)

# The bond dimension grows with entanglement
# For a GHZ state, exact bond dim is 2
```

Two-qubit gates on non-adjacent qubits are handled by SWAP routing internally:

```python
# This works even though qubits 0 and 49 are far apart
# MPS will insert SWAPs to make them adjacent
CNOT(mps, control=0, target=49)
```

> **Warning:** Long-range two-qubit gates require O(distance) SWAP operations,
> which can increase bond dimension and reduce fidelity. For best results,
> design circuits with nearest-neighbor connectivity.

## 3. Measurement

```python
from qforge.measurement import measure_all, measure_one, pauli_expectation
from qforge.algo import Hamiltonian

# Measure all qubits (collapses the state)
mps_copy = MatrixProductState(n_qubits=50, max_bond_dim=64)
H(mps_copy, target=0)
for i in range(49):
    CNOT(mps_copy, control=i, target=i + 1)

result = measure_all(mps_copy)
print(f"Measured: {result}")  # Should be all 0s or all 1s (GHZ)

# Measure a single qubit
mps2 = MatrixProductState(n_qubits=50, max_bond_dim=64)
H(mps2, target=0)
for i in range(49):
    CNOT(mps2, control=i, target=i + 1)

bit = measure_one(mps2, target=25)
print(f"Qubit 25: {bit}")

# Expectation values
obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 0), ('Z', 49)]])
mps3 = MatrixProductState(n_qubits=50, max_bond_dim=64)
H(mps3, target=0)
for i in range(49):
    CNOT(mps3, control=i, target=i + 1)

exp_val = pauli_expectation(mps3, obs)
print(f"<Z0 Z49> = {exp_val:.4f}")  # Should be ~1.0 for GHZ
```

## 4. Entanglement Entropy

The MPS representation gives direct access to entanglement structure:

```python
import numpy as np
import matplotlib.pyplot as plt

mps = MatrixProductState(n_qubits=20, max_bond_dim=64)

# Create a random circuit with nearest-neighbor gates
np.random.seed(42)
depth = 10
for layer in range(depth):
    for i in range(20):
        RY(mps, target=i, theta=np.random.uniform(0, 2 * np.pi))
    for i in range(0, 19, 2 if layer % 2 == 0 else 1):
        if i + 1 < 20:
            CNOT(mps, control=i, target=i + 1)

# Compute entanglement entropy across each bond
entropies = []
for bond in range(19):
    S = mps.entanglement_entropy(bond)
    entropies.append(S)

plt.figure(figsize=(8, 4))
plt.plot(range(19), entropies, 'o-')
plt.xlabel('Bond index')
plt.ylabel('Entanglement entropy (ebits)')
plt.title('Entanglement profile across the MPS chain')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mps_entropy.png', dpi=150)
plt.show()
```

> **Tip:** If the entanglement entropy at any bond saturates at log2(chi),
> the bond dimension is too small and results may be inaccurate. Increase
> `max_bond_dim` until entropies are well below the saturation limit.

## 5. Bond Dimension Convergence

Verify that your bond dimension is sufficient by running the same circuit at
multiple chi values:

```python
from qforge.measurement import pauli_expectation
from qforge.algo import Hamiltonian

obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 10)]])

def build_circuit(chi):
    mps = MatrixProductState(n_qubits=20, max_bond_dim=chi)
    np.random.seed(42)
    for layer in range(5):
        for i in range(20):
            RY(mps, target=i, theta=np.random.uniform(0, 2 * np.pi))
        for i in range(19):
            CNOT(mps, control=i, target=i + 1)
    return mps

chis = [4, 8, 16, 32, 64, 128]
expectations = []
for chi in chis:
    mps = build_circuit(chi)
    exp = pauli_expectation(mps, obs)
    expectations.append(exp)
    print(f"chi={chi:4d}: <Z_10> = {exp:.6f}")

# Check convergence
for i in range(1, len(chis)):
    diff = abs(expectations[i] - expectations[i - 1])
    print(f"chi {chis[i-1]}->{chis[i]}: delta = {diff:.2e}")
```

## 6. DMRG for Ground-State Problems

The Density Matrix Renormalization Group (DMRG) finds ground states of 1D
Hamiltonians directly in MPS form, without building a circuit:

```python
from qforge.dmrg import DMRG

# Heisenberg model: H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
dmrg = DMRG.heisenberg(
    n_sites=20,
    J=1.0,
    max_bond_dim=64,
    boundary='open',
)

energy, psi = dmrg.run(n_sweeps=10)
print(f"Ground-state energy: {energy:.6f}")
print(f"Energy per site:     {energy / 20:.6f}")
```

### DMRG Convergence

```python
# Track energy convergence across sweeps
dmrg = DMRG.heisenberg(n_sites=40, max_bond_dim=128)
energy, psi = dmrg.run(n_sweeps=20, return_history=True)

plt.figure(figsize=(8, 4))
plt.semilogy(
    range(len(dmrg.energy_history) - 1),
    [abs(dmrg.energy_history[i+1] - dmrg.energy_history[i])
     for i in range(len(dmrg.energy_history) - 1)],
    'o-'
)
plt.xlabel('Sweep')
plt.ylabel('|Delta E|')
plt.title('DMRG Energy Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Custom Hamiltonians with DMRG

```python
# Transverse-field Ising model: H = -J sum Z_i Z_{i+1} - h sum X_i
dmrg_ising = DMRG(
    n_sites=30,
    hamiltonian_terms={
        'ZZ': -1.0,   # Nearest-neighbor ZZ coupling
        'X': -0.5,    # Transverse field
    },
    max_bond_dim=64,
)

energy_ising, psi_ising = dmrg_ising.run(n_sweeps=15)
print(f"Ising ground-state energy: {energy_ising:.6f}")
```

## 7. Entanglement Entropy from DMRG

```python
# Compute entanglement entropy of the DMRG ground state
entropies = []
for bond in range(39):
    S = psi.entanglement_entropy(bond)
    entropies.append(S)

plt.figure(figsize=(8, 4))
plt.plot(range(39), entropies, 'o-', markersize=3)
plt.xlabel('Bond index')
plt.ylabel('Entanglement entropy')
plt.title('Heisenberg ground state (n=40, chi=128)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

> **Note:** For gapped 1D systems, entanglement entropy follows an area law
> (bounded constant). For critical systems (e.g., Heisenberg), it grows
> logarithmically: S ~ (c/3) * log(L) where c is the central charge.

## 8. Performance Tips

1. **Nearest-neighbor circuits** -- MPS is most efficient when two-qubit gates
   act on adjacent qubits. Avoid long-range CNOT gates.

2. **Bond dimension selection** -- Start with chi=16, double until expectation
   values converge. For area-law states, chi=64-128 is usually sufficient.

3. **DMRG vs circuit MPS** -- For ground-state problems, DMRG is vastly more
   efficient than VQE with an MPS backend. Use DMRG when possible.

4. **Canonical form** -- The MPS is kept in mixed-canonical form automatically.
   This ensures numerical stability during gate application and measurement.

## Summary

| Concept | API |
|---------|-----|
| Create MPS | `MatrixProductState(n_qubits, max_bond_dim)` |
| Gates | Same as wavefunction: `H(mps, ...)`, `CNOT(mps, ...)` |
| Entropy | `mps.entanglement_entropy(bond)` |
| DMRG (Heisenberg) | `DMRG.heisenberg(n_sites, max_bond_dim=...)` |
| DMRG (custom) | `DMRG(n_sites, hamiltonian_terms=..., max_bond_dim=...)` |
| Run DMRG | `energy, psi = dmrg.run(n_sweeps=10)` |

---

Next: [Tutorial 11: Quantum Chemistry](11-chemistry.md)
