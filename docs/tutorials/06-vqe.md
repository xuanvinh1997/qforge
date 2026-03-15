# Tutorial 6: Variational Quantum Eigensolver (VQE)

The Variational Quantum Eigensolver finds the ground-state energy of a quantum
Hamiltonian by optimizing a parameterized circuit. This tutorial walks through
a complete VQE workflow for the hydrogen molecule (H2).

## Prerequisites

```python
pip install qforge numpy matplotlib
```

## 1. Define the H2 Hamiltonian

The H2 molecular Hamiltonian in the STO-3G basis, after qubit mapping, can be
expressed as a sum of Pauli terms:

```python
from qforge.algo import Hamiltonian

# H2 at equilibrium bond length (0.735 Angstrom)
# Coefficients from Jordan-Wigner transformation
h2_hamiltonian = Hamiltonian(
    coeffs=[
        -0.8105,   # Identity (nuclear repulsion + constant)
         0.1721,   # Z0
        -0.2257,   # Z1
         0.1721,   # Z0 Z1
         0.1235,   # X0 X1
         0.1235,   # Y0 Y1
    ],
    terms=[
        [],                          # Identity
        [('Z', 0)],                  # Z on qubit 0
        [('Z', 1)],                  # Z on qubit 1
        [('Z', 0), ('Z', 1)],       # ZZ interaction
        [('X', 0), ('X', 1)],       # XX interaction
        [('Y', 0), ('Y', 1)],       # YY interaction
    ]
)
```

## 2. Create a Parameterized Ansatz

A hardware-efficient ansatz alternates single-qubit rotations with entangling
layers:

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT, RX, RY, RZ
from qforge import Parameter, ParameterVector

def h2_ansatz(params):
    """Two-qubit ansatz for H2."""
    qc = Qubit(n_qubits=2)

    # Layer 1: single-qubit rotations
    RY(qc, target=0, theta=params[0])
    RY(qc, target=1, theta=params[1])

    # Entangling gate
    CNOT(qc, control=0, target=1)

    # Layer 2: single-qubit rotations
    RY(qc, target=0, theta=params[2])
    RY(qc, target=1, theta=params[3])

    return qc
```

You can also use the built-in hardware-efficient ansatz:

```python
from qforge.algo import hardware_efficient_ansatz

# Generates a parameterized ansatz automatically
ansatz_fn = hardware_efficient_ansatz(n_qubits=2, n_layers=2, rotation_gates=['RY', 'RZ'])
```

## 3. Set Up and Run VQE

```python
from qforge.algo import VQE, Adam
import numpy as np

vqe = VQE(
    n_qubits=2,
    hamiltonian=h2_hamiltonian,
    circuit_fn=h2_ansatz,
    n_layers=2,
)

# Optimize with Adam
result = vqe.optimize(
    optimizer=Adam(lr=0.05),
    n_iterations=200,
    initial_params=np.random.uniform(-np.pi, np.pi, size=4),
)

print(f"Ground-state energy: {result.energy:.6f} Ha")
print(f"Exact energy:        -1.1373 Ha")
print(f"Parameters:          {result.optimal_params}")
```

> **Note:** The exact ground-state energy of H2 at equilibrium bond length in
> this basis is approximately -1.1373 Hartree. VQE should converge to within
> chemical accuracy (1.6 mHa).

## 4. Plot the Convergence

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(result.energy_history, linewidth=2)
plt.axhline(y=-1.1373, color='r', linestyle='--', label='Exact energy')
plt.xlabel('Iteration')
plt.ylabel('Energy (Hartree)')
plt.title('VQE Convergence for H2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vqe_convergence.png', dpi=150)
plt.show()
```

## 5. Using the SPSA Optimizer

For noisy or shot-based simulations, the Simultaneous Perturbation Stochastic
Approximation (SPSA) optimizer is more robust:

```python
from qforge.algo import SPSA

result_spsa = vqe.optimize(
    optimizer=SPSA(a=0.1, c=0.1),
    n_iterations=300,
    initial_params=np.random.uniform(-np.pi, np.pi, size=4),
)

print(f"SPSA energy: {result_spsa.energy:.6f} Ha")
```

> **Tip:** SPSA uses only two function evaluations per iteration regardless of
> the number of parameters, making it efficient for large circuits. Adam uses
> the parameter-shift rule and scales linearly with parameter count, but
> typically converges faster in noiseless simulation.

## 6. Computing Gradients Manually

If you want full control over the optimization loop, use `parameter_shift`
directly:

```python
from qforge.algo import parameter_shift
from qforge.measurement import pauli_expectation

def cost_fn(params):
    qc = h2_ansatz(params)
    return pauli_expectation(qc, h2_hamiltonian)

params = np.random.uniform(-np.pi, np.pi, size=4)

# Get exact gradients via parameter-shift rule
grads = parameter_shift(cost_fn, params)
print(f"Gradients: {grads}")

# Manual gradient descent
lr = 0.05
for step in range(200):
    grads = parameter_shift(cost_fn, params)
    params = params - lr * grads
    if step % 20 == 0:
        energy = cost_fn(params)
        print(f"Step {step:3d}: energy = {energy:.6f}")
```

## 7. Potential Energy Surface

Scan the bond length to map the full potential energy surface:

```python
from qforge.chem import Molecule, jordan_wigner

bond_lengths = np.linspace(0.3, 2.5, 30)
vqe_energies = []

for r in bond_lengths:
    mol = Molecule([('H', (0, 0, 0)), ('H', (0, 0, r))], basis='sto-3g')
    ham = jordan_wigner(mol)

    vqe = VQE(n_qubits=2, hamiltonian=ham, circuit_fn=h2_ansatz, n_layers=2)
    result = vqe.optimize(
        optimizer=Adam(lr=0.05),
        n_iterations=150,
        initial_params=np.random.uniform(-np.pi, np.pi, size=4),
    )
    vqe_energies.append(result.energy)

plt.figure(figsize=(8, 5))
plt.plot(bond_lengths, vqe_energies, 'o-', label='VQE')
plt.xlabel('Bond length (Angstrom)')
plt.ylabel('Energy (Hartree)')
plt.title('H2 Potential Energy Surface')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Summary

| Concept | API |
|---------|-----|
| Hamiltonian | `Hamiltonian(coeffs, terms)` |
| VQE solver | `VQE(n_qubits, hamiltonian, circuit_fn, n_layers)` |
| Optimizers | `Adam(lr=...)`, `SPSA(a=..., c=...)` |
| Gradient | `parameter_shift(cost_fn, params)` |
| Built-in ansatz | `hardware_efficient_ansatz(n_qubits, n_layers, ...)` |

> **Warning:** VQE is sensitive to initial parameters. If convergence stalls,
> try different random seeds or increase the number of layers in the ansatz.

---

Next: [Tutorial 7: QAOA for Max-Cut](07-qaoa.md)
