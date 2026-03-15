# How-To: Use the DensityMatrix Backend

The `DensityMatrix` backend represents quantum states as density matrices
(rho), enabling simulation of mixed states, noise channels, and open quantum
systems.

## When to Use DensityMatrix

Use `DensityMatrix` instead of `Qubit` when you need:

- **Noise simulation** -- noise channels produce mixed states
- **Partial trace** -- tracing out qubits from an entangled system
- **Fidelity and purity** -- metrics that require the full density matrix
- **Decoherence modeling** -- T1/T2 processes

> **Warning:** The density matrix has 2^n x 2^n complex entries, so memory
> and time scale as O(4^n). This limits practical simulation to about 13
> qubits, compared to ~25 for the wavefunction backend.

## Creating a DensityMatrix

```python
from Qforge import DensityMatrix

# Start in |0...0> state
dm = DensityMatrix(n_qubits=3)
print(f"Shape: {dm.density_matrix.shape}")  # (8, 8)
print(f"Trace: {dm.trace():.1f}")           # 1.0
print(f"Purity: {dm.purity():.1f}")         # 1.0 (pure state)
```

## Applying Gates

Gates work identically to the wavefunction backend:

```python
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT, RX, RY, RZ

dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
CNOT(dm, control=0, target=1)

# Bell state density matrix
print("Bell state rho:")
print(dm.density_matrix.real)
```

## Pure vs Mixed States

### Pure State

A pure state has purity = 1 and can be written as |psi><psi|:

```python
dm_pure = DensityMatrix(n_qubits=1)
H(dm_pure, target=0)

print(f"Purity: {dm_pure.purity():.4f}")  # 1.0
```

### Mixed State (via noise)

```python
from Qforge.noise import Depolarizing

dm_mixed = DensityMatrix(n_qubits=1)
H(dm_mixed, target=0)
Depolarizing(dm_mixed, target=0, p=0.3)

print(f"Purity: {dm_mixed.purity():.4f}")  # < 1.0
```

### Maximally Mixed State

```python
import numpy as np

dm_max = DensityMatrix(n_qubits=2)
# Manually set to maximally mixed state
dm_max.density_matrix = np.eye(4) / 4

print(f"Purity: {dm_max.purity():.4f}")  # 0.25 = 1/d
```

## Partial Trace

Trace out qubits to get the reduced density matrix of a subsystem:

```python
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT

# Create a Bell state
dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
CNOT(dm, control=0, target=1)

# Trace out qubit 1 to get the state of qubit 0
rho_0 = dm.partial_trace(keep_qubits=[0])
print("Reduced density matrix of qubit 0:")
print(rho_0)
# Should be maximally mixed: [[0.5, 0], [0, 0.5]]

print(f"Purity of subsystem: {np.trace(rho_0 @ rho_0).real:.4f}")  # 0.5
```

## Measurement

### Expectation Values

```python
from Qforge.measurement import pauli_expectation
from Qforge.algo import Hamiltonian

dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
CNOT(dm, control=0, target=1)

obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 0), ('Z', 1)]])
exp = pauli_expectation(dm, obs)
print(f"<ZZ> = {exp:.4f}")  # 1.0 for Bell state |00> + |11>
```

### Projective Measurement

```python
from Qforge.measurement import measure_all, measure_one

dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
CNOT(dm, control=0, target=1)

# Measurement collapses to a pure state
bit = measure_one(dm, target=0)
print(f"Qubit 0: {bit}")
print(f"Purity after measurement: {dm.purity():.4f}")  # 1.0 (collapsed)
```

## Fidelity

Compute the fidelity between two states:

```python
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT
from Qforge.noise import Depolarizing

# Ideal Bell state
dm_ideal = DensityMatrix(n_qubits=2)
H(dm_ideal, target=0)
CNOT(dm_ideal, control=0, target=1)

# Noisy Bell state
dm_noisy = DensityMatrix(n_qubits=2)
H(dm_noisy, target=0)
Depolarizing(dm_noisy, target=0, p=0.1)
CNOT(dm_noisy, control=0, target=1)
Depolarizing(dm_noisy, target=1, p=0.1)

fidelity = dm_ideal.fidelity(dm_noisy)
print(f"Fidelity: {fidelity:.4f}")
```

## Von Neumann Entropy

```python
import numpy as np

dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
CNOT(dm, control=0, target=1)

# Entropy of the full system (0 for a pure state)
S_full = dm.von_neumann_entropy()
print(f"Full system entropy: {S_full:.4f}")  # 0.0

# Entropy of subsystem (1 ebit for maximally entangled Bell state)
rho_0 = dm.partial_trace(keep_qubits=[0])
eigenvalues = np.linalg.eigvalsh(rho_0)
eigenvalues = eigenvalues[eigenvalues > 1e-12]
S_sub = -np.sum(eigenvalues * np.log2(eigenvalues))
print(f"Subsystem entropy: {S_sub:.4f}")  # 1.0
```

## Noise Model Integration

```python
from Qforge import DensityMatrix
from Qforge.noise import NoiseModel, Depolarizing, AmplitudeDamping
from Qforge.gates import H, CNOT, RY

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    Depolarizing(p=0.01), gate_names=['H', 'RY']
)
noise_model.add_all_qubit_quantum_error(
    Depolarizing(p=0.03), gate_names=['CNOT']
)

dm = DensityMatrix(n_qubits=3, noise_model=noise_model)
H(dm, target=0)
RY(dm, target=1, theta=0.5)
CNOT(dm, control=0, target=1)
CNOT(dm, control=1, target=2)

print(f"Purity: {dm.purity():.4f}")
print(f"Trace:  {dm.trace():.4f}")
```

## Comparing Wavefunction and DensityMatrix

For noiseless circuits, both backends give the same results:

```python
from Qforge.circuit import Qubit
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT, RY
from Qforge.measurement import pauli_expectation
from Qforge.algo import Hamiltonian
import numpy as np

obs = Hamiltonian(coeffs=[1.0, 0.5], terms=[[('Z', 0)], [('X', 1)]])

# Wavefunction
qc = Qubit(n_qubits=2)
H(qc, target=0)
RY(qc, target=1, theta=0.8)
CNOT(qc, control=0, target=1)
exp_wf = pauli_expectation(qc, obs)

# DensityMatrix
dm = DensityMatrix(n_qubits=2)
H(dm, target=0)
RY(dm, target=1, theta=0.8)
CNOT(dm, control=0, target=1)
exp_dm = pauli_expectation(dm, obs)

print(f"Wavefunction: {exp_wf:.6f}")
print(f"DensityMatrix: {exp_dm:.6f}")
print(f"Match: {np.isclose(exp_wf, exp_dm)}")
```

## Summary

| Operation | API |
|-----------|-----|
| Create | `DensityMatrix(n_qubits)` or `DensityMatrix(n_qubits, noise_model=...)` |
| Apply gates | Same as `Qubit`: `H(dm, ...)`, `CNOT(dm, ...)` |
| Density matrix | `dm.density_matrix` |
| Trace | `dm.trace()` |
| Purity | `dm.purity()` |
| Fidelity | `dm.fidelity(other_dm)` |
| Partial trace | `dm.partial_trace(keep_qubits=[...])` |
| Entropy | `dm.von_neumann_entropy()` |
| Measurement | `measure_all(dm)`, `measure_one(dm, target)` |
