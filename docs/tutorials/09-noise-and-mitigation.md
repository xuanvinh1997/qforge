# Tutorial 9: Noise Simulation and Error Mitigation

Real quantum hardware suffers from noise. This tutorial shows how to simulate
noise channels, build noise models, use the density matrix backend, and apply
error mitigation techniques.

## 1. Noise Channels

Qforge provides several standard noise channels. Each is applied after a gate
operation to simulate hardware imperfections.

### Bit-Flip Channel

Flips the qubit state with probability p:

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT, X
from Qforge.noise import BitFlip
from Qforge import DensityMatrix

# Create a density matrix backend (required for noise)
qc = DensityMatrix(n_qubits=1)

X(qc, target=0)
BitFlip(qc, target=0, p=0.1)

print(qc.density_matrix)
# Diagonal should show ~0.9 for |1><1| and ~0.1 for |0><0|
```

### Depolarizing Channel

Replaces the qubit state with the maximally mixed state with probability p:

```python
from Qforge.noise import Depolarizing

qc = DensityMatrix(n_qubits=1)
H(qc, target=0)
Depolarizing(qc, target=0, p=0.05)

print(qc.density_matrix)
```

### Amplitude Damping

Models energy relaxation (T1 decay):

```python
from Qforge.noise import AmplitudeDamping

qc = DensityMatrix(n_qubits=1)
X(qc, target=0)  # Prepare |1>
AmplitudeDamping(qc, target=0, gamma=0.1)

print(qc.density_matrix)
# Shows partial decay from |1> toward |0>
```

> **Note:** Amplitude damping with gamma=1.0 fully resets the qubit to |0>.
> This models complete T1 relaxation.

## 2. Building a Noise Model

A `NoiseModel` attaches noise channels to specific gates automatically:

```python
from Qforge.noise import NoiseModel, Depolarizing, BitFlip, AmplitudeDamping

noise_model = NoiseModel()

# Add depolarizing noise after every single-qubit gate
noise_model.add_all_qubit_quantum_error(
    Depolarizing(p=0.01),
    gate_names=['H', 'RX', 'RY', 'RZ', 'X', 'Y', 'Z']
)

# Add stronger depolarizing noise after two-qubit gates
noise_model.add_all_qubit_quantum_error(
    Depolarizing(p=0.03),
    gate_names=['CNOT', 'CRZ']
)

# Add amplitude damping to specific qubits (modeling T1 variation)
noise_model.add_quantum_error(
    AmplitudeDamping(gamma=0.02),
    gate_names=['H', 'X'],
    qubits=[0]
)
```

## 3. Running a Noisy Simulation

Use the noise model with the DensityMatrix backend:

```python
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT
from Qforge.measurement import measure_all

# Create a noisy Bell state
qc = DensityMatrix(n_qubits=2, noise_model=noise_model)
H(qc, target=0)
CNOT(qc, control=0, target=1)

# The density matrix shows mixed state (noise reduces off-diagonal elements)
print("Density matrix diagonal:", qc.density_matrix.diagonal().real)

# Sample from the noisy state
counts = {}
for _ in range(1000):
    qc_copy = DensityMatrix(n_qubits=2, noise_model=noise_model)
    H(qc_copy, target=0)
    CNOT(qc_copy, control=0, target=1)
    result = measure_all(qc_copy)
    counts[result] = counts.get(result, 0) + 1

print("\nNoisy Bell state samples:")
for bitstring, count in sorted(counts.items()):
    print(f"  |{bitstring}>: {count}")
```

> **Tip:** For noiseless simulation, use `Qubit` (wavefunction backend). The
> `DensityMatrix` backend is slower (O(4^n) vs O(2^n)) but necessary for
> mixed-state simulation.

## 4. Comparing Ideal vs Noisy Circuits

```python
from Qforge.circuit import Qubit
from Qforge import DensityMatrix
from Qforge.gates import H, CNOT, RZ
from Qforge.measurement import pauli_expectation
from Qforge.algo import Hamiltonian
import numpy as np

obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 0), ('Z', 1)]])

# Ideal
qc_ideal = Qubit(n_qubits=2)
H(qc_ideal, target=0)
CNOT(qc_ideal, control=0, target=1)
exp_ideal = pauli_expectation(qc_ideal, obs)

# Noisy
qc_noisy = DensityMatrix(n_qubits=2, noise_model=noise_model)
H(qc_noisy, target=0)
CNOT(qc_noisy, control=0, target=1)
exp_noisy = pauli_expectation(qc_noisy, obs)

print(f"Ideal <ZZ>: {exp_ideal:.4f}")
print(f"Noisy <ZZ>: {exp_noisy:.4f}")
print(f"Error:      {abs(exp_ideal - exp_noisy):.4f}")
```

## 5. Zero-Noise Extrapolation (ZNE)

ZNE runs the circuit at multiple noise levels and extrapolates to the
zero-noise limit:

```python
from Qforge.mitigation import zero_noise_extrapolation

def circuit_fn():
    """Returns a circuit for expectation measurement."""
    qc = DensityMatrix(n_qubits=2, noise_model=noise_model)
    H(qc, target=0)
    CNOT(qc, control=0, target=1)
    return qc

mitigated_value = zero_noise_extrapolation(
    circuit_fn=circuit_fn,
    observable=obs,
    noise_model=noise_model,
    scale_factors=[1, 2, 3],       # Run at 1x, 2x, 3x noise
    extrapolation='linear',         # 'linear', 'quadratic', or 'exponential'
)

print(f"Ideal:     {exp_ideal:.4f}")
print(f"Noisy:     {exp_noisy:.4f}")
print(f"ZNE:       {mitigated_value:.4f}")
print(f"ZNE error: {abs(exp_ideal - mitigated_value):.4f}")
```

> **Note:** ZNE amplifies the noise by inserting identity-equivalent gate
> pairs (e.g., CNOT-CNOT). The `scale_factors` parameter controls the noise
> amplification levels. Richardson extrapolation (quadratic) often gives better
> results than linear.

## 6. Probabilistic Error Cancellation (PEC)

PEC uses a quasi-probability decomposition to cancel noise exactly (in the
limit of infinite samples):

```python
from Qforge.mitigation import probabilistic_error_cancellation

mitigated_pec = probabilistic_error_cancellation(
    circuit_fn=circuit_fn,
    observable=obs,
    noise_model=noise_model,
    n_samples=5000,
)

print(f"Ideal:      {exp_ideal:.4f}")
print(f"Noisy:      {exp_noisy:.4f}")
print(f"PEC:        {mitigated_pec:.4f}")
print(f"PEC error:  {abs(exp_ideal - mitigated_pec):.4f}")
```

> **Warning:** PEC has a sampling overhead that grows exponentially with
> circuit depth and noise rate. It is most effective for short circuits with
> well-characterized noise. The `n_samples` parameter controls the trade-off
> between accuracy and runtime.

## 7. Mitigation in a VQE Workflow

Combine noise simulation with VQE to study the effect of mitigation:

```python
from Qforge.algo import VQE, Adam, Hamiltonian
from Qforge.mitigation import zero_noise_extrapolation
import numpy as np

h2_ham = Hamiltonian(
    coeffs=[-0.8105, 0.1721, -0.2257, 0.1721, 0.1235, 0.1235],
    terms=[
        [],
        [('Z', 0)],
        [('Z', 1)],
        [('Z', 0), ('Z', 1)],
        [('X', 0), ('X', 1)],
        [('Y', 0), ('Y', 1)],
    ]
)

def noisy_ansatz(params):
    qc = DensityMatrix(n_qubits=2, noise_model=noise_model)
    from Qforge.gates import RY, CNOT
    RY(qc, target=0, theta=params[0])
    RY(qc, target=1, theta=params[1])
    CNOT(qc, control=0, target=1)
    RY(qc, target=0, theta=params[2])
    RY(qc, target=1, theta=params[3])
    return qc

def mitigated_cost(params):
    def circuit_fn():
        return noisy_ansatz(params)
    return zero_noise_extrapolation(
        circuit_fn=circuit_fn,
        observable=h2_ham,
        noise_model=noise_model,
        scale_factors=[1, 2, 3],
        extrapolation='quadratic',
    )

# Compare noisy vs mitigated optimization
params = np.array([0.5, -0.3, 1.2, 0.8])

noisy_energy = pauli_expectation(noisy_ansatz(params), h2_ham)
mitigated_energy = mitigated_cost(params)

print(f"Noisy energy:     {noisy_energy:.4f}")
print(f"Mitigated energy: {mitigated_energy:.4f}")
print(f"Exact energy:     -1.1373")
```

## 8. Custom Noise Channels

Build a custom noise channel from Kraus operators:

```python
from Qforge.noise import NoiseModel
import numpy as np

# Phase damping channel (T2 process)
gamma = 0.05
K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
K1 = np.array([[0, 0], [0, np.sqrt(gamma)]])

noise_model_custom = NoiseModel()
noise_model_custom.add_all_qubit_kraus_error(
    kraus_ops=[K0, K1],
    gate_names=['H', 'RZ']
)
```

## Summary

| Concept | API |
|---------|-----|
| Bit flip | `BitFlip(qc, target, p)` |
| Depolarizing | `Depolarizing(qc, target, p)` |
| Amplitude damping | `AmplitudeDamping(qc, target, gamma)` |
| Noise model | `NoiseModel()`, `.add_all_qubit_quantum_error(...)` |
| Density matrix | `DensityMatrix(n_qubits, noise_model=...)` |
| ZNE | `zero_noise_extrapolation(circuit_fn, observable, ...)` |
| PEC | `probabilistic_error_cancellation(circuit_fn, observable, ...)` |

---

Next: [Tutorial 10: MPS for Large-Scale Simulation](10-mps-large-scale.md)
