# How-To: Compute Gradients with the Parameter-Shift Rule

The parameter-shift rule computes exact analytic gradients of quantum circuits
with respect to gate parameters. This is the foundation of variational quantum
algorithms in Qforge.

## Basic Usage

```python
from qforge.algo import parameter_shift
from qforge.circuit import Qubit
from qforge.gates import RY, CNOT
from qforge.measurement import pauli_expectation
from qforge.algo import Hamiltonian
import numpy as np

# Define an observable
obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 0)]])

# Define a cost function that takes parameters and returns a scalar
def cost_fn(params):
    qc = Qubit(n_qubits=2)
    RY(qc, target=0, theta=params[0])
    RY(qc, target=1, theta=params[1])
    CNOT(qc, control=0, target=1)
    RY(qc, target=0, theta=params[2])
    return pauli_expectation(qc, obs)

params = np.array([0.5, 1.2, -0.3])

# Compute gradients
grads = parameter_shift(cost_fn, params)
print(f"Gradients: {grads}")
```

## How It Works

For a gate `exp(-i * theta/2 * G)` where G is a Pauli generator, the
parameter-shift rule gives:

```
df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2
```

This requires exactly 2 circuit evaluations per parameter. For n parameters,
the total cost is 2n circuit evaluations.

## Custom Shift Values

Some gates require different shift values:

```python
grads = parameter_shift(
    cost_fn,
    params,
    shift=np.pi / 2,  # Default for standard rotation gates
)
```

> **Note:** The default shift of pi/2 is correct for all standard rotation
> gates (RX, RY, RZ, CRX, CRY, CRZ, Phase, CP). Only custom gates with
> non-standard generators may need a different shift.

## Manual Gradient Descent

```python
params = np.random.uniform(-np.pi, np.pi, size=3)
lr = 0.1

history = []
for step in range(100):
    energy = cost_fn(params)
    history.append(energy)
    grads = parameter_shift(cost_fn, params)
    params = params - lr * grads

    if step % 20 == 0:
        print(f"Step {step:3d}: energy = {energy:.6f}, |grad| = {np.linalg.norm(grads):.6f}")
```

## Using Built-In Optimizers

For convenience, use the built-in optimizers which call `parameter_shift`
internally:

```python
from qforge.algo import Adam, SPSA

# Adam optimizer (uses parameter_shift internally)
from qforge.algo import VQE

vqe = VQE(n_qubits=2, hamiltonian=obs, circuit_fn=my_ansatz, n_layers=1)
result = vqe.optimize(optimizer=Adam(lr=0.05), n_iterations=200)
```

### Adam

Adaptive learning rate with momentum. Best for noiseless simulation:

```python
optimizer = Adam(
    lr=0.05,       # Learning rate
    beta1=0.9,     # First moment decay
    beta2=0.999,   # Second moment decay
    eps=1e-8,      # Numerical stability
)
```

### SPSA

Stochastic approximation using only 2 evaluations per step (regardless of
parameter count). Best for noisy or shot-based simulation:

```python
optimizer = SPSA(
    a=0.1,    # Step size scaling
    c=0.1,    # Perturbation size
    A=10,     # Stability constant
    alpha=0.602,
    gamma=0.101,
)
```

> **Tip:** SPSA is preferred when circuit evaluations are expensive (many
> parameters or shot-based simulation) because it uses only 2 evaluations
> per step regardless of parameter count, compared to 2n for parameter-shift.

## Parallel Gradient Computation

For circuits with many parameters, compute gradients in parallel:

```python
from qforge.algo import parallel_parameter_shift

# Uses all available CPU cores (or GPU batching)
grads = parallel_parameter_shift(
    cost_fn,
    params,
    n_workers=4,  # Number of parallel workers
)
```

> **Warning:** `parallel_parameter_shift` spawns multiple Python processes.
> Each process creates its own circuit, so memory usage scales linearly with
> `n_workers`. For GPU backends, batching is handled automatically and
> `n_workers` controls CPU-side parallelism only.

## Gradient of Multi-Term Hamiltonians

The parameter-shift rule works with any cost function that returns a scalar.
For Hamiltonians with multiple Pauli terms, `pauli_expectation` sums the
contributions internally:

```python
from qforge.algo import Hamiltonian

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

def h2_cost(params):
    qc = Qubit(n_qubits=2)
    RY(qc, target=0, theta=params[0])
    RY(qc, target=1, theta=params[1])
    CNOT(qc, control=0, target=1)
    RY(qc, target=0, theta=params[2])
    RY(qc, target=1, theta=params[3])
    return pauli_expectation(qc, h2_ham)

params = np.zeros(4)
grads = parameter_shift(h2_cost, params)
print(f"H2 gradients: {grads}")
```

## Numerical Gradient Verification

Verify parameter-shift gradients against finite differences:

```python
def numerical_gradient(cost_fn, params, eps=1e-5):
    grads = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grads[i] = (cost_fn(params_plus) - cost_fn(params_minus)) / (2 * eps)
    return grads

params = np.array([0.5, 1.2, -0.3, 0.8])
grads_ps = parameter_shift(h2_cost, params)
grads_fd = numerical_gradient(h2_cost, params)

print("Parameter-shift vs finite difference:")
for i in range(len(params)):
    print(f"  param {i}: PS={grads_ps[i]:.6f}, FD={grads_fd[i]:.6f}, "
          f"diff={abs(grads_ps[i] - grads_fd[i]):.2e}")
```

> **Note:** Parameter-shift gradients are exact (up to floating-point
> precision), while finite differences have O(eps^2) truncation error.
> The parameter-shift rule is also compatible with shot-based estimation.

## Summary

| Function | Cost | Best For |
|----------|------|----------|
| `parameter_shift(cost_fn, params)` | 2n evals | Exact gradients, small n |
| `parallel_parameter_shift(cost_fn, params, n_workers)` | 2n evals (parallel) | Large parameter counts |
| `Adam(lr=...)` | 2n evals/step | Noiseless optimization |
| `SPSA(a=..., c=...)` | 2 evals/step | Noisy or shot-based |
