# Tutorial 7: QAOA for Max-Cut

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid
quantum-classical algorithm for combinatorial optimization. This tutorial
solves the Max-Cut problem on a small graph.

## Background

Given a graph G = (V, E), Max-Cut partitions the vertices into two sets to
maximize the number of edges crossing the partition. QAOA encodes this as a
cost Hamiltonian and uses alternating mixer/cost layers to approximate the
optimal solution.

## 1. Define the Graph

```python
import numpy as np

# A simple 4-node graph
#   0 --- 1
#   |     |
#   3 --- 2
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
n_qubits = 4
```

## 2. Set Up QAOA

```python
from Qforge.algo import QAOA

qaoa = QAOA(
    n_qubits=n_qubits,
    edges=edges,
    p_layers=2,   # Number of QAOA layers (depth)
)
```

Under the hood, QAOA constructs:

- **Cost Hamiltonian:** `C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)`
- **Mixer Hamiltonian:** `B = sum_i X_i`
- **Circuit:** alternating `exp(-i * gamma * C)` and `exp(-i * beta * B)` layers

## 3. Optimize the Parameters

```python
from Qforge.algo import Adam

result = qaoa.optimize(
    optimizer=Adam(lr=0.1),
    n_iterations=100,
    initial_params=np.random.uniform(0, np.pi, size=2 * 2),  # 2 gammas + 2 betas
)

print(f"Optimal cost: {result.energy:.4f}")
print(f"Optimal parameters: {result.optimal_params}")
```

> **Note:** For p=2 layers, there are 4 parameters: gamma_1, gamma_2, beta_1,
> beta_2. The maximum cut for this graph is 4 (all edges cut).

## 4. Extract the Solution

After optimization, sample the circuit to find the most probable bitstrings:

```python
from Qforge.circuit import Qubit
from Qforge.measurement import measure_all

# Build the optimized circuit
qc = qaoa.build_circuit(result.optimal_params)

# Sample multiple times
counts = {}
for _ in range(1000):
    qc_copy = qaoa.build_circuit(result.optimal_params)
    bitstring = measure_all(qc_copy)
    counts[bitstring] = counts.get(bitstring, 0) + 1

# Sort by frequency
sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
print("\nTop measurement results:")
for bitstring, count in sorted_counts[:5]:
    cut_value = sum(1 for (i, j) in edges if bitstring[i] != bitstring[j])
    print(f"  {bitstring}: count={count}, cut={cut_value}")
```

## 5. Visualize the Result

```python
import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(result.energy_history, linewidth=2)
plt.axhline(y=-4.0, color='r', linestyle='--', label='Max cut = 4')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('QAOA Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot measurement distribution
plt.subplot(1, 2, 2)
top_states = sorted_counts[:8]
labels = [s for s, _ in top_states]
values = [c for _, c in top_states]
plt.bar(labels, values)
plt.xlabel('Bitstring')
plt.ylabel('Counts')
plt.title('Measurement Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('qaoa_maxcut.png', dpi=150)
plt.show()
```

## 6. Increasing Circuit Depth

Higher values of `p_layers` give better approximation ratios at the cost of
more parameters and deeper circuits:

```python
for p in [1, 2, 3, 4]:
    qaoa_p = QAOA(n_qubits=n_qubits, edges=edges, p_layers=p)
    result_p = qaoa_p.optimize(
        optimizer=Adam(lr=0.1),
        n_iterations=150,
        initial_params=np.random.uniform(0, np.pi, size=2 * p),
    )
    ratio = abs(result_p.energy) / 4.0  # max cut = 4
    print(f"p={p}: energy={result_p.energy:.4f}, approximation ratio={ratio:.4f}")
```

> **Tip:** For Max-Cut, QAOA with p=1 guarantees an approximation ratio of at
> least 0.6924 on 3-regular graphs. Increasing p improves this ratio but makes
> classical optimization harder.

## 7. Weighted Max-Cut

For weighted graphs, pass edge weights:

```python
weighted_edges = [(0, 1, 2.0), (1, 2, 1.5), (2, 3, 3.0), (3, 0, 1.0)]

qaoa_w = QAOA(
    n_qubits=4,
    edges=weighted_edges,
    p_layers=3,
)

result_w = qaoa_w.optimize(
    optimizer=Adam(lr=0.08),
    n_iterations=200,
    initial_params=np.random.uniform(0, np.pi, size=6),
)

print(f"Weighted max-cut cost: {result_w.energy:.4f}")
```

## 8. Larger Graph Example

```python
# Random 8-node graph
import random
random.seed(42)

n = 8
edges_large = []
for i in range(n):
    for j in range(i + 1, n):
        if random.random() < 0.4:
            edges_large.append((i, j))

print(f"Graph: {n} nodes, {len(edges_large)} edges")

qaoa_large = QAOA(n_qubits=n, edges=edges_large, p_layers=3)
result_large = qaoa_large.optimize(
    optimizer=Adam(lr=0.05),
    n_iterations=300,
    initial_params=np.random.uniform(0, np.pi, size=6),
)

print(f"Best cost: {result_large.energy:.4f}")
```

> **Warning:** QAOA circuit depth grows linearly with the number of edges.
> For dense graphs with many qubits, consider using the MPS backend
> (see [Tutorial 10](10-mps-large-scale.md)) or a GPU backend.

## Summary

| Concept | API |
|---------|-----|
| QAOA setup | `QAOA(n_qubits, edges, p_layers)` |
| Optimization | `qaoa.optimize(optimizer, n_iterations, initial_params)` |
| Build circuit | `qaoa.build_circuit(params)` |
| Weighted graphs | Pass `(i, j, weight)` tuples as edges |

---

Next: [Tutorial 8: Quantum Machine Learning](08-quantum-ml.md)
