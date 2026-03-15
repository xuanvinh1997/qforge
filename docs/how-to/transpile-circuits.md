# How-To: Transpile and Optimize Circuits

Qforge's transpiler converts circuits into equivalent but optimized forms.
This is useful for reducing gate count, adapting to hardware constraints, and
improving simulation performance.

## DAGCircuit: The Intermediate Representation

The transpiler works on a Directed Acyclic Graph (DAG) representation of the
circuit:

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT, RZ, X, S, T
from Qforge.transpiler import DAGCircuit

# Build a circuit
qc = Qubit(n_qubits=3)
H(qc, target=0)
CNOT(qc, control=0, target=1)
X(qc, target=1)
X(qc, target=1)  # Redundant: X*X = I
RZ(qc, target=0, theta=0.3)
RZ(qc, target=0, theta=0.5)  # Can merge: RZ(0.3)*RZ(0.5) = RZ(0.8)
CNOT(qc, control=0, target=2)

# Convert to DAG
dag = DAGCircuit.from_circuit(qc)
print(f"DAG nodes: {dag.n_nodes}")
print(f"DAG depth: {dag.depth()}")
```

## Optimization Passes

### CancelInverses

Removes adjacent gate pairs that cancel (e.g., X-X, H-H, CNOT-CNOT):

```python
from Qforge.transpiler import CancelInverses

optimized_dag = CancelInverses().run(dag)
print(f"Before: {dag.n_nodes} gates")
print(f"After:  {optimized_dag.n_nodes} gates")
```

### Optimize1qRotations

Merges consecutive single-qubit rotation gates:

```python
from Qforge.transpiler import Optimize1qRotations

optimized_dag = Optimize1qRotations().run(dag)
# RZ(0.3) + RZ(0.5) -> RZ(0.8)
```

### Decompose

Breaks complex gates into a basis gate set:

```python
from Qforge.transpiler import Decompose

# Decompose into {H, CNOT, RZ} basis
decomposed_dag = Decompose(basis_gates=['H', 'CNOT', 'RZ']).run(dag)
```

> **Note:** The `Decompose` pass converts gates like SWAP, CCNOT, CRZ, etc.
> into sequences of basis gates. This is essential for mapping circuits to
> hardware that supports only a limited gate set.

## PassManager: Chaining Passes

Combine multiple passes into a pipeline:

```python
from Qforge.transpiler import PassManager, CancelInverses, Optimize1qRotations, Decompose

# Create a pass manager with multiple optimization passes
pm = PassManager([
    Decompose(basis_gates=['H', 'CNOT', 'RZ']),
    CancelInverses(),
    Optimize1qRotations(),
    CancelInverses(),  # Run again after rotation merging
])

# Apply to a circuit
qc = Qubit(n_qubits=3)
H(qc, target=0)
S(qc, target=0)
T(qc, target=0)
CNOT(qc, control=0, target=1)
X(qc, target=1)
X(qc, target=1)

dag = DAGCircuit.from_circuit(qc)
optimized_dag = pm.run(dag)

# Convert back to circuit
qc_optimized = optimized_dag.to_circuit()

print(f"Original gates:  {dag.n_nodes}")
print(f"Optimized gates: {optimized_dag.n_nodes}")
print(f"Original depth:  {dag.depth()}")
print(f"Optimized depth: {optimized_dag.depth()}")
```

## Repeated Optimization

Some passes benefit from multiple iterations:

```python
pm = PassManager([
    Decompose(basis_gates=['H', 'CNOT', 'RZ']),
    CancelInverses(),
    Optimize1qRotations(),
], repeat=3)  # Run the full pipeline 3 times

optimized_dag = pm.run(dag)
```

## Practical Example: VQE Circuit Optimization

Optimize a hardware-efficient ansatz before running VQE:

```python
from Qforge.circuit import Qubit
from Qforge.gates import RY, RZ, CNOT
from Qforge.transpiler import DAGCircuit, PassManager, CancelInverses, Optimize1qRotations
import numpy as np

def build_ansatz(params):
    qc = Qubit(n_qubits=4)
    idx = 0
    for layer in range(3):
        for q in range(4):
            RY(qc, target=q, theta=params[idx])
            RZ(qc, target=q, theta=params[idx + 1])
            idx += 2
        for q in range(3):
            CNOT(qc, control=q, target=q + 1)
    return qc

# Check gate count before and after optimization
params = np.random.uniform(-np.pi, np.pi, size=24)
qc = build_ansatz(params)

dag = DAGCircuit.from_circuit(qc)
print(f"Before: {dag.n_nodes} gates, depth {dag.depth()}")

pm = PassManager([CancelInverses(), Optimize1qRotations()])
optimized_dag = pm.run(dag)
print(f"After:  {optimized_dag.n_nodes} gates, depth {optimized_dag.depth()}")
```

## Inspecting the DAG

```python
dag = DAGCircuit.from_circuit(qc)

# List all gates in topological order
for node in dag.topological_order():
    print(f"  {node.gate_name} on qubits {node.qubits}, params={node.params}")

# Get the gate count by type
print("\nGate counts:")
for gate_name, count in dag.gate_counts().items():
    print(f"  {gate_name}: {count}")
```

## Summary

| Component | API |
|-----------|-----|
| DAG from circuit | `DAGCircuit.from_circuit(qc)` |
| DAG to circuit | `dag.to_circuit()` |
| Cancel inverses | `CancelInverses().run(dag)` |
| Merge rotations | `Optimize1qRotations().run(dag)` |
| Decompose gates | `Decompose(basis_gates=[...]).run(dag)` |
| Pipeline | `PassManager([pass1, pass2, ...]).run(dag)` |
| DAG info | `dag.n_nodes`, `dag.depth()`, `dag.gate_counts()` |
