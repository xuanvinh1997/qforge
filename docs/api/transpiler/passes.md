# Qforge.transpiler.passes

Transpiler passes for circuit optimization and rewriting. Each pass inherits
from `TranspilerPass` and implements a `run(dag)` method.

## Usage

```python
from Qforge.transpiler import PassManager, CancelInverses, Optimize1qRotations, BasisTranslator

pm = PassManager([
    CancelInverses(),
    Optimize1qRotations(),
    BasisTranslator(basis_gates=['H', 'CNOT', 'RZ', 'RX']),
])
optimized_circuit = pm.run(circuit)
```

## Base Class

### `TranspilerPass` (ABC)

Abstract base for all transpiler passes.

**Methods:**

- `run(dag) -> DAGCircuit` -- Transform the DAG in-place or return a new DAG.

---

### `PassManager`

```python
PassManager(passes: list[TranspilerPass])
```

Runs a sequence of passes on a circuit.

**Methods:**

- `run(circuit) -> Circuit` -- Apply all passes and return the optimized circuit.
- `append(pass_)` -- Add a pass to the pipeline.

## Optimization Passes

### `Decompose(gate_names=None)`

Decompose multi-qubit gates into a basis set. If `gate_names` is `None`,
decomposes all non-basis gates.

---

### `CancelInverses()`

Cancel adjacent pairs of self-inverse gates (e.g., H-H, X-X, CNOT-CNOT).

---

### `Optimize1qRotations()`

Merge consecutive single-qubit rotations (RX, RY, RZ) into a single
rotation using the ZYZ decomposition.

---

### `CommutationAnalysis()`

Analyze gate commutation relations and reorder gates to expose further
cancellation and optimization opportunities.

---

### `BasisTranslator(basis_gates)`

Translate gates into a target basis gate set.

```python
BasisTranslator(basis_gates=['H', 'CNOT', 'RZ', 'RX'])
```

**Parameters:**

- `basis_gates` -- List of allowed gate names in the target basis.

## Full API

::: Qforge.transpiler.passes
    options:
      show_source: false
