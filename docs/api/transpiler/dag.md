# Qforge.transpiler.dag

Directed Acyclic Graph (DAG) representation of quantum circuits. The DAG
encodes data dependencies between gate operations, enabling efficient
analysis and rewriting.

## Usage

```python
from Qforge.transpiler.dag import DAGCircuit, DAGNode
from Qforge.ir import Circuit, GateOp

circ = Circuit(n_qubits=2)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('CNOT', (0, 1)))

dag = DAGCircuit.from_circuit(circ)
print(f"Nodes: {dag.n_nodes}")
print(f"Depth: {dag.depth}")

# Convert back
circ2 = dag.to_circuit()
```

## Classes

### `DAGNode`

A single node in the DAG representing a gate operation.

**Attributes:**

- `op` -- The `GateOp`, `MeasureOp`, or `ConditionalOp`.
- `predecessors` -- Set of predecessor node indices.
- `successors` -- Set of successor node indices.
- `qubits` -- Qubit indices this node acts on.

---

### `DAGCircuit`

DAG representation of a quantum circuit.

**Class methods:**

- `DAGCircuit.from_circuit(circuit)` -- Build a DAG from a `Circuit`.

**Properties:**

- `n_nodes` -- Number of operation nodes.
- `depth` -- Circuit depth (longest path through the DAG).
- `nodes` -- List of `DAGNode` objects.

**Methods:**

- `to_circuit()` -- Convert back to a `Circuit` (topological order).
- `topological_order()` -- Return node indices in topological order.
- `predecessors(node_idx)` -- Return predecessor node indices.
- `successors(node_idx)` -- Return successor node indices.
- `remove_node(node_idx)` -- Remove a node and update edges.
- `substitute_node(node_idx, new_ops)` -- Replace a node with a sequence of operations.

## Full API

::: Qforge.transpiler.dag
    options:
      show_source: false
