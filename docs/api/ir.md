# Qforge.ir

Circuit intermediate representation providing `GateOp`, `MeasureOp`, `ConditionalOp`,
`ClassicalRegister`, and `Circuit` so that quantum programs can be built, inspected,
transformed, serialized, and replayed on any backend.

## Usage

```python
from Qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp, ClassicalRegister

# Build a circuit programmatically
circ = Circuit(n_qubits=2)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('CNOT', (0, 1)))
circ.append(MeasureOp(qubit=0, classical_bit=0))
circ.append(ConditionalOp(
    classical_bit=0,
    expected_value=1,
    op=GateOp('X', (1,)),
))

print(circ)
print(f"Depth: {circ.depth}")
```

## Classes

### `GateOp`

A single gate instruction in a circuit (frozen dataclass).

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Gate name (`'H'`, `'CNOT'`, `'RX'`, etc.) |
| `qubits` | `tuple[int, ...]` | Qubit indices the gate acts on |
| `params` | `tuple[float, ...]` | Rotation angles / numeric parameters |
| `matrix` | `np.ndarray \| None` | Explicit unitary (for `QubitUnitary`) |
| `is_adjoint` | `bool` | If True, apply the conjugate transpose |
| `controls` | `tuple[int, ...]` | Extra control qubits (multi-controlled) |

---

### `MeasureOp`

Mid-circuit measurement instruction (frozen dataclass).

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `qubit` | `int` | Qubit to measure |
| `classical_bit` | `int` | Classical register index for the result |

---

### `ConditionalOp`

Classically-conditioned gate operation (frozen dataclass).

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `classical_bit` | `int` | Classical bit to condition on |
| `expected_value` | `int` | Value that triggers the operation (0 or 1) |
| `op` | `GateOp` | Gate to apply when the condition is met |

---

### `ClassicalRegister`

Classical register for storing measurement results.

---

### `Circuit`

Ordered sequence of quantum instructions with methods for inspection
and manipulation.

**Key methods:**

- `append(op)` -- Add an operation.
- `extend(ops)` -- Add multiple operations.
- `bind_parameters(param_dict)` -- Resolve symbolic parameters.
- `adjoint()` -- Return the adjoint (reversed, conjugate-transposed) circuit.
- `depth` -- Circuit depth (property).
- `n_ops` -- Number of operations (property).

## Full API

::: Qforge.ir
    options:
      show_source: false
