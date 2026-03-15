# Qforge.qasm

OpenQASM 2.0 and 3.0 import/export for round-trip conversion between
qforge circuits and the standard OpenQASM representation.

## Usage

```python
from Qforge.ir import Circuit, GateOp
from Qforge.qasm import circuit_to_qasm2, circuit_to_qasm3, qasm2_to_circuit, qasm3_to_circuit

circ = Circuit(n_qubits=2)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('CNOT', (0, 1)))

# Export
qasm2_str = circuit_to_qasm2(circ)
qasm3_str = circuit_to_qasm3(circ)

# Import
circ_from_2 = qasm2_to_circuit(qasm2_str)
circ_from_3 = qasm3_to_circuit(qasm3_str)
```

## Export Functions

### `circuit_to_qasm2(circuit) -> str`

Convert a qforge `Circuit` to an OpenQASM 2.0 string. Supports all standard
gates in the qforge gate catalog.

---

### `circuit_to_qasm3(circuit) -> str`

Convert a qforge `Circuit` to an OpenQASM 3.0 string. Includes support for
mid-circuit measurement, classical conditionals, and parameterized gates.

## Import Functions

### `qasm2_to_circuit(qasm_str) -> Circuit`

Parse an OpenQASM 2.0 string and return a qforge `Circuit`.

---

### `qasm3_to_circuit(qasm_str) -> Circuit`

Parse an OpenQASM 3.0 string and return a qforge `Circuit`.

## Full API

::: Qforge.qasm
    options:
      show_source: false
