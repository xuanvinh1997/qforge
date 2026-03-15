# Qforge.serialization

Circuit serialization to JSON and compact binary formats. Enables saving,
loading, and transmitting quantum circuits.

## Usage

```python
from qforge.ir import Circuit, GateOp
from qforge.serialization import (
    circuit_to_json, circuit_from_json,
    circuit_to_binary, circuit_from_binary,
)

circ = Circuit(n_qubits=2)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('CNOT', (0, 1)))

# JSON round-trip
json_str = circuit_to_json(circ)
circ2 = circuit_from_json(json_str)

# Binary round-trip
data = circuit_to_binary(circ)
circ3 = circuit_from_binary(data)
```

## Functions

### `circuit_to_json(circuit) -> str`

Serialize a `Circuit` to a JSON string. Gate parameters, qubit indices, and
optional unitary matrices are preserved.

---

### `circuit_from_json(json_str) -> Circuit`

Deserialize a `Circuit` from a JSON string produced by `circuit_to_json`.

---

### `circuit_to_binary(circuit) -> bytes`

Serialize a `Circuit` to a compact binary format. More space-efficient than
JSON for large circuits.

---

### `circuit_from_binary(data) -> Circuit`

Deserialize a `Circuit` from binary data produced by `circuit_to_binary`.

## Full API

::: qforge.serialization
    options:
      show_source: false
