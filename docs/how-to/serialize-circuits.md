# How-To: Serialize and Export Circuits

Qforge supports multiple serialization formats for saving, sharing, and
interoperating with other quantum frameworks.

## JSON Serialization

Human-readable format, ideal for debugging and version control:

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT, RZ
from qforge.serialization import circuit_to_json, circuit_from_json

# Build a circuit
qc = Qubit(n_qubits=3)
H(qc, target=0)
CNOT(qc, control=0, target=1)
RZ(qc, target=2, theta=0.5)
CNOT(qc, control=1, target=2)

# Serialize to JSON string
json_str = circuit_to_json(qc)
print(json_str)
```

Output:

```json
{
  "n_qubits": 3,
  "gates": [
    {"name": "H", "qubits": [0], "params": []},
    {"name": "CNOT", "qubits": [0, 1], "params": []},
    {"name": "RZ", "qubits": [2], "params": [0.5]},
    {"name": "CNOT", "qubits": [1, 2], "params": []}
  ]
}
```

### Save to and Load from File

```python
# Save to file
json_str = circuit_to_json(qc)
with open('my_circuit.json', 'w') as f:
    f.write(json_str)

# Load from file
with open('my_circuit.json', 'r') as f:
    json_str = f.read()
qc_loaded = circuit_from_json(json_str)
```

### Deserialize

```python
qc_restored = circuit_from_json(json_str)

# Verify: the restored circuit produces the same state
import numpy as np
np.testing.assert_allclose(
    qc.wavefunction.amplitude,
    qc_restored.wavefunction.amplitude,
    atol=1e-12,
)
print("Circuits match!")
```

## Binary Serialization

Compact binary format, ideal for storage and network transfer:

```python
from qforge.serialization import circuit_to_binary, circuit_from_binary

# Serialize to bytes
binary_data = circuit_to_binary(qc)
print(f"Binary size: {len(binary_data)} bytes")

# Save to file
with open('my_circuit.qfbin', 'wb') as f:
    f.write(binary_data)

# Load from file
with open('my_circuit.qfbin', 'rb') as f:
    binary_data = f.read()
qc_restored = circuit_from_binary(binary_data)
```

> **Tip:** Binary format is typically 5-10x smaller than JSON for large
> circuits. Use it for archival storage or when transmitting circuits over
> a network.

## OpenQASM 2.0 Export

Export circuits in the widely supported OpenQASM 2.0 format:

```python
from qforge.qasm import circuit_to_qasm2

qasm_str = circuit_to_qasm2(qc)
print(qasm_str)
```

Output:

```
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];

h q[0];
cx q[0], q[1];
rz(0.5) q[2];
cx q[1], q[2];
```

### Save to File

```python
qasm_str = circuit_to_qasm2(qc)
with open('my_circuit.qasm', 'w') as f:
    f.write(qasm_str)
```

## OpenQASM 2.0 Import

Load circuits from OpenQASM 2.0 files:

```python
from qforge.qasm import qasm2_to_circuit

qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];
"""

qc_imported = qasm2_to_circuit(qasm_str)
print(f"Imported circuit: {qc_imported.n_qubits} qubits")
```

### Load from File

```python
with open('external_circuit.qasm', 'r') as f:
    qasm_str = f.read()
qc_imported = qasm2_to_circuit(qasm_str)
```

> **Note:** OpenQASM import supports the standard `qelib1.inc` gate library.
> Custom gates defined in the QASM file are not automatically mapped to Qforge
> custom gates -- they are decomposed into basis gates during import.

## Interoperability Examples

### Export for IBM Qiskit

```python
from qforge.qasm import circuit_to_qasm2

qasm_str = circuit_to_qasm2(qc)

# In Qiskit:
# from qiskit import QuantumCircuit
# qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
```

### Import from Qiskit

```python
from qforge.qasm import qasm2_to_circuit

# In Qiskit:
# qasm_str = qiskit_circuit.qasm()

# Then in Qforge:
# qc = qasm2_to_circuit(qasm_str)
```

### Export for Cirq or other frameworks

Any framework that supports OpenQASM 2.0 can import qforge circuits:

```python
qasm_str = circuit_to_qasm2(qc)
# Use the QASM string with any compatible framework
```

## Round-Trip Verification

Verify that serialization preserves the circuit:

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT, RX, RY, RZ, Phase
from qforge.serialization import circuit_to_json, circuit_from_json
from qforge.qasm import circuit_to_qasm2, qasm2_to_circuit
import numpy as np

# Build a circuit with various gate types
qc = Qubit(n_qubits=3)
H(qc, target=0)
RX(qc, target=1, theta=1.23)
RY(qc, target=2, theta=0.456)
CNOT(qc, control=0, target=1)
RZ(qc, target=0, theta=0.789)
Phase(qc, target=2, theta=np.pi / 3)
CNOT(qc, control=1, target=2)

original_state = qc.wavefunction.amplitude.copy()

# JSON round-trip
json_str = circuit_to_json(qc)
qc_json = circuit_from_json(json_str)
np.testing.assert_allclose(original_state, qc_json.wavefunction.amplitude, atol=1e-12)
print("JSON round-trip: OK")

# QASM round-trip
qasm_str = circuit_to_qasm2(qc)
qc_qasm = qasm2_to_circuit(qasm_str)
np.testing.assert_allclose(original_state, qc_qasm.wavefunction.amplitude, atol=1e-10)
print("QASM round-trip: OK")
```

> **Warning:** OpenQASM 2.0 has limited numerical precision for gate
> parameters (typically 10-15 significant digits in the text representation).
> For exact round-trips, prefer JSON or binary formats.

## Summary

| Format | API | Size | Human-Readable | Precision |
|--------|-----|------|----------------|-----------|
| JSON | `circuit_to_json` / `circuit_from_json` | Medium | Yes | Exact |
| Binary | `circuit_to_binary` / `circuit_from_binary` | Small | No | Exact |
| OpenQASM 2.0 | `circuit_to_qasm2` / `qasm2_to_circuit` | Medium | Yes | ~15 digits |
