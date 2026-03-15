# Qforge.noise.model

The `NoiseModel` class maps gate operations to noise channels that are applied
after each matching gate during circuit execution.

## Usage

```python
from qforge.noise import NoiseModel, Depolarizing, ReadoutError

model = NoiseModel()

# Apply depolarizing noise after all single-qubit gates
model.add_all_qubit_quantum_error(Depolarizing(0.01), ['H', 'X', 'Y', 'Z'])

# Apply stronger noise to CNOT on specific qubits
model.add_quantum_error(Depolarizing(0.05), ['CNOT'], [0, 1])

# Add readout error to qubit 0
model.add_readout_error(ReadoutError(0.02, 0.03), [0])

# Query errors for a gate
errors = model.get_errors('H', (0,))
```

## Class

### `NoiseModel`

```python
NoiseModel()
```

**Methods:**

- `add_all_qubit_quantum_error(channel, gate_names)` -- Register a channel for all qubits.
- `add_quantum_error(channel, gate_names, qubits)` -- Register a channel for specific qubits.
- `add_readout_error(error, qubits)` -- Register a `ReadoutError` for specific qubits.
- `get_errors(gate_name, qubits)` -- Return list of channels matching the gate and qubits.
- `get_readout_error(qubit)` -- Return the `ReadoutError` for a qubit (or `None`).

## Full API

::: qforge.noise.model
    options:
      show_source: false
