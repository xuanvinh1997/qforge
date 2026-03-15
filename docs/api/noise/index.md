# Qforge.noise

Noise modelling for quantum circuit simulation. Provides quantum noise channels
defined by Kraus operators and a `NoiseModel` that associates channels with
gate operations.

## Usage

```python
from qforge.noise import NoiseModel, Depolarizing, BitFlip, AmplitudeDamping

# Build a noise model
model = NoiseModel()
model.add_all_qubit_quantum_error(Depolarizing(0.01), ['H', 'X', 'Y', 'Z'])
model.add_all_qubit_quantum_error(BitFlip(0.005), ['CNOT'])
model.add_quantum_error(AmplitudeDamping(0.02), ['RX', 'RY'], [0, 1])
```

## Sub-modules

| Module | Description |
|--------|-------------|
| [`channels`](channels.md) | Quantum noise channels (Kraus operator definitions) |
| [`model`](model.md) | `NoiseModel` for attaching channels to gates |

## Exported Names

```python
from qforge.noise import (
    QuantumChannel,
    BitFlip, PhaseFlip, Depolarizing,
    AmplitudeDamping, PhaseDamping, ThermalRelaxation,
    ReadoutError, KrausChannel,
    NoiseModel,
)
```
