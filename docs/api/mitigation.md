# Qforge.mitigation

Error mitigation techniques for improving the accuracy of noisy quantum circuit
results without requiring additional qubits.

## Usage

```python
from qforge.mitigation import (
    zero_noise_extrapolation, fold_circuit,
    probabilistic_error_cancellation, PEC,
    calibrate_readout, correct_readout,
)
import numpy as np

# Zero-noise extrapolation
mitigated_value = zero_noise_extrapolation(
    executor_fn=run_circuit,
    scale_factors=[1, 2, 3],
    extrapolation='linear',
)

# Readout error correction
cal_matrix = calibrate_readout(backend, qubits=[0, 1, 2])
corrected_counts = correct_readout(raw_counts, cal_matrix)
```

## Zero-Noise Extrapolation (ZNE)

### `zero_noise_extrapolation(executor_fn, scale_factors, extrapolation='linear')`

Estimate the zero-noise expectation value by running the circuit at multiple
noise scale factors and extrapolating to the zero-noise limit.

**Parameters:**

- `executor_fn` -- `callable(scale_factor) -> float` that returns the expectation value at a given noise scale.
- `scale_factors` -- List of noise scale factors (e.g. `[1, 2, 3]`).
- `extrapolation` -- Extrapolation method: `'linear'`, `'polynomial'`, or `'exponential'`.

**Returns:** `float` -- Extrapolated zero-noise value.

---

### `fold_circuit(circuit, scale_factor)`

Create a noise-scaled circuit by folding (appending circuit and its inverse).

**Parameters:**

- `circuit` -- Original `Circuit`.
- `scale_factor` -- Integer scale factor (1 = no folding, 3 = one fold, etc.).

**Returns:** Folded `Circuit`.

## Probabilistic Error Cancellation (PEC)

### `probabilistic_error_cancellation(circuit, noise_model, n_samples=1000)`

Run PEC by sampling quasi-probability decompositions of the ideal circuit.

**Parameters:**

- `circuit` -- Ideal `Circuit`.
- `noise_model` -- `NoiseModel` describing the hardware noise.
- `n_samples` -- Number of Monte Carlo samples.

**Returns:** Mitigated expectation value.

---

### `PEC`

Class-based interface for probabilistic error cancellation.

## Readout Correction

### `calibrate_readout(backend, qubits)`

Build a calibration (confusion) matrix by preparing and measuring all
computational basis states.

**Returns:** Calibration matrix (numpy array).

---

### `correct_readout(raw_counts, calibration_matrix)`

Apply inverse calibration to raw measurement counts.

**Returns:** Corrected counts dictionary.

## Full API

::: qforge.mitigation
    options:
      show_source: false
