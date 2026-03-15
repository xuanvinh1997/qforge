# Qforge.algo.gradient

Parameter-shift gradient rule for computing exact gradients of variational
quantum circuits. Supports both sequential and parallel (threaded) evaluation.

## Usage

```python
from Qforge.algo.gradient import parameter_shift, parallel_parameter_shift
import numpy as np

def cost_fn(params):
    # Build circuit, apply ansatz, return expectation value
    ...

params = np.array([0.1, 0.5, -0.3])

# Sequential gradient
grad = parameter_shift(cost_fn, params)

# Parallel gradient (multi-threaded)
grad = parallel_parameter_shift(cost_fn, params, max_workers=4)
```

## Functions

### `parameter_shift(cost_fn, params, shift=pi/2)`

Compute the gradient via the parameter-shift rule. For each parameter
theta_i, evaluates `(E(theta+shift) - E(theta-shift)) / (2*sin(shift))`.

A shift of pi/2 gives exact gradients for RX, RY, and RZ gates.

**Parameters:**

- `cost_fn` -- `callable(params) -> float` that evaluates the cost.
- `params` -- 1-D parameter array.
- `shift` -- Shift amount in radians (default `pi/2`).

**Returns:** Gradient array with the same shape as `params`.

---

### `parallel_parameter_shift(cost_fn, params, shift=pi/2, max_workers=None)`

Compute the parameter-shift gradient with concurrent circuit evaluations
using a thread pool. The GIL is released during C++ gate kernels, so
threads overlap in practice.

**Parameters:**

- `cost_fn` -- `callable(params) -> float`. Must be thread-safe (each call constructs its own state).
- `params` -- 1-D parameter array.
- `shift` -- Shift amount in radians (default `pi/2`).
- `max_workers` -- Maximum thread pool size (default: CPU count).

**Returns:** Gradient array with the same shape as `params`.

## Full API

::: Qforge.algo.gradient
    options:
      show_source: false
