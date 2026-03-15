# Qforge.algo.optimizers

Classical optimizers for variational quantum algorithms. All optimizers share
a common `.step(params, grad)` interface.

## Usage

```python
from Qforge.algo.optimizers import Adam, GradientDescent, SPSA, LBFGS
from Qforge.algo.gradient import parameter_shift
import numpy as np

opt = Adam(lr=0.01)
params = np.random.randn(8)

for i in range(100):
    grad = parameter_shift(cost_fn, params)
    params = opt.step(params, grad)
```

## Classes

### `GradientDescent`

Vanilla gradient-descent optimizer.

```python
GradientDescent(lr=0.1)
```

**Methods:**

- `step(params, grad)` -- Return `params - lr * grad`.

---

### `Adam`

Adam optimizer (Kingma & Ba, 2014) with bias-corrected moment estimates.

```python
Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
```

**Methods:**

- `step(params, grad)` -- Return bias-corrected Adam update.
- `reset()` -- Reset moment estimates for reuse on a new problem.

---

### `SPSA`

Simultaneous Perturbation Stochastic Approximation. Estimates gradients using
only two function evaluations per iteration, regardless of parameter dimension.

```python
SPSA(lr=0.1, perturbation=0.1)
```

**Methods:**

- `step(params, grad)` -- Return updated parameters.

---

### `LBFGS`

Limited-memory BFGS quasi-Newton optimizer. Wraps `scipy.optimize.minimize`
with the L-BFGS-B method.

```python
LBFGS(maxiter=100)
```

**Methods:**

- `minimize(cost_fn, params)` -- Run L-BFGS-B optimization. Returns `(optimal_params, result)`.

## Full API

::: Qforge.algo.optimizers
    options:
      show_source: false
