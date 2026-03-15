# Qforge.interfaces

Bridges to external machine learning frameworks (JAX, PyTorch, TensorFlow),
enabling automatic differentiation through quantum circuits. Each bridge is
lazily imported -- only the frameworks installed in your environment are loaded.

## Usage

### JAX

```python
from qforge.interfaces import qnode_jax
import jax
import jax.numpy as jnp

@qnode_jax(n_qubits=2, n_layers=1)
def circuit(params):
    ...

grad_fn = jax.grad(circuit)
params = jnp.zeros(4)
grads = grad_fn(params)
```

### PyTorch

```python
from qforge.interfaces import QNodeFunction
import torch

qfn = QNodeFunction(n_qubits=2, circuit_fn=my_circuit, cost_fn=my_cost)
params = torch.randn(4, requires_grad=True)
loss = qfn(params)
loss.backward()
```

### TensorFlow

```python
from qforge.interfaces import qnode_tf
import tensorflow as tf

@qnode_tf(n_qubits=2, n_layers=1)
def circuit(params):
    ...

with tf.GradientTape() as tape:
    loss = circuit(params)
grads = tape.gradient(loss, params)
```

## Functions and Classes

### `qnode_jax(n_qubits, n_layers=1, **kwargs)`

Decorator that wraps a qforge circuit as a JAX-differentiable function.
Uses the parameter-shift rule under the hood for gradient computation.

**Requires:** `jax` installed.

---

### `QNodeFunction`

PyTorch `autograd.Function` subclass for differentiable quantum circuits.

```python
QNodeFunction(n_qubits, circuit_fn, cost_fn, **kwargs)
```

**Requires:** `torch` installed.

---

### `qnode_tf(n_qubits, n_layers=1, **kwargs)`

Decorator that wraps a qforge circuit as a TensorFlow-differentiable function.

**Requires:** `tensorflow` installed.

## Full API

::: qforge.interfaces
    options:
      show_source: false
