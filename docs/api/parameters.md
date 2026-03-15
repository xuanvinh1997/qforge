# Qforge.parameters

Symbolic parameters for parameterized quantum circuits. Circuits can be built
with named, unbound parameters and later resolved via `Circuit.bind_parameters`.

## Usage

```python
from Qforge.parameters import Parameter, ParameterVector
from Qforge.ir import Circuit, GateOp

# Single parameter
theta = Parameter('theta')
print(theta)              # Parameter('theta')
print(theta.is_bound())   # False

theta_bound = theta.bind(0.5)
print(float(theta_bound)) # 0.5

# Parameter vector
pv = ParameterVector('theta', 4)
print(len(pv))    # 4
print(pv[0])      # Parameter('theta_0')
```

## Classes

### `Parameter`

A named symbolic parameter.

```python
Parameter(name: str, value: float | None = None)
```

**Methods:**

- `is_bound()` -- Returns `True` if the parameter has a numeric value.
- `bind(value)` -- Return a new `Parameter` with the given value.
- `float(param)` -- Convert to float (raises `ValueError` if unbound).

---

### `ParameterVector`

A named collection of `Parameter` objects, supporting indexing and iteration.

```python
ParameterVector(prefix: str, length: int)
```

**Usage:**

```python
pv = ParameterVector('theta', 4)
for p in pv:
    print(p.name)  # theta_0, theta_1, theta_2, theta_3
```

## Full API

::: Qforge.parameters
    options:
      show_source: false
