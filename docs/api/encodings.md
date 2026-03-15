# Qforge.encodings

Quantum data encoding strategies for mapping classical data into quantum states.
All encodings return a `Wavefunction` and compose existing gates from `Qforge.gates`,
so they automatically benefit from C++ acceleration.

## Usage

```python
from Qforge.encodings import amplitude_encode, qubit_encode, yz_cx_encode
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0])

# Amplitude encoding (log2(n) qubits)
wf = amplitude_encode(data)

# Qubit encoding (one qubit per feature)
wf = qubit_encode(np.array([0.3, 0.7, 1.2]))

# Entangled encoding with YZ rotations and CX entanglement
wf = yz_cx_encode(np.array([0.1, 0.2, 0.3, 0.4]), ent='linear')
```

## Encoding Functions

### `amplitude_encode(sample)`

Encode a classical vector directly into quantum amplitudes. The sample is
L2-normalized and mapped to the amplitude vector of a `ceil(log2(len(sample)))`-qubit
state. Uses the fewest qubits but requires exponentially deep state preparation
on hardware.

**Parameters:** `sample` -- Real-valued array of length up to 2^n.

**Returns:** `Wavefunction` initialized with the encoded amplitudes.

---

### `qubit_encode(sample)`

Encode each feature into one qubit via rotation angles. Feature `x_i` maps to
`cos(x_i/2)|0> - sin(x_i/2)|1>` on qubit *i*. The total state is a tensor product.

**Parameters:** `sample` -- Array of rotation angles (radians), one per qubit.

**Returns:** `Wavefunction` with `len(sample)` qubits.

---

### `dense_encode(sample)`

Dense angle encoding using pairs of features per qubit. Uses `len(sample)/2`
qubits, encoding two features per qubit via combined RY and RZ rotations.

**Parameters:** `sample` -- Array of length `2 * n_qubits`.

**Returns:** `Wavefunction`.

---

### `unit_encode(sample)`

Unit encoding using RY gates for each feature.

**Parameters:** `sample` -- Real-valued feature array.

**Returns:** `Wavefunction`.

---

### `yz_cx_encode(sample, ent='linear')`

Entangled encoding with RY/RZ rotations followed by CNOT entanglement layers.
Supports `'linear'` and `'circular'` entanglement patterns.

**Parameters:**

- `sample` -- Feature array (length = 2 * n_qubits).
- `ent` -- Entanglement pattern: `'linear'` or `'circular'`.

**Returns:** `Wavefunction`.

---

### `high_dim_encode(sample)`

High-dimensional encoding using Hadamard and RZ gates.

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

---

### `hzy_cz_encode(sample, ent='linear')`

Encoding with H, RZ, RY rotations and CZ entanglement layers.

**Parameters:**

- `sample` -- Feature array.
- `ent` -- Entanglement pattern: `'linear'` or `'circular'`.

**Returns:** `Wavefunction`.

---

### `chebyshev_encode(sample)`

Chebyshev encoding using arccos-based rotations for polynomial feature maps.

**Parameters:** `sample` -- Feature array with values in [-1, 1].

**Returns:** `Wavefunction`.

---

### `param_z_feature_map_encode(sample)`

Parameterized Z feature map encoding.

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

---

### `separable_rx_encode(sample)`

Separable encoding using RX gates (no entanglement).

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

---

### `hardware_efficient_embed_encode(sample)`

Hardware-efficient embedding using alternating rotation and entanglement layers.

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

---

### `z_feature_map_encode(sample)`

Z feature map encoding (first-order Pauli-Z expansion).

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

---

### `zz_feature_map_encode(sample)`

ZZ feature map encoding (second-order Pauli-ZZ expansion with entanglement).

**Parameters:** `sample` -- Feature array.

**Returns:** `Wavefunction`.

## Full API

::: Qforge.encodings
    options:
      show_source: false
