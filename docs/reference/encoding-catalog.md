# Encoding Catalog

Complete reference for all data encoding strategies in `Qforge.encodings`.
Each encoding maps classical data to a quantum state (Wavefunction).

## Summary Table

| Encoding | Signature | # Qubits | Features/Qubit | Entanglement |
|----------|-----------|----------|----------------|--------------|
| `amplitude_encode` | `(sample)` | ceil(log2(N)) | Exponential | None |
| `qubit_encode` | `(sample)` | N | 1 | None |
| `dense_encode` | `(sample)` | N/2 | 2 | None |
| `unit_encode` | `(sample)` | N | 1 | None |
| `yz_cx_encode` | `(sample, ent)` | N/2 | 2 | CNOT (linear/circular) |
| `high_dim_encode` | `(sample)` | N | 1 | None |
| `hzy_cz_encode` | `(sample, ent)` | N | 1 | CZ (linear/circular) |
| `chebyshev_encode` | `(sample)` | N | 1 | None |
| `param_z_feature_map_encode` | `(sample)` | N | 1 | None |
| `separable_rx_encode` | `(sample)` | N | 1 | None |
| `hardware_efficient_embed_encode` | `(sample)` | N | 1 | CNOT |
| `z_feature_map_encode` | `(sample)` | N | 1 | None |
| `zz_feature_map_encode` | `(sample)` | N | 1 | ZZ interaction |

## Detailed Descriptions

### `amplitude_encode(sample)`

Encodes a classical vector of length N directly into the amplitudes of a
ceil(log2(N))-qubit state. The input is L2-normalized. This is the most
qubit-efficient encoding but requires exponentially deep state-preparation
circuits on real hardware.

**Input:** Real-valued array of length up to 2^n.

**Qubits:** ceil(log2(len(sample)))

---

### `qubit_encode(sample)`

Maps each feature x_i to a single qubit via the rotation
cos(x_i/2)|0> - sin(x_i/2)|1>. The total state is a tensor product
(no entanglement).

**Input:** Array of rotation angles (radians), one per qubit.

**Qubits:** len(sample)

---

### `dense_encode(sample)`

Dense angle encoding: each pair of features (x_i, x_{i+n}) encodes into
one qubit using combined RY and RZ rotations, effectively doubling the
information per qubit compared to qubit encoding.

**Input:** Array of length 2 * n_qubits.

**Qubits:** len(sample) / 2

---

### `unit_encode(sample)`

Encodes features using RY gates. Each feature maps to an RY rotation angle.

**Input:** Real-valued feature array.

**Qubits:** len(sample)

---

### `yz_cx_encode(sample, ent='linear')`

Entangled encoding with RY and RZ rotations followed by CNOT entanglement
layers. Supports linear and circular entanglement topologies.

**Input:** Feature array (length = 2 * n_qubits).

**Qubits:** len(sample) / 2

**Entanglement:** `'linear'` (chain) or `'circular'` (ring)

---

### `high_dim_encode(sample)`

High-dimensional encoding using alternating Hadamard and RZ gates.
Suitable for kernel methods where features map to high-dimensional Hilbert
space.

**Input:** Feature array.

**Qubits:** len(sample)

---

### `hzy_cz_encode(sample, ent='linear')`

Encoding with H, RZ, RY rotations per qubit followed by CZ entanglement
layers. Provides a different entanglement structure from yz_cx_encode.

**Input:** Feature array.

**Qubits:** len(sample)

**Entanglement:** `'linear'` or `'circular'`

---

### `chebyshev_encode(sample)`

Chebyshev polynomial encoding using arccos-based rotations. Maps features
in [-1, 1] to rotation angles via arccos, enabling polynomial feature maps
in the computational basis.

**Input:** Feature array with values in [-1, 1].

**Qubits:** len(sample)

---

### `param_z_feature_map_encode(sample)`

Parameterized Z feature map using RZ rotations with nonlinear feature
transformations.

**Input:** Feature array.

**Qubits:** len(sample)

---

### `separable_rx_encode(sample)`

Separable encoding using only RX gates (no entanglement). Each feature
maps to an RX rotation on a dedicated qubit.

**Input:** Feature array.

**Qubits:** len(sample)

---

### `hardware_efficient_embed_encode(sample)`

Hardware-efficient embedding using alternating single-qubit rotation layers
and CNOT entanglement layers. Designed for near-term quantum devices.

**Input:** Feature array.

**Qubits:** len(sample)

---

### `z_feature_map_encode(sample)`

First-order Pauli-Z feature map. Applies Hadamard gates followed by
parameterized RZ rotations encoding the input features.

**Input:** Feature array.

**Qubits:** len(sample)

---

### `zz_feature_map_encode(sample)`

Second-order Pauli-ZZ feature map with entanglement. Extends the Z feature
map by adding ZZ interaction terms between pairs of qubits, capturing
two-body correlations in the data.

**Input:** Feature array.

**Qubits:** len(sample)
