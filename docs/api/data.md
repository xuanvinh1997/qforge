# Qforge.data

Quantum data analysis utilities including Pauli-Z expectation values, connected
correlators, entanglement entropy, and classical dataset complexity measures.
Hot-path computations dispatch to C++ when available.

## Usage

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT
from qforge.data import (
    PauliZExpectation, ConnectedCorrelator, EntanglementEntropy,
    intrinsic_dim_from_cov, spectral_complex_kernel, kolmogorov_complex,
)
import numpy as np

# Quantum analysis
wf = Qubit(4)
H(wf, 0); CNOT(wf, 0, 1)
pz = PauliZExpectation(wf)
print(pz.one_body(0))       # <Z_0>
print(pz.two_body(0, 1))    # <Z_0 Z_1>

ee = EntanglementEntropy(wf)
print(ee.von_neumann([0]))   # S(rho_A)

# Classical dataset analysis
dataset = np.random.randn(100, 5)
print(intrinsic_dim_from_cov(dataset))
print(kolmogorov_complex(dataset))
```

## Classes

### `PauliZExpectation(wavefunction)`

Compute 1- through 4-body Pauli-Z expectation values from a wavefunction.
Uses C++ acceleration when available.

**Methods:**

- `one_body(i)` -- `<Z_i>`
- `two_body(i, j)` -- `<Z_i Z_j>`
- `three_body(i, j, k)` -- `<Z_i Z_j Z_k>`
- `four_body(i, j, k, l)` -- `<Z_i Z_j Z_k Z_l>`

---

### `ConnectedCorrelator(wavefunction)`

Compute connected correlation functions from Pauli-Z expectation values.

---

### `EntanglementEntropy(wavefunction)`

Compute entanglement entropy of a subsystem via the reduced density matrix.

**Methods:**

- `von_neumann(subsystem)` -- Von Neumann entropy of the specified subsystem.
- `renyi(subsystem, alpha=2)` -- Renyi entropy of order `alpha`.

## Functions

### `intrinsic_dim_from_cov(dataset)`

Estimate the intrinsic dimension of a dataset using the spectrum of its
covariance matrix: `(sum(lambda))^2 / sum(lambda^2)`.

**Parameters:** `dataset` -- 2D array (samples x features).

**Returns:** `float` -- Estimated intrinsic dimension.

---

### `spectral_complex_kernel(kernel, lambda_K=0.1)`

Compute the spectral complexity of a kernel matrix:
`Tr(K @ inv(K + lambda_K * I))`.

**Parameters:**

- `kernel` -- Square kernel matrix.
- `lambda_K` -- Regularization parameter (default 0.1).

**Returns:** `float` -- Spectral complexity.

---

### `kolmogorov_complex(dataset)`

Estimate the Kolmogorov complexity of a dataset via compression ratios
(zlib, gzip, lzma, bz2).

**Parameters:** `dataset` -- Numpy array.

**Returns:** `dict` with keys `'zlib'`, `'gzip'`, `'lzma'`, `'bz2'`, `'original_bytes'`, `'best_bytes'`.

## Full API

::: qforge.data
    options:
      show_source: false
