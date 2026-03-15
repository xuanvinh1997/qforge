# Qforge.chem.ucc

Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz for quantum chemistry.

## Usage

```python
from qforge.chem import uccsd_ansatz, uccsd_n_params
from qforge.circuit import Qubit
import numpy as np

n_electrons = 2
n_orbitals = 4
n_p = uccsd_n_params(n_electrons, n_orbitals)

wf = Qubit(n_orbitals)
params = np.zeros(n_p)
uccsd_ansatz(wf, params, n_electrons=n_electrons, n_orbitals=n_orbitals)
```

## Functions

### `uccsd_ansatz(wf, params, n_electrons, n_orbitals)`

Apply the UCCSD ansatz to a wavefunction. Implements single and double
excitation operators using Trotterized exponentiation.

**Parameters:**

- `wf` -- Wavefunction (modified in-place).
- `params` -- 1-D parameter array of length `uccsd_n_params(n_electrons, n_orbitals)`.
- `n_electrons` -- Number of electrons.
- `n_orbitals` -- Number of spatial orbitals (= number of qubits).

---

### `uccsd_n_params(n_electrons, n_orbitals) -> int`

Return the number of variational parameters for a UCCSD ansatz.

**Parameters:**

- `n_electrons` -- Number of electrons.
- `n_orbitals` -- Number of spatial orbitals.

**Returns:** `int` -- Number of parameters (singles + doubles excitations).

## Full API

::: qforge.chem.ucc
    options:
      show_source: false
