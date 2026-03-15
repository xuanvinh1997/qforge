# Qforge.dmrg

Density Matrix Renormalization Group (DMRG) solver for finding ground states of
1D Hamiltonians represented as Matrix Product Operators (MPOs). Supports both
C++ and Python backends.

## Usage

```python
from qforge.dmrg import DMRG
import numpy as np

# Heisenberg chain (4 sites, open boundary conditions)
dmrg = DMRG.heisenberg(n_sites=4, J=1.0, boundary='open')
energy = dmrg.run(n_sweeps=10, chi_max=32)
print(f"Ground-state energy: {energy:.4f}")

# Access the ground-state MPS
mps = dmrg.mps
```

## Class

### `DMRG`

```python
DMRG(mpo, n_sites: int, chi_max: int = 64)
```

**Factory methods:**

- `DMRG.heisenberg(n_sites, J=1.0, boundary='open')` -- Heisenberg XXX model.
- `DMRG.xxz(n_sites, Jxy=1.0, Jz=1.0, boundary='open')` -- XXZ model.

**Key methods:**

- `run(n_sweeps=10, chi_max=64)` -- Run DMRG sweeps and return the ground-state energy.
- `mps` -- The optimized MPS ground state (property).
- `energies` -- List of energies per sweep (property).

## Full API

::: qforge.dmrg
    options:
      show_source: false
      members:
        - DMRG
