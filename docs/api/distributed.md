# qforge.distributed

MPI-distributed state vector backend for simulating 25-40+ qubit circuits across multiple processes or nodes.

## Usage

```python
# Run with: mpirun -np 4 python script.py
from qforge.distributed import DistributedQubit

wf = DistributedQubit(30)
```

The returned `Wavefunction` is compatible with all gate and measurement functions in `qforge.gates` and `qforge.measurement`.

## API

::: qforge.distributed
    options:
      show_source: false
      members:
        - DistributedQubit
