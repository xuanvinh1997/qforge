# Installation

## Quick Install (pip)

```bash
pip install qforge
```

This installs the pure-Python version. All gates, measurements, and circuit features work out of the box.

## From Source with C++ Engine

The C++ acceleration engine delivers 100--200x speedup over pure Python and is strongly recommended for circuits with 10+ qubits.

### Prerequisites

- Python 3.9+
- A C++17 compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- `pybind11 >= 2.10`

### Build Steps

```bash
git clone https://github.com/yourorg/Qforge.git
cd qforge
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -e .
```

The `pip install -e .` command compiles the C++ extension (`_qforge_core`) automatically. The build uses `-O3 -march=native -ffast-math` for maximum performance.

### Verify Installation

```python
# Check that the C++ engine loaded
from qforge import _HAS_CPP
print(f"C++ engine available: {_HAS_CPP}")

# Full check
from qforge._qforge_core import StateVector
print("C++ StateVector loaded successfully")
```

If `_HAS_CPP` is `False`, Qforge falls back to pure Python automatically. All functionality remains available, just slower for large circuits.

### Verify with a Quick Test

```bash
python -c "
from qforge.circuit import Qubit
from qforge.gates import H, CNOT
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)
print('Bell state probabilities:', wf.probabilities())
"
```

Expected output:

```
Bell state probabilities: [0.5 0.  0.  0.5]
```

## Optional Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Required -- core dependency |
| `pybind11 >= 2.10` | Required at build time for C++ engine |
| `scikit-learn` | Quantum kernel methods and ML extensions |
| `pandas` | Data analysis utilities |
| `matplotlib` | Plotting in notebooks and tutorials |

Install all optional dependencies:

```bash
pip install scikit-learn pandas matplotlib
```

## GPU Backends (Experimental)

Qforge supports CUDA and Metal backends for GPU-accelerated simulation. These are selected automatically when available, or you can request them explicitly:

```python
wf = Qubit(16, backend='cuda')   # NVIDIA GPU
wf = Qubit(16, backend='metal')  # Apple Silicon GPU
wf = Qubit(16, backend='cpu')    # C++ CPU engine
wf = Qubit(16, backend='python') # Pure Python fallback
```

The default `backend='auto'` picks the fastest available backend in order: CUDA > Metal > C++ > Python.

> **Note:** GPU backends require additional build steps and hardware. See the repository README for GPU-specific build instructions.

## Running Tests

```bash
python test.py           # Basic Bell state smoke test
python benchmark.py      # Full correctness + performance suite
```

## Performance Reference

With the C++ engine (depth=10, H+CNOT chain):

| Qubits | Python | C++ | Speedup |
|--------|--------|-----|---------|
| 8 | 7 ms | 0.07 ms | 107x |
| 10 | 35 ms | 0.2 ms | 172x |
| 12 | 163 ms | 0.8 ms | 205x |
| 14 | 735 ms | 4 ms | 196x |
| 20 | impractical | 370 ms | -- |
| 22 | impractical | 1.66 s | -- |
