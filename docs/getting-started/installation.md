# Installation

!!! warning "Beta Release"
    Qforge is currently in **beta (v0.1-beta)**. Install from source only — PyPI package is not yet available.

## From Source with C++ Engine

The C++ acceleration engine delivers 100--200x speedup over pure Python and is strongly recommended for circuits with 10+ qubits.

### Prerequisites

- Python 3.9+
- A C++17 compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- `pybind11 >= 2.10`

### Build Steps

```bash
git clone https://github.com/xuanvinh1997/qforge.git
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
| `numpy`, `scipy` | Required -- core dependencies |
| `pybind11 >= 2.10` | Required at build time for C++ engine |
| `matplotlib` | Plotting in notebooks and tutorials |
| `scikit-learn`, `pandas` | Quantum kernel methods and ML extensions |
| `numba` | JIT compilation for select Python kernels |

Install manually:

```bash
pip install matplotlib              # plotting
pip install scikit-learn pandas     # ML extensions
pip install pytest pytest-cov numba # development
```

## GPU Backends

Qforge supports CUDA and Metal backends for GPU-accelerated simulation.

### CUDA (NVIDIA GPU)

**Prerequisites:**

- NVIDIA GPU with compute capability 3.5+
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed
- `nvcc` compiler on PATH, or `CUDA_HOME` / `CUDA_PATH` environment variable set

**Build:**

```bash
git clone https://github.com/xuanvinh1997/qforge.git
cd qforge
QFORGE_CUDA=1 pip install -e .
```

**Verify:**

```python
from qforge import _HAS_CUDA
print(f"CUDA available: {_HAS_CUDA}")
```

!!! tip "CUDA auto-detection"
    The build system searches for `nvcc` in: `$CUDA_HOME/bin`, `$CUDA_PATH/bin`, `/usr/local/cuda/bin`, and `PATH`. On Windows it also scans `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`.

### Metal (Apple Silicon GPU)

**Prerequisites:**

- macOS with Apple Silicon (M1/M2/M3/M4) or AMD GPU
- Xcode Command Line Tools installed (`xcode-select --install`)

**Build:**

```bash
git clone https://github.com/xuanvinh1997/qforge.git
cd qforge
QFORGE_METAL=1 pip install -e .
```

**Verify:**

```python
from qforge import _HAS_METAL
print(f"Metal available: {_HAS_METAL}")
```

!!! note "Metal framework"
    The build system uses Objective-C++ (`.mm` files) to interface with Metal Performance Shaders. The Metal framework is auto-detected via `ctypes.util.find_library('Metal')`.

### MPI Distributed (Multi-node)

For distributed quantum simulation across multiple nodes.

**Prerequisites:**

- OpenMPI or MPICH installed

```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt install libopenmpi-dev openmpi-bin

# CentOS/RHEL
sudo yum install openmpi-devel
```

**Build:**

```bash
QFORGE_MPI=1 pip install -e .
```

**Verify:**

```python
from qforge import _HAS_DISTRIBUTED
print(f"MPI available: {_HAS_DISTRIBUTED}")
```

### All Backends at Once

```bash
QFORGE_CUDA=1 QFORGE_METAL=1 QFORGE_MPI=1 pip install -e .
```

### Selecting a Backend at Runtime

```python
from qforge.circuit import Qubit

wf = Qubit(16, backend='cuda')   # NVIDIA GPU
wf = Qubit(16, backend='metal')  # Apple Silicon GPU
wf = Qubit(16, backend='cpu')    # C++ CPU engine
wf = Qubit(16, backend='python') # Pure Python fallback
wf = Qubit(16, backend='auto')   # Auto-select fastest (default)
```

The default `backend='auto'` picks the fastest available backend in order: **CUDA > Metal > C++ CPU > Python**.

### MPS/DMRG Tensor Networks

The MPS/DMRG backend enables simulation of low-entanglement circuits with 50--1000+ qubits. Requires [Eigen3](https://eigen.tuxfamily.org/).

```bash
# macOS
brew install eigen

# Ubuntu/Debian
sudo apt install libeigen3-dev

# conda
conda install eigen
```

Eigen3 is auto-detected during `pip install -e .`. Verify:

```python
from qforge import _HAS_MPS
print(f"MPS available: {_HAS_MPS}")
```

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
