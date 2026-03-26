# Qforge

Large-scale quantum circuit simulator with C++/CUDA/Metal acceleration, MPS/DMRG tensor networks, and a full QML algorithm suite.

## Features

- **Statevector simulation** up to 22+ qubits (C++ engine: 100-200x over pure Python)
- **GPU backends**: CUDA (NVIDIA) and Metal (Apple Silicon)
- **MPI distributed**: shard across nodes for 30-40+ qubits
- **Tensor networks**: MPS, DMRG, TEBD, iTEBD for 50-1000+ qubits
- **30+ quantum gates**: single-qubit, controlled, multi-controlled, swap, noise channels
- **QML algorithms**: VQE, QAOA, VQC, QSVM, QCNN, QGAN, Quantum Reservoir, Data Re-uploading
- **Quantum chemistry**: UCC ansatz, fermion operators, molecular Hamiltonians
- **Transpiler**: circuit optimization passes (commutation, cancellation, decomposition)
- **Interoperability**: OpenQASM 2.0 import/export, PyTorch/JAX/TensorFlow interfaces

## Installation

### Requirements

- Python >= 3.8
- numpy
- pybind11 >= 2.10 (build-time)
- C++17 compiler (for acceleration)

### Basic install (C++ CPU backend)

```bash
git clone https://github.com/xuanvinh1997/qforge.git
cd qforge
pip install -e .
```

Verify:

```bash
python -c "from qforge import _HAS_CPP; print(f'C++ engine: {_HAS_CPP}')"
```

### Pure Python (no compiler needed)

```bash
pip install -e .
```

If the C++ build fails, Qforge falls back to pure Python automatically. All functionality remains available, just slower for large circuits.

### GPU backends

#### CUDA (NVIDIA)

Requires CUDA Toolkit installed (`nvcc` on PATH or `CUDA_HOME` set).

```bash
QFORGE_CUDA=1 pip install -e .
```

```bash
python -c "from qforge import _HAS_CUDA; print(f'CUDA: {_HAS_CUDA}')"
```

#### Metal (Apple Silicon)

macOS only. Uses Metal Performance Shaders for GPU-accelerated simulation.

```bash
QFORGE_METAL=1 pip install -e .
```

```bash
python -c "from qforge import _HAS_METAL; print(f'Metal: {_HAS_METAL}')"
```

### MPI distributed (multi-node)

Requires MPI (OpenMPI or MPICH).

```bash
# Install MPI first
# macOS: brew install open-mpi
# Ubuntu: sudo apt install libopenmpi-dev openmpi-bin

QFORGE_MPI=1 pip install -e .
```

```bash
python -c "from qforge import _HAS_DISTRIBUTED; print(f'MPI: {_HAS_DISTRIBUTED}')"
```

### MPS/DMRG tensor networks

The MPS/DMRG backend requires Eigen3:

```bash
# macOS: brew install eigen
# Ubuntu: sudo apt install libeigen3-dev
# conda: conda install eigen

pip install -e .  # Eigen3 is auto-detected during build
```

```bash
python -c "from qforge import _HAS_MPS; print(f'MPS: {_HAS_MPS}')"
```

### All backends at once

```bash
QFORGE_CUDA=1 QFORGE_METAL=1 QFORGE_MPI=1 pip install -e .
```

### Optional dependencies

```bash
# For QML algorithms
pip install scikit-learn pandas

# For benchmarks comparison
pip install pennylane qiskit

# For notebooks
pip install matplotlib jupyter

# For documentation
pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

## Quick start

```python
from qforge.circuit import Qubit
from qforge.gates import H, CNOT
from qforge.measurement import measure_all

# Bell state
wf = Qubit(2)
H(wf, 0)
CNOT(wf, 0, 1)

print(wf.print_state())
print(measure_all(wf, 1000))
```

## Check available backends

```python
from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL, _HAS_MPS, _HAS_DISTRIBUTED

print(f"C++ CPU:      {_HAS_CPP}")
print(f"CUDA:         {_HAS_CUDA}")
print(f"Metal:        {_HAS_METAL}")
print(f"MPS/DMRG:     {_HAS_MPS}")
print(f"Distributed:  {_HAS_DISTRIBUTED}")
```

## Benchmarks

### Correctness tests + CPU performance

```bash
python benchmark.py
```

Runs gate correctness tests, then benchmarks H+CNOT circuits from 2 to 20 qubits.

### Benchmark framework (all suites)

```bash
python -c "from qforge.benchmarks import run_benchmarks; run_benchmarks()"
```

This runs all 11 benchmark suites and generates an HTML report with charts in `benchmark_results/`.

Available suites:

| Suite | What it measures |
|-------|-----------------|
| `gates` | Individual gate throughput (H, CNOT, RY, etc.) |
| `circuits` | Full circuit execution (GHZ, random, QFT-like) |
| `scaling` | Performance scaling from 2 to 20+ qubits |
| `measurement` | Sampling, expectation values, probabilities |
| `gradient` | Parameter-shift gradient computation |
| `vqe` | VQE optimization convergence and speed |
| `qaoa` | QAOA Max-Cut optimization |
| `accuracy` | Numerical accuracy vs analytical results |
| `memory` | Peak memory usage at various qubit counts |
| `mps` | MPS circuit simulation (50+ qubits) |
| `dmrg` | DMRG ground state finding |

Run specific suites:

```bash
python -c "from qforge.benchmarks import run_benchmarks; run_benchmarks(suites=['gates', 'scaling'])"
```

### Compare with PennyLane and Qiskit

Install the comparison frameworks first:

```bash
pip install pennylane qiskit
```

Then run:

```bash
python -c "
from qforge.benchmarks import run_benchmarks, BenchConfig
config = BenchConfig(frameworks=['qforge', 'pennylane', 'qiskit'])
run_benchmarks(config=config)
"
```

### GPU benchmarks

#### CUDA

```bash
# Build with CUDA
QFORGE_CUDA=1 pip install -e .

# Run — CUDA backend auto-selects when available
python benchmark.py
```

#### Metal

```bash
# Build with Metal
QFORGE_METAL=1 pip install -e .

# Run
python benchmark.py
```

### MPI distributed benchmarks

```bash
# Build with MPI
QFORGE_MPI=1 pip install -e .

# Run on 4 processes (single machine)
mpirun -np 4 python -c "
from qforge.distributed import DistributedQubit
from qforge.gates import H, CNOT
from qforge.measurement import measure_one
import time

for n in [24, 26, 28, 30]:
    wf = DistributedQubit(n)
    t0 = time.perf_counter()
    H(wf, 0)
    for i in range(n - 1):
        CNOT(wf, i, i + 1)
    dt = time.perf_counter() - t0
    p = measure_one(wf, 0)
    print(f'{n} qubits: {dt:.4f}s, P(0)={p[0]:.4f}')
"

# Multi-node: create hostfile and run
# echo "node1 slots=4" > hostfile
# echo "node2 slots=4" >> hostfile
# mpirun --hostfile hostfile -np 8 python my_benchmark.py
```

### MPS/DMRG benchmarks

```bash
python -c "from qforge.benchmarks import run_benchmarks; run_benchmarks(suites=['mps', 'dmrg'])"
```

Or manually:

```python
from qforge.mps import MatrixProductState
from qforge.gates import H, CNOT
from qforge.dmrg import DMRG
import time

# MPS: 100-qubit GHZ
t0 = time.perf_counter()
psi = MatrixProductState(n_qubits=100, max_bond_dim=64)
H(psi, 0)
for i in range(99):
    CNOT(psi, i, i + 1)
print(f"100-qubit GHZ (MPS): {time.perf_counter() - t0:.4f}s")

# DMRG: 50-site Ising ground state
t0 = time.perf_counter()
dmrg = DMRG.ising(n_sites=50, J=1.0, h=1.0, max_bond_dim=32)
energy, _ = dmrg.run(n_sweeps=20)
print(f"50-site Ising DMRG: {time.perf_counter() - t0:.4f}s, E={energy:.6f}")
```

### Custom benchmark config

```python
from qforge.benchmarks import run_benchmarks, BenchConfig

config = BenchConfig(
    n_warmup=3,        # warmup runs before timing
    n_runs=10,         # timed runs (reports median)
    max_qubits=22,     # max qubits for scaling tests
    steps=100,         # optimizer steps for VQE/QAOA
    output_dir="my_benchmarks",  # output directory for reports
)
run_benchmarks(config=config)
```

## Notebooks

Tutorial notebooks are in [`notebooks/`](notebooks/):

| Notebook | Topic |
|----------|-------|
| `01_quantum_circuits.ipynb` | Gates, Bell/GHZ states, measurement, entanglement entropy |
| `02_vqe_qaoa.ipynb` | VQE for H₂, QAOA for Max-Cut |
| `03_qml_classification.ipynb` | VQC, QSVM, QCNN, Reservoir, Data Re-uploading comparison |
| `04_qgan.ipynb` | QGAN for distribution learning |
| `05_tensor_networks.ipynb` | MPS, DMRG, TEBD, iTEBD |

```bash
pip install matplotlib jupyter
jupyter notebook notebooks/
```

## Documentation

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve     # http://localhost:8000
mkdocs build     # static site in site/
```

## Project structure

```
qforge/
├── circuit.py          # Qubit() entry point
├── wavefunction.py     # Wavefunction class
├── gates.py            # 30+ gate functions (C++ dispatch + Python fallback)
├── measurement.py      # measure_all, measure_one, pauli_expectation
├── encodings.py        # 13 quantum data encoding strategies
├── data.py             # PauliZ correlators, entanglement entropy
├── kernels.py          # Quantum kernel methods
├── mps.py              # Matrix Product State backend
├── dmrg.py             # DMRG solver
├── ir.py               # Circuit IR (build, compose, adjoint)
├── algo/               # QML & variational algorithms
│   ├── vqe.py          #   VQE
│   ├── qaoa.py         #   QAOA
│   ├── vqc.py          #   Variational Quantum Classifier
│   ├── qsvm.py         #   Quantum SVM
│   ├── qcnn.py         #   Quantum CNN
│   ├── qgan.py         #   Quantum GAN
│   ├── reservoir.py    #   Quantum Reservoir Computing
│   ├── data_reuploading.py  # Data Re-uploading Classifier
│   ├── hamiltonian.py  #   Hamiltonian class
│   ├── gradient.py     #   Parameter-shift rule
│   ├── optimizers.py   #   Adam, SGD, SPSA, L-BFGS
│   ├── ansatz.py       #   Hardware-efficient, strongly-entangling ansätze
│   └── standard/       #   QFT, QPE, Grover, HHL
├── distributed.py      # MPI distributed backend
├── noise/              # Noise channels and noise models
├── chem/               # Quantum chemistry (UCC, fermion ops)
├── transpiler/         # Circuit optimization passes
├── cpp/                # C++ acceleration engine (pybind11)
└── benchmarks/         # Benchmark framework (11 suites)
```

## License

See [LICENSE](LICENSE) for details.
