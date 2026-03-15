# Backend Comparison

Performance comparison between the pure-Python and C++ backends for the Qforge
quantum simulation engine. Benchmarks use a depth-10 circuit of alternating
Hadamard and CNOT gates.

## Performance Table

| Qubits | Python (s) | C++ (s) | Speedup |
|--------|-----------|---------|---------|
| 8      | 0.007     | 0.00007 | 107x    |
| 10     | 0.035     | 0.0002  | 172x    |
| 12     | 0.163     | 0.0008  | 205x    |
| 14     | 0.735     | 0.004   | 196x    |
| 20     | --        | 0.37    | --      |
| 22     | --        | 1.66    | --      |

Entries marked with `--` indicate that the Python backend is impractical at
that scale (prohibitive runtime or memory).

## Speedup Factors

The C++ engine achieves **100-200x speedup** over pure Python through:

- **Bitwise indexing** instead of string-based state lookup
- **In-place amplitude updates** using double-buffered scratch arrays (no per-gate allocation)
- **64-byte aligned memory** for SIMD-friendly data layout (`posix_memalign`)
- **OpenMP parallelization** for state vectors larger than 4096 elements
- **GIL release** during all C++ gate operations
- **Zero-copy numpy** views into the C++ `StateVector` (no data copying on read)

## Feature Support Matrix

| Feature | Python | C++ (CPU) | CUDA | Metal |
|---------|--------|-----------|------|-------|
| Single-qubit gates (H, X, Y, Z, S, T) | Yes | Yes | Yes | Yes |
| Rotation gates (RX, RY, RZ, Phase) | Yes | Yes | Yes | Yes |
| Controlled gates (CNOT, CCNOT, CRX, ...) | Yes | Yes | Yes | Yes |
| SWAP / CSWAP / ISWAP / SISWAP | Yes | Yes | Yes | Yes |
| Multi-controlled gates (mcx, mcz, mcp) | Yes | Yes | Yes | Yes |
| Depolarizing noise (E, E_all) | Yes | Yes | No | No |
| Measurement (measure_all, measure_one) | Yes | Yes | Yes | Yes |
| Collapse (collapse_one) | Yes | Yes | Yes | Yes |
| Pauli expectation | Yes | Yes | Yes | Yes |
| PauliZ n-body | Yes | Yes | No | No |
| Reduced density matrix | Yes | Yes | No | No |
| MPS backend | Yes | Yes (via pybind11) | No | No |
| DMRG | Yes | Yes (via pybind11) | No | No |
| Walk_Qubit | Yes | No (pure Python only) | No | No |
| QubitUnitary (arbitrary) | Yes | Yes | Yes | Yes |
| IR / Circuit recording | Yes | Yes | Yes | Yes |
| OpenQASM export/import | Yes | Yes | Yes | Yes |
| Noise model simulation | Yes | Yes | No | No |
| Transpiler passes | Yes | Yes | Yes | Yes |

## Backend Selection

The `Qubit()` constructor accepts a `backend` parameter:

```python
from qforge.circuit import Qubit

wf = Qubit(10, backend='auto')    # CUDA > Metal > CPU > Python
wf = Qubit(10, backend='cpu')     # Force C++ CPU backend
wf = Qubit(10, backend='cuda')    # Force CUDA backend
wf = Qubit(10, backend='metal')   # Force Metal backend
wf = Qubit(10, backend='python')  # Force pure-Python backend
```

The default `'auto'` mode selects the fastest available backend in priority order:
CUDA, Metal, CPU (C++), Python.

## Memory Requirements

| Qubits | State Vector Size | Memory (complex128) |
|--------|-------------------|---------------------|
| 10     | 1,024             | 16 KB               |
| 14     | 16,384            | 256 KB              |
| 20     | 1,048,576         | 16 MB               |
| 22     | 4,194,304         | 64 MB               |
| 25     | 33,554,432        | 512 MB              |
| 28     | 268,435,456       | 4 GB                |
| 30     | 1,073,741,824     | 16 GB               |

Each amplitude is a `complex128` (16 bytes). The C++ backend uses 64-byte
aligned allocation for optimal SIMD performance.

## Scaling Characteristics

- **Python backend:** O(2^n) per gate with significant Python overhead per amplitude.
  Practical up to ~14 qubits.
- **C++ CPU backend:** O(2^n) per gate with minimal overhead. OpenMP parallelization
  activates at 4096 amplitudes. Practical up to ~25 qubits on a workstation.
- **CUDA backend:** O(2^n) per gate with GPU parallelism. Practical up to ~30 qubits
  depending on GPU memory.
- **Metal backend:** Similar to CUDA but for Apple Silicon GPUs.
- **MPS backend:** O(n * chi^3 * d^2) per gate where chi is bond dimension and d=2.
  Efficient for low-entanglement states regardless of qubit count.
