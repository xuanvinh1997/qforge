# How-To: Choose the Right Backend

Qforge supports multiple simulation backends. This guide helps you pick the
right one for your workload.

## Available Backends

| Backend | Flag | Requirements | Max Qubits | Best For |
|---------|------|-------------|------------|----------|
| Auto | `'auto'` | -- | varies | Default; picks the best available |
| C++ CPU | `'cpu'` | C++ engine compiled | ~25 | General-purpose, fast |
| CUDA GPU | `'cuda'` | NVIDIA GPU + CUDA | ~30 | Large circuits, batched VQE |
| Metal GPU | `'metal'` | Apple Silicon | ~28 | macOS with M-series chips |
| Python | `'python'` | numpy only | ~20 | Fallback, debugging, education |
| MPS | -- | -- | 100+ | Low-entanglement circuits |
| DensityMatrix | -- | -- | ~13 | Noise simulation, mixed states |

## Setting the Backend

### Global Default

```python
import Qforge
Qforge.set_backend('cpu')  # All subsequent circuits use C++ CPU
```

### Per-Circuit

```python
from Qforge.circuit import Qubit

qc = Qubit(n_qubits=10, backend='cuda')
```

### Auto Selection

The `'auto'` backend (default) selects in this order:
1. CUDA -- if an NVIDIA GPU is detected
2. Metal -- if running on Apple Silicon
3. CPU -- if the C++ engine is compiled
4. Python -- fallback

```python
from Qforge.circuit import Qubit

qc = Qubit(n_qubits=10)  # Uses 'auto'
print(qc.backend)         # Shows which backend was selected
```

## Decision Flowchart

**Do you need noise simulation?**
- Yes: Use `DensityMatrix`. It supports all noise channels and error
  mitigation techniques. Limited to ~13 qubits due to O(4^n) scaling.

**Do you need more than 25 qubits?**
- Yes, with low entanglement: Use `MatrixProductState` (see
  [MPS tutorial](../tutorials/10-mps-large-scale.md)).
- Yes, with high entanglement: Use `'cuda'` or `'metal'` for up to ~30 qubits.

**Do you need gradients for VQE/QML?**
- All backends support `parameter_shift`. GPU backends are fastest for
  batched gradient evaluation.

**Are you debugging or teaching?**
- Use `'python'` for the clearest error messages and stack traces.

## Performance Comparison

### Single Gate Application (H gate, depth=10)

| Qubits | Python | C++ CPU | CUDA | Metal |
|--------|--------|---------|------|-------|
| 8      | 7 ms   | 0.07 ms | 0.1 ms | 0.08 ms |
| 12     | 163 ms | 0.8 ms  | 0.3 ms | 0.4 ms  |
| 16     | 2.9 s  | 15 ms   | 3 ms   | 5 ms    |
| 20     | --     | 370 ms  | 50 ms  | 80 ms   |
| 24     | --     | 6.5 s   | 0.8 s  | 1.3 s   |

> **Tip:** The C++ CPU backend is fastest for small circuits (under 14 qubits)
> due to lower kernel launch overhead compared to GPU backends. GPU backends
> excel at 16+ qubits.

### VQE Optimization (H2, 200 iterations)

| Backend | Time | Notes |
|---------|------|-------|
| Python  | 45 s | Baseline |
| CPU     | 0.4 s | 100x speedup |
| CUDA    | 0.2 s | Batched parameter-shift |
| Metal   | 0.3 s | Apple M2 Pro |

## Backend-Specific Notes

### C++ CPU (`'cpu'`)

- Uses OpenMP for parallelization when dim > 4096 (12+ qubits)
- Set thread count: `export OMP_NUM_THREADS=8`
- 64-byte aligned memory for SIMD vectorization
- Releases the GIL during computation; safe to use with Python threading

### CUDA (`'cuda'`)

- Requires `pip install qforge[cuda]` or manual CUDA toolkit setup
- Best with NVIDIA Ampere (A100) or newer
- Supports batched circuit evaluation for gradient computation
- Memory limited by GPU VRAM (e.g., 40 GB A100 supports ~33 qubits)

### Metal (`'metal'`)

- Automatic on Apple Silicon Macs
- Uses unified memory; no explicit host-device transfers
- Performance comparable to mid-range NVIDIA GPUs for quantum simulation

### Python (`'python'`)

- Pure numpy implementation
- No compilation required
- Full feature parity with C++ (just slower)
- Used automatically for `Walk_Qubit` (quantum walk) operations

### MPS (separate backend)

```python
from Qforge.mps import MatrixProductState

mps = MatrixProductState(n_qubits=50, max_bond_dim=64)
# Same gate API as Qubit
```

- Not selected by `'auto'`; must be created explicitly
- Scales as O(n * chi^2) instead of O(2^n)
- See [MPS tutorial](../tutorials/10-mps-large-scale.md)

### DensityMatrix (separate backend)

```python
from Qforge import DensityMatrix

dm = DensityMatrix(n_qubits=4, noise_model=noise_model)
# Same gate API as Qubit
```

- Required for noise simulation and mixed states
- O(4^n) memory and time -- use only when noise is needed
- See [Noise tutorial](../tutorials/09-noise-and-mitigation.md)

## Checking Backend Availability

```python
import Qforge

print(f"C++ engine: {Qforge._HAS_CPP}")
print(f"CUDA:       {Qforge.has_cuda()}")
print(f"Metal:      {Qforge.has_metal()}")
```

## Summary

| Scenario | Recommended Backend |
|----------|-------------------|
| Quick prototyping, < 15 qubits | `'auto'` (picks CPU) |
| Production VQE/QAOA, < 25 qubits | `'cpu'` or `'cuda'` |
| Large circuits, 25-30 qubits | `'cuda'` or `'metal'` |
| Low-entanglement, 50+ qubits | `MatrixProductState` |
| Noise simulation | `DensityMatrix` |
| Teaching / debugging | `'python'` |
