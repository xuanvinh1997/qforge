# Benchmark Results

Comprehensive benchmark results for Qforge across all backends and benchmark suites.
For feature support details, see the [Backend Comparison](backend-comparison.md) page.

---

## System Configuration

=== "macOS (Apple Silicon)"

    | Spec | Value |
    |------|-------|
    | Machine | MacBook Pro (2021) |
    | Chip | Apple M1 Max (10-core CPU, 32-core GPU) |
    | RAM | 64 GB Unified Memory |
    | OS | macOS 26.3 |
    | Python | 3.9.6 |
    | Qforge | v3.0.0 (C++ + Metal) |
    | Architecture | arm64 |

=== "Linux (CUDA)"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## How to Reproduce

```python
from qforge.benchmarks import run_benchmarks, BenchConfig

# Run all suites
results = run_benchmarks()

# Run specific suites
results = run_benchmarks(suites=["gates", "scaling"])

# Custom configuration
config = BenchConfig(n_runs=10, max_qubits=24)
results = run_benchmarks(config=config)
```

For CUDA benchmarks, set the environment variable:

```bash
QFORGE_BENCH_CUDA=1 python -c "from qforge.benchmarks import run_benchmarks; run_benchmarks()"
```

!!! note "Methodology"
    All benchmarks use **2 warmup runs** and report the **median of 5 runs** unless otherwise noted.
    Times are measured with `time.perf_counter()` for high-resolution wall-clock timing.

---

## 1. Primitive Gate Operations

Microseconds per gate (median, 1000 repetitions per measurement).

=== "CPU"

    **4 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

    **12 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

    **20 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

=== "Metal"

    **4 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

    **12 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

    **20 qubits**

    | Gate  | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-------|-------------|----------------|-------------|---------|
    | H     | TBD         | TBD            | TBD         | TBD     |
    | X     | TBD         | TBD            | TBD         | TBD     |
    | RX    | TBD         | TBD            | TBD         | TBD     |
    | RZ    | TBD         | TBD            | TBD         | TBD     |
    | CNOT  | TBD         | TBD            | TBD         | TBD     |
    | SWAP  | TBD         | TBD            | TBD         | TBD     |
    | CCNOT | TBD         | TBD            | TBD         | TBD     |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 2. Circuit Execution Patterns

Time in milliseconds for complete circuit execution.

=== "CPU"

    | Circuit Pattern | Qubits | Qforge (ms) | PennyLane (ms) | Qiskit (ms) | Speedup |
    |-----------------|--------|-------------|----------------|-------------|---------|
    | H + CNOT chain (depth=10) | 8  | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 12 | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 16 | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 20 | TBD | TBD | TBD | TBD |
    | QFT-like | 8  | TBD | TBD | TBD | TBD |
    | QFT-like | 12 | TBD | TBD | TBD | TBD |
    | QFT-like | 16 | TBD | TBD | TBD | TBD |
    | Random 50 gates | 8  | TBD | TBD | TBD | TBD |
    | Random 50 gates | 12 | TBD | TBD | TBD | TBD |
    | Random 50 gates | 16 | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 8  | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 12 | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 16 | TBD | TBD | TBD | TBD |

=== "Metal"

    | Circuit Pattern | Qubits | Qforge (ms) | PennyLane (ms) | Qiskit (ms) | Speedup |
    |-----------------|--------|-------------|----------------|-------------|---------|
    | H + CNOT chain (depth=10) | 8  | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 12 | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 16 | TBD | TBD | TBD | TBD |
    | H + CNOT chain (depth=10) | 20 | TBD | TBD | TBD | TBD |
    | QFT-like | 8  | TBD | TBD | TBD | TBD |
    | QFT-like | 12 | TBD | TBD | TBD | TBD |
    | QFT-like | 16 | TBD | TBD | TBD | TBD |
    | Random 50 gates | 8  | TBD | TBD | TBD | TBD |
    | Random 50 gates | 12 | TBD | TBD | TBD | TBD |
    | Random 50 gates | 16 | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 8  | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 12 | TBD | TBD | TBD | TBD |
    | HEA 3 layers | 16 | TBD | TBD | TBD | TBD |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 3. VQE Algorithm

Variational Quantum Eigensolver performance.

=== "CPU"

    | Molecule | Qubits | Params | Qforge (s) | PennyLane (s) | Qiskit (s) | Speedup |
    |----------|--------|--------|------------|---------------|------------|---------|
    | H2       | 2      | TBD    | TBD        | TBD           | TBD        | TBD     |
    | LiH      | 6      | TBD    | TBD        | TBD           | TBD        | TBD     |

=== "Metal"

    | Molecule | Qubits | Params | Qforge (s) | PennyLane (s) | Qiskit (s) | Speedup |
    |----------|--------|--------|------------|---------------|------------|---------|
    | H2       | 2      | TBD    | TBD        | TBD           | TBD        | TBD     |
    | LiH      | 6      | TBD    | TBD        | TBD           | TBD        | TBD     |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 4. QAOA (Max-Cut)

QAOA algorithm performance on Max-Cut problems.

=== "CPU"

    | Graph Size | Layers | Qforge (s) | PennyLane (s) | Qiskit (s) | Speedup |
    |------------|--------|------------|---------------|------------|---------|
    | TBD        | TBD    | TBD        | TBD           | TBD        | TBD     |

=== "Metal"

    | Graph Size | Layers | Qforge (s) | PennyLane (s) | Qiskit (s) | Speedup |
    |------------|--------|------------|---------------|------------|---------|
    | TBD        | TBD    | TBD        | TBD           | TBD        | TBD     |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 5. Gradient Computation

Parameter-shift gradient computation times.

=== "CPU"

    | Qubits | Params | Qforge (ms) | PennyLane (ms) | Speedup |
    |--------|--------|-------------|----------------|---------|
    | 4      | TBD    | TBD         | TBD            | TBD     |
    | 8      | TBD    | TBD         | TBD            | TBD     |
    | 12     | TBD    | TBD         | TBD            | TBD     |
    | 16     | TBD    | TBD         | TBD            | TBD     |

=== "Metal"

    | Qubits | Params | Qforge (ms) | PennyLane (ms) | Speedup |
    |--------|--------|-------------|----------------|---------|
    | 4      | TBD    | TBD         | TBD            | TBD     |
    | 8      | TBD    | TBD         | TBD            | TBD     |
    | 12     | TBD    | TBD         | TBD            | TBD     |
    | 16     | TBD    | TBD         | TBD            | TBD     |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 6. Measurement Operations

Time per measurement operation.

=== "CPU"

    | Operation | Qubits | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-----------|--------|-------------|----------------|-------------|---------|
    | measure_all | 8    | TBD         | TBD            | TBD         | TBD     |
    | measure_all | 16   | TBD         | TBD            | TBD         | TBD     |
    | measure_one | 8    | TBD         | TBD            | TBD         | TBD     |
    | measure_one | 16   | TBD         | TBD            | TBD         | TBD     |
    | pauli_expectation | 8  | TBD     | TBD            | TBD         | TBD     |
    | pauli_expectation | 16 | TBD     | TBD            | TBD         | TBD     |

=== "Metal"

    | Operation | Qubits | Qforge (us) | PennyLane (us) | Qiskit (us) | Speedup |
    |-----------|--------|-------------|----------------|-------------|---------|
    | measure_all | 8    | TBD         | TBD            | TBD         | TBD     |
    | measure_all | 16   | TBD         | TBD            | TBD         | TBD     |
    | measure_one | 8    | TBD         | TBD            | TBD         | TBD     |
    | measure_one | 16   | TBD         | TBD            | TBD         | TBD     |
    | pauli_expectation | 8  | TBD     | TBD            | TBD         | TBD     |
    | pauli_expectation | 16 | TBD     | TBD            | TBD         | TBD     |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 7. Scalability

HEA 1-layer forward pass time (ms) vs qubit count.

=== "CPU"

    | Qubits | Qforge (ms) | PennyLane (ms) | Qiskit (ms) |
    |--------|-------------|----------------|-------------|
    | 2      | TBD         | TBD            | TBD         |
    | 4      | TBD         | TBD            | TBD         |
    | 6      | TBD         | TBD            | TBD         |
    | 8      | TBD         | TBD            | TBD         |
    | 10     | TBD         | TBD            | TBD         |
    | 12     | TBD         | TBD            | TBD         |
    | 14     | TBD         | TBD            | TBD         |
    | 16     | TBD         | TBD            | TBD         |
    | 18     | TBD         | TBD            | TBD         |
    | 20     | TBD         | TBD            | TBD         |
    | 22     | TBD         | TBD            | TBD         |
    | 24     | TBD         | TBD            | TBD         |

=== "Metal"

    | Qubits | Qforge (ms) | PennyLane (ms) | Qiskit (ms) |
    |--------|-------------|----------------|-------------|
    | 2      | TBD         | TBD            | TBD         |
    | 4      | TBD         | TBD            | TBD         |
    | 6      | TBD         | TBD            | TBD         |
    | 8      | TBD         | TBD            | TBD         |
    | 10     | TBD         | TBD            | TBD         |
    | 12     | TBD         | TBD            | TBD         |
    | 14     | TBD         | TBD            | TBD         |
    | 16     | TBD         | TBD            | TBD         |
    | 18     | TBD         | TBD            | TBD         |
    | 20     | TBD         | TBD            | TBD         |
    | 22     | TBD         | TBD            | TBD         |
    | 24     | TBD         | TBD            | TBD         |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 8. Accuracy & Correctness

Fidelity and numerical accuracy across backends. Tolerance: `1e-12` for CPU, `1e-6` for Metal/CUDA (float32 GPU precision).

| Test | CPU | Metal | CUDA |
|------|-----|-------|------|
| Bell state fidelity | TBD | TBD | TBD |
| GHZ state fidelity | TBD | TBD | TBD |
| Random circuit max diff | TBD | TBD | TBD |
| VQE energy accuracy | TBD | TBD | TBD |
| Gradient accuracy (vs analytic) | TBD | TBD | TBD |

---

## 9. Memory Usage

Peak memory consumption (MB) during circuit execution.

=== "CPU"

    | Qubits | State Vector | Peak Memory (MB) |
    |--------|-------------|-----------------|
    | 10     | 1,024       | TBD             |
    | 14     | 16,384      | TBD             |
    | 18     | 262,144     | TBD             |
    | 20     | 1,048,576   | TBD             |
    | 22     | 4,194,304   | TBD             |
    | 24     | 16,777,216  | TBD             |

=== "Metal"

    | Qubits | State Vector | Peak Memory (MB) |
    |--------|-------------|-----------------|
    | 10     | 1,024       | TBD             |
    | 14     | 16,384      | TBD             |
    | 18     | 262,144     | TBD             |
    | 20     | 1,048,576   | TBD             |
    | 22     | 4,194,304   | TBD             |
    | 24     | 16,777,216  | TBD             |

=== "CUDA"

    !!! info "Coming Soon"
        CUDA benchmark results will be added in a future update.

---

## 10. MPS Benchmarks

Matrix Product State backend performance (CPU only).

| Qubits | Bond Dim | Qforge MPS (ms) | Exact (ms) | Fidelity |
|--------|----------|-----------------|------------|----------|
| TBD    | TBD      | TBD             | TBD        | TBD      |

---

## 11. DMRG Benchmarks

Density Matrix Renormalization Group performance (CPU only).

| System | Qubits | Bond Dim | Qforge DMRG (s) | Energy | Exact Energy |
|--------|--------|----------|-----------------|--------|-------------|
| TBD    | TBD    | TBD      | TBD             | TBD    | TBD         |

---

!!! tip "Generating Your Own Report"
    The benchmark suite also generates an interactive HTML report with charts:
    ```python
    from qforge.benchmarks import run_benchmarks
    results = run_benchmarks()  # Creates benchmark_results/report.html
    ```
