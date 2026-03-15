# Changelog

All notable changes to Qforge are documented here.

Published: [DOI 10.1088/2632-2153/ac5997](https://doi.org/10.1088/2632-2153/ac5997)

---

## v2.2.0 -- MPS, DMRG, and Tensor Network Methods

### Added

- **Matrix Product State (MPS) backend** (`Qforge.mps.MatrixProductState`)
  - Simulate 50+ qubits for circuits with bounded entanglement
  - Same gate API as the wavefunction backend: `H(mps, ...)`, `CNOT(mps, ...)`
  - Configurable maximum bond dimension (`max_bond_dim`)
  - Entanglement entropy computation per bond: `mps.entanglement_entropy(bond)`
  - Automatic SWAP routing for non-adjacent two-qubit gates
  - Bond dimension profiling
  - C++ acceleration via pybind11 (`_qforge_mps` module)

- **DMRG solver** (`Qforge.dmrg.DMRG`)
  - Find ground states of 1D Hamiltonians directly in MPS form
  - Factory methods: `DMRG.heisenberg()`, `DMRG.ising()`, `DMRG.xxz()`
  - Custom Hamiltonians via MPO specification
  - Variational sweeps with energy convergence tracking
  - Magnetization profiles and correlation functions
  - Python fallback when C++ extension is not available

### Fixed

- MPS two-qubit gate application with bond dimension > 1 (einsum reshape bug)
- DMRG `_apply_heff_py` einsum index typo (`->csth` corrected to `->astc`)
- DMRG Heisenberg MPO factor-of-2 in Sp/Sm terms
- DMRG initial state randomization (|00...0> was an eigenstate of the Heisenberg model)
- DMRG `_single_site_expect_py` einsum contraction

---

## v2.1.0 -- Feature Parity with Qiskit/PennyLane

### Added

- **Circuit IR** -- `Circuit`, `GateOp`, `MeasureOp`, `ConditionalOp` for declarative circuit building
- **Symbolic parameters** -- `Parameter`, `ParameterVector` with `bind_parameters()`
- **Variational algorithms** -- `VQE`, `QAOA`, `VQA` with parameter-shift gradients
  - `Adam`, `GradientDescent`, `SPSA`, `LBFGS` optimizers
  - `hardware_efficient_ansatz`, `strongly_entangling_ansatz` builders
  - `Hamiltonian` class for Pauli-string cost functions
  - `parameter_shift` and `parallel_parameter_shift` gradient computation
- **Quantum chemistry** -- `Molecule`, `FermionicOperator`, `jordan_wigner`, `bravyi_kitaev`, `uccsd_ansatz`
  - Active space reduction for larger molecules
- **Noise simulation** -- `BitFlip`, `PhaseFlip`, `Depolarizing`, `AmplitudeDamping`, `PhaseDamping`, `ThermalRelaxation`
  - `NoiseModel` for attaching noise channels to gates automatically
  - Custom Kraus operator support
- **Error mitigation** -- `zero_noise_extrapolation`, `probabilistic_error_cancellation`, readout correction
- **Transpiler** -- `DAGCircuit`, `PassManager` with `Decompose`, `CancelInverses`, `Optimize1qRotations`, `BasisTranslator`
- **DensityMatrix backend** -- full density matrix simulation for mixed states
  - Partial trace, fidelity, purity, von Neumann entropy
- **StabilizerState** -- Clifford circuit simulation
- **Serialization** -- JSON and binary circuit import/export (`circuit_to_json`, `circuit_from_json`, `circuit_to_binary`, `circuit_from_binary`)
- **OpenQASM** -- QASM 2.0 and 3.0 import/export (`circuit_to_qasm2`, `qasm2_to_circuit`)
- **Visualization** -- `draw_circuit()` for ASCII and graphical circuit diagrams
- **Standard algorithms** -- `qft`, `inverse_qft`, `qpe`, `grover_search`, `hhl`
- **ML interfaces** -- JAX, PyTorch, TensorFlow bridges
- **Multi-controlled gates** -- `mcx`, `mcz`, `mcp` with recursive decomposition
- **Quantum kernels** -- `ProjectedQuantumKernel`, `swap_test`, `hadamard_test`
- **Custom gates** -- `CustomGate`, `register_gate`, `QubitUnitary`
- **Multi-backend support** -- `'auto'`, `'cpu'`, `'cuda'`, `'metal'`, `'python'` backends
  - Per-circuit backend selection: `Qubit(n_qubits=10, backend='cuda')`
  - Global default: `Qforge.set_backend('cpu')`

---

## v2.0.0 -- C++ Acceleration Engine

### Added

- **pybind11 C++ backend** (`_qforge_core` module) -- 100-200x speedup over pure Python
  - 64-byte aligned memory allocation (SIMD-friendly via `posix_memalign`)
  - OpenMP parallelization for circuits with 4096+ amplitudes (12+ qubits)
  - GIL release during all gate operations
  - Zero-copy numpy integration via `StateVector.amplitude` property
- **Double-buffered scratch space** -- no per-gate memory allocation in C++
- **Bitwise indexing** -- replaced string-based state lookups with bit operations
- **CUDA and Metal backends** -- GPU acceleration (experimental)
- All existing gates ported to C++ with Python fallback preserved

### Performance

| Qubits | Python (s) | C++ (s) | Speedup |
|--------|-----------|---------|---------|
| 8      | 0.007     | 0.00007 | 107x    |
| 10     | 0.035     | 0.0002  | 172x    |
| 12     | 0.163     | 0.0008  | 205x    |
| 14     | 0.735     | 0.004   | 196x    |
| 20     | --        | 0.37    | --      |
| 22     | --        | 1.66    | --      |

### Build

- pybind11 extension build via `setup.py` and `pyproject.toml`
- Compiler flags: `-O3 -march=native -ffast-math` (C++17)

---

## v1.0.0 -- Initial Release

### Added

- Pure Python wavefunction-based quantum virtual machine
- Amplitude vector + basis state representation (complex128)
- Gate library: H, X, Y, Z, RX, RY, RZ, Phase, S, T, CNOT, CCNOT, SWAP, CSWAP, ISWAP
- Encoding strategies: `amplitude_encode`, `yz_cx_encode`
- Measurement: `measure_all`, `measure_one`
- Depolarizing noise: `E`, `E_all`
- Data analysis: `PauliZExpectation`, `reduced_density_matrix`
- Quantum walk: `Walk_Qubit` (gate walk)
- Quantum kernel methods
- Entanglement entropy computation
- Published in *Machine Learning: Science and Technology*
  ([DOI: 10.1088/2632-2153/ac5997](https://doi.org/10.1088/2632-2153/ac5997))
