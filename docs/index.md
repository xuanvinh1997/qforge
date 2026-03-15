# Qforge - Quantum Virtual Machine

Qforge is a wavefunction-based quantum virtual machine (QVM) for simulating quantum circuits on classical hardware. It provides native tools for variational quantum algorithms, quantum differentiable programming via the parameter-shift rule, and quantum machine learning applications.

Published in *Machine Learning: Science and Technology* ([DOI: 10.1088/2632-2153/ac5997](https://doi.org/10.1088/2632-2153/ac5997)).

## Key Features

- **Wavefunction simulation** of up to 22+ qubits with full statevector access
- **C++ acceleration engine** via pybind11 delivering 100--200x speedup over pure Python
- **Multi-backend support**: automatic dispatch across CUDA, Metal, C++, and pure Python
- **30+ quantum gates** including single-qubit, controlled, multi-controlled, swap, and noise channels
- **Circuit IR** for building, composing, inverting, and parameterizing circuits
- **Mid-circuit measurement** with classical conditioning
- **Measurement toolkit**: sampling, single-qubit probabilities, projective collapse, Pauli expectation values
- **Entanglement analysis**: von Neumann entropy, reduced density matrices, Pauli-Z correlators, connected correlators
- **Custom gate registry** for defining reusable gates from matrices or sub-circuits
- **ASCII circuit visualization** built in

## Quick Example: Bell State

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT
from Qforge.measurement import measure_all

# Create a 2-qubit circuit in |00>
wf = Qubit(2)

# Apply Hadamard on qubit 0, then CNOT
H(wf, 0)
CNOT(wf, 0, 1)

# Inspect the state
print(wf.print_state())
# (0.707+0j)|00> + (0+0j)|01> + (0+0j)|10> + (0.707+0j)|11>

print(wf.probabilities())
# [0.5 0.  0.  0.5]

# Sample measurements
states, counts = measure_all(wf, 1000)
print(dict(zip(states, counts)))
# {'00': 502, '11': 498}  (approximate)

# Visualize
wf.visual_circuit()
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Installation](getting-started/installation.md) | Install Qforge with or without the C++ engine |
| [Quickstart](getting-started/quickstart.md) | Build your first Bell state in 5 minutes |
| [Core Concepts](getting-started/concepts.md) | Wavefunction model, qubit indexing, backends, APIs |
| [First Circuit](tutorials/01-first-circuit.md) | Step-by-step tutorial with both functional and IR APIs |
| [Gates and Rotations](tutorials/02-gates-and-rotations.md) | Complete gate catalog with examples |
| [Measurement](tutorials/03-measurement.md) | Sampling, probabilities, collapse, expectation values |
| [Entanglement](tutorials/04-entanglement.md) | Bell states, GHZ, entropy, correlators |
| [Circuit IR](tutorials/05-circuit-ir.md) | Build, compose, adjoint, parameterize, and replay circuits |

## Citation

If you use Qforge in your research, please cite:

```bibtex
@article{qforge2022,
  title={Qforge: A quantum virtual machine for quantum computing simulation},
  journal={Machine Learning: Science and Technology},
  year={2022},
  doi={10.1088/2632-2153/ac5997}
}
```

## License

See the repository root for license details.
