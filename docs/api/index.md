# API Reference

Complete API documentation for the Qforge quantum simulation framework.

## Core Modules

| Module | Description |
|--------|-------------|
| [`qforge.circuit`](circuit.md) | Circuit initialization (`Qubit`, `Walk_Qubit`) |
| [`qforge.wavefunction`](wavefunction.md) | `Wavefunction` class for quantum state representation |
| [`qforge.gates`](gates.md) | Gate library (30+ gates with C++ dispatch) |
| [`qforge.measurement`](measurement.md) | Measurement operations and expectation values |
| [`qforge.encodings`](encodings.md) | 13 quantum data encoding strategies |
| [`qforge.data`](data.md) | Data analysis: Pauli-Z expectation, correlators, entropy |
| [`qforge.kernels`](kernels.md) | Quantum kernel methods (state product, SWAP test, projected kernels) |

## Intermediate Representation

| Module | Description |
|--------|-------------|
| [`qforge.ir`](ir.md) | Circuit IR: `Circuit`, `GateOp`, `MeasureOp`, `ConditionalOp` |
| [`qforge.parameters`](parameters.md) | Symbolic parameters for parameterized circuits |
| [`qforge.visualization`](visualization.md) | ASCII circuit drawing |
| [`qforge.serialization`](serialization.md) | JSON and binary circuit serialization |

## Algorithms

| Module | Description |
|--------|-------------|
| [`qforge.algo`](algo/index.md) | Algorithm sub-package overview |
| [`qforge.algo.hamiltonian`](algo/hamiltonian.md) | Hamiltonian and Pauli-string expectations |
| [`qforge.algo.gradient`](algo/gradient.md) | Parameter-shift gradient rule |
| [`qforge.algo.optimizers`](algo/optimizers.md) | GradientDescent, Adam, SPSA, LBFGS |
| [`qforge.algo.ansatz`](algo/ansatz.md) | Hardware-efficient and strongly-entangling ansatze |
| [`qforge.algo.vqe`](algo/vqe.md) | Variational Quantum Eigensolver |
| [`qforge.algo.qaoa`](algo/qaoa.md) | Quantum Approximate Optimization Algorithm |
| [`qforge.algo.standard`](algo/standard.md) | QFT, QPE, Grover, HHL |

## Tensor Network Methods

| Module | Description |
|--------|-------------|
| [`qforge.mps`](mps.md) | Matrix Product State simulation |
| [`qforge.dmrg`](dmrg.md) | Density Matrix Renormalization Group |

## Noise and Error Mitigation

| Module | Description |
|--------|-------------|
| [`qforge.noise`](noise/index.md) | Noise modelling overview |
| [`qforge.noise.channels`](noise/channels.md) | Quantum noise channels (Kraus operators) |
| [`qforge.noise.model`](noise/model.md) | `NoiseModel` for associating channels with gates |
| [`qforge.mitigation`](mitigation.md) | ZNE, PEC, readout correction |

## Quantum Chemistry

| Module | Description |
|--------|-------------|
| [`qforge.chem`](chem/index.md) | Chemistry sub-package overview |
| [`qforge.chem.molecule`](chem/molecule.md) | Molecular Hamiltonian construction |
| [`qforge.chem.fermion`](chem/fermion.md) | Fermionic operators, Jordan-Wigner, Bravyi-Kitaev |
| [`qforge.chem.ucc`](chem/ucc.md) | UCCSD ansatz |

## Transpiler

| Module | Description |
|--------|-------------|
| [`qforge.transpiler`](transpiler/index.md) | Transpiler overview |
| [`qforge.transpiler.dag`](transpiler/dag.md) | DAG circuit representation |
| [`qforge.transpiler.passes`](transpiler/passes.md) | Optimization and rewriting passes |

## Interoperability

| Module | Description |
|--------|-------------|
| [`qforge.qasm`](qasm.md) | OpenQASM 2.0/3.0 import and export |
| [`qforge.interfaces`](interfaces.md) | JAX, PyTorch, TensorFlow bridges |
