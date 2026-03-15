# API Reference

Complete API documentation for the Qforge quantum simulation framework.

## Core Modules

| Module | Description |
|--------|-------------|
| [`Qforge.circuit`](circuit.md) | Circuit initialization (`Qubit`, `Walk_Qubit`) |
| [`Qforge.wavefunction`](wavefunction.md) | `Wavefunction` class for quantum state representation |
| [`Qforge.gates`](gates.md) | Gate library (30+ gates with C++ dispatch) |
| [`Qforge.measurement`](measurement.md) | Measurement operations and expectation values |
| [`Qforge.encodings`](encodings.md) | 13 quantum data encoding strategies |
| [`Qforge.data`](data.md) | Data analysis: Pauli-Z expectation, correlators, entropy |
| [`Qforge.kernels`](kernels.md) | Quantum kernel methods (state product, SWAP test, projected kernels) |

## Intermediate Representation

| Module | Description |
|--------|-------------|
| [`Qforge.ir`](ir.md) | Circuit IR: `Circuit`, `GateOp`, `MeasureOp`, `ConditionalOp` |
| [`Qforge.parameters`](parameters.md) | Symbolic parameters for parameterized circuits |
| [`Qforge.visualization`](visualization.md) | ASCII circuit drawing |
| [`Qforge.serialization`](serialization.md) | JSON and binary circuit serialization |

## Algorithms

| Module | Description |
|--------|-------------|
| [`Qforge.algo`](algo/index.md) | Algorithm sub-package overview |
| [`Qforge.algo.hamiltonian`](algo/hamiltonian.md) | Hamiltonian and Pauli-string expectations |
| [`Qforge.algo.gradient`](algo/gradient.md) | Parameter-shift gradient rule |
| [`Qforge.algo.optimizers`](algo/optimizers.md) | GradientDescent, Adam, SPSA, LBFGS |
| [`Qforge.algo.ansatz`](algo/ansatz.md) | Hardware-efficient and strongly-entangling ansatze |
| [`Qforge.algo.vqe`](algo/vqe.md) | Variational Quantum Eigensolver |
| [`Qforge.algo.qaoa`](algo/qaoa.md) | Quantum Approximate Optimization Algorithm |
| [`Qforge.algo.standard`](algo/standard.md) | QFT, QPE, Grover, HHL |

## Tensor Network Methods

| Module | Description |
|--------|-------------|
| [`Qforge.mps`](mps.md) | Matrix Product State simulation |
| [`Qforge.dmrg`](dmrg.md) | Density Matrix Renormalization Group |

## Noise and Error Mitigation

| Module | Description |
|--------|-------------|
| [`Qforge.noise`](noise/index.md) | Noise modelling overview |
| [`Qforge.noise.channels`](noise/channels.md) | Quantum noise channels (Kraus operators) |
| [`Qforge.noise.model`](noise/model.md) | `NoiseModel` for associating channels with gates |
| [`Qforge.mitigation`](mitigation.md) | ZNE, PEC, readout correction |

## Quantum Chemistry

| Module | Description |
|--------|-------------|
| [`Qforge.chem`](chem/index.md) | Chemistry sub-package overview |
| [`Qforge.chem.molecule`](chem/molecule.md) | Molecular Hamiltonian construction |
| [`Qforge.chem.fermion`](chem/fermion.md) | Fermionic operators, Jordan-Wigner, Bravyi-Kitaev |
| [`Qforge.chem.ucc`](chem/ucc.md) | UCCSD ansatz |

## Transpiler

| Module | Description |
|--------|-------------|
| [`Qforge.transpiler`](transpiler/index.md) | Transpiler overview |
| [`Qforge.transpiler.dag`](transpiler/dag.md) | DAG circuit representation |
| [`Qforge.transpiler.passes`](transpiler/passes.md) | Optimization and rewriting passes |

## Interoperability

| Module | Description |
|--------|-------------|
| [`Qforge.qasm`](qasm.md) | OpenQASM 2.0/3.0 import and export |
| [`Qforge.interfaces`](interfaces.md) | JAX, PyTorch, TensorFlow bridges |
