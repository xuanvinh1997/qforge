# Qforge.transpiler

Circuit optimization and rewriting framework. Converts circuits into a DAG
representation, applies transformation passes, and produces optimized circuits.

## Usage

```python
from Qforge.transpiler import (
    DAGCircuit, PassManager,
    Decompose, CancelInverses, Optimize1qRotations,
    CommutationAnalysis, BasisTranslator,
)
from Qforge.ir import Circuit, GateOp

circ = Circuit(n_qubits=3)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('H', (0,)))   # redundant: H*H = I
circ.append(GateOp('CNOT', (0, 1)))

pm = PassManager([CancelInverses(), Optimize1qRotations()])
optimized = pm.run(circ)
```

## Sub-modules

| Module | Description |
|--------|-------------|
| [`dag`](dag.md) | DAG circuit representation |
| [`passes`](passes.md) | Optimization and rewriting passes |

## Exported Names

```python
from Qforge.transpiler import (
    DAGCircuit, DAGNode,
    PassManager, TranspilerPass,
    Decompose, CancelInverses, Optimize1qRotations,
    CommutationAnalysis, BasisTranslator,
)
```
