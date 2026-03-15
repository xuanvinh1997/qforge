# Qforge.visualization

ASCII text circuit drawing for inspecting quantum circuits built with the
`qforge.ir.Circuit` intermediate representation.

## Usage

```python
from qforge.ir import Circuit, GateOp, MeasureOp
from qforge.visualization import draw_circuit

circ = Circuit(n_qubits=3)
circ.append(GateOp('H', (0,)))
circ.append(GateOp('CNOT', (0, 1)))
circ.append(GateOp('CNOT', (1, 2)))
circ.append(MeasureOp(qubit=2, classical_bit=0))

print(draw_circuit(circ))
# q0: --[H]---@-----------
# q1: --------X----@------
# q2: -------------X-|M|--
```

## Functions

### `draw_circuit(circuit, output='text')`

Draw a circuit diagram.

**Parameters:**

- `circuit` -- A `Circuit` object from `qforge.ir`.
- `output` -- Output format. Currently only `'text'` (ASCII art) is supported.

**Returns:** `str` -- String representation of the circuit.

## Full API

::: qforge.visualization
    options:
      show_source: false
