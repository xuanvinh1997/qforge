# -*- coding: utf-8 -*-
# author: vinhpx
"""Circuit visualization — ASCII text drawing."""
from __future__ import annotations

from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp


def draw_circuit(circuit: Circuit, output: str = 'text') -> str:
    """Draw a circuit diagram.

    Args:
        circuit: The circuit to draw.
        output:  ``'text'`` for ASCII art (default).

    Returns:
        String representation of the circuit.
    """
    if output == 'text':
        return _draw_text(circuit)
    raise ValueError(f"Unknown output format: {output!r}")


def _draw_text(circuit: Circuit) -> str:
    n = circuit.n_qubits
    lines = [[] for _ in range(n)]

    for op in circuit.ops:
        col = ['-----'] * n

        if isinstance(op, MeasureOp):
            col[op.qubit] = '-|M|-'
        elif isinstance(op, ConditionalOp):
            inner = op.op
            _place_gate(col, inner, n, cond=True)
        elif isinstance(op, GateOp):
            _place_gate(col, op, n)

        for i in range(n):
            lines[i].append(col[i])

    result = []
    for i in range(n):
        label = f'q{i}: '
        wire = ''.join(lines[i])
        if not wire:
            wire = '-----'
        result.append(label + wire + '-')

    return '\n'.join(result)


def _gate_label(op: GateOp) -> str:
    name = op.name
    if op.params:
        param_str = ','.join(f'{p:.2g}' for p in op.params)
        return f'{name}({param_str})'
    return name


def _place_gate(col: list[str], op: GateOp, n: int, cond: bool = False) -> None:
    prefix = 'c:' if cond else ''
    qubits = op.qubits

    if len(qubits) == 1:
        label = prefix + _gate_label(op)
        col[qubits[0]] = f'-[{label}]-'
    elif len(qubits) == 2:
        q0, q1 = qubits
        lo, hi = min(q0, q1), max(q0, q1)
        if op.name in ('CNOT', 'CX'):
            col[q0] = '--*--'
            col[q1] = '-(+)-'
        elif op.name == 'SWAP':
            col[q0] = '--x--'
            col[q1] = '--x--'
        else:
            label = prefix + _gate_label(op)
            col[q0] = '--o--'
            col[q1] = f'-[{label}]-'
        for q in range(lo + 1, hi):
            col[q] = '--+--'
    elif len(qubits) >= 3:
        label = prefix + op.name
        q0, q1, q2 = qubits[0], qubits[1], qubits[2]
        lo, hi = min(qubits), max(qubits)
        col[q0] = '--o--'
        col[q1] = '--o--'
        col[q2] = f'-[{label}]-'
        for q in range(lo + 1, hi):
            if q not in qubits:
                col[q] = '--+--'
