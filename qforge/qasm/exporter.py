# -*- coding: utf-8 -*-
# author: vinhpx
"""Export qforge circuits to OpenQASM 2.0 and 3.0 format."""
from __future__ import annotations

import math

from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp


# ============================================================
# Gate name mapping: qforge -> QASM
# ============================================================

_QASM_GATE_MAP = {
    'H': 'h',
    'X': 'x',
    'Y': 'y',
    'Z': 'z',
    'S': 's',
    'T': 't',
    'CNOT': 'cx',
    'CCNOT': 'ccx',
    'SWAP': 'swap',
    'RX': 'rx',
    'RY': 'ry',
    'RZ': 'rz',
    'Phase': 'p',
    'CRX': 'crx',
    'CRY': 'cry',
    'CRZ': 'crz',
    'CPhase': 'cp',
    'CP': 'cp',
    'Xsquare': 'sx',
    'CSWAP': 'cswap',
}


def _format_params(params: tuple[float, ...]) -> str:
    """Format gate parameters for QASM output."""
    if not params:
        return ''
    parts = []
    for p in params:
        # Use pi-based representation where appropriate
        if abs(p - math.pi) < 1e-10:
            parts.append('pi')
        elif abs(p + math.pi) < 1e-10:
            parts.append('-pi')
        elif abs(p - math.pi / 2) < 1e-10:
            parts.append('pi/2')
        elif abs(p + math.pi / 2) < 1e-10:
            parts.append('-pi/2')
        elif abs(p - math.pi / 4) < 1e-10:
            parts.append('pi/4')
        elif abs(p + math.pi / 4) < 1e-10:
            parts.append('-pi/4')
        else:
            parts.append(f'{p:.10g}')
    return '(' + ', '.join(parts) + ')'


def _qubits_str(qubits: tuple[int, ...], reg: str = 'q') -> str:
    """Format qubit arguments for QASM."""
    return ', '.join(f'{reg}[{q}]' for q in qubits)


# ============================================================
# QASM 2.0 exporter
# ============================================================

def circuit_to_qasm2(circuit: Circuit) -> str:
    """Convert a qforge Circuit to an OpenQASM 2.0 string.

    Args:
        circuit: The circuit to export.

    Returns:
        A string containing valid OpenQASM 2.0 code.
    """
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{circuit.n_qubits}];',
        f'creg c[{circuit.n_qubits}];',
    ]

    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            lines.append(f'measure q[{op.qubit}] -> c[{op.classical_bit}];')

        elif isinstance(op, ConditionalOp):
            # QASM 2.0: if(c[bit]==val) gate q[...];
            inner = op.op
            gate_name = _QASM_GATE_MAP.get(inner.name, inner.name.lower())
            params = _format_params(inner.params)
            qubits = _qubits_str(inner.qubits)
            lines.append(
                f'if(c=={op.expected_value}) {gate_name}{params} {qubits};'
            )

        elif isinstance(op, GateOp):
            gate_name = _QASM_GATE_MAP.get(op.name)
            if gate_name is None:
                # Unknown gate: use lowercase name as-is
                gate_name = op.name.lower()

            params = _format_params(op.params)
            qubits = _qubits_str(op.qubits)
            lines.append(f'{gate_name}{params} {qubits};')

    return '\n'.join(lines) + '\n'


# ============================================================
# QASM 3.0 exporter
# ============================================================

def circuit_to_qasm3(circuit: Circuit) -> str:
    """Convert a qforge Circuit to an OpenQASM 3.0 string.

    Args:
        circuit: The circuit to export.

    Returns:
        A string containing valid OpenQASM 3.0 code.
    """
    lines = [
        'OPENQASM 3.0;',
        'include "stdgates.inc";',
        f'qubit[{circuit.n_qubits}] q;',
        f'bit[{circuit.n_qubits}] c;',
    ]

    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            lines.append(f'c[{op.classical_bit}] = measure q[{op.qubit}];')

        elif isinstance(op, ConditionalOp):
            inner = op.op
            gate_name = _QASM_GATE_MAP.get(inner.name, inner.name.lower())
            params = _format_params(inner.params)
            qubits = _qubits_str(inner.qubits)
            lines.append(
                f'if (c[{op.classical_bit}] == {op.expected_value}) {{'
            )
            lines.append(f'  {gate_name}{params} {qubits};')
            lines.append('}')

        elif isinstance(op, GateOp):
            gate_name = _QASM_GATE_MAP.get(op.name)
            if gate_name is None:
                gate_name = op.name.lower()

            params = _format_params(op.params)
            qubits = _qubits_str(op.qubits)
            lines.append(f'{gate_name}{params} {qubits};')

    return '\n'.join(lines) + '\n'
