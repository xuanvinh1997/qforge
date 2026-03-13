# -*- coding: utf-8 -*-
# author: vinhpx
"""Import OpenQASM 2.0 and 3.0 circuits into qforge."""
from __future__ import annotations

import re
import math as _math

from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp


# ============================================================
# QASM gate name -> qforge name mapping
# ============================================================

_GATE_MAP = {
    'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z',
    's': 'S', 't': 'T', 'sx': 'Xsquare',
    'cx': 'CNOT', 'cnot': 'CNOT',
    'ccx': 'CCNOT', 'cswap': 'CSWAP', 'swap': 'SWAP',
    'rx': 'RX', 'ry': 'RY', 'rz': 'RZ',
    'p': 'Phase', 'phase': 'Phase',
    'crx': 'CRX', 'cry': 'CRY', 'crz': 'CRZ',
    'cp': 'CPhase', 'cphase': 'CPhase',
}

_RE_REG = re.compile(r'(\w+)\[(\d+)\]')


def _eval_param(token: str) -> float:
    """Evaluate a QASM parameter expression (simple cases)."""
    token = token.strip()
    token = token.replace('pi', str(_math.pi))
    try:
        return float(eval(token, {"__builtins__": {}}, {}))
    except Exception:
        return float(token)


def _parse_params(s: str) -> tuple[float, ...]:
    """Parse '(param1, param2, ...)' -> tuple of floats."""
    s = s.strip()
    if not s or s == '()':
        return ()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    parts = s.split(',')
    return tuple(_eval_param(p) for p in parts)


def _parse_qubits(s: str) -> list[int]:
    """Parse 'q[0], q[1]' -> [0, 1]."""
    qubits = []
    for m in _RE_REG.finditer(s):
        qubits.append(int(m.group(2)))
    return qubits


# ============================================================
# QASM 2.0 importer
# ============================================================

def qasm2_to_circuit(qasm_str: str) -> Circuit:
    """Parse an OpenQASM 2.0 string into a qforge Circuit.

    Args:
        qasm_str: OpenQASM 2.0 source.

    Returns:
        A :class:`~qforge.ir.Circuit`.
    """
    lines = qasm_str.strip().split('\n')
    n_qubits = None

    for line in lines:
        line = line.strip()
        m = re.match(r'qreg\s+\w+\[(\d+)\];', line)
        if m:
            n_qubits = int(m.group(1))
            break

    if n_qubits is None:
        raise ValueError("No qreg declaration found in QASM string")

    circuit = Circuit(n_qubits)

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        if line.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'barrier')):
            continue

        # Conditional: if(c==val) gate ...
        cond_match = re.match(r'if\s*\(\s*c\s*==\s*(\d+)\s*\)\s*(.+)', line)
        if cond_match:
            cond_val = int(cond_match.group(1))
            gate_line = cond_match.group(2)
            op = _parse_gate_line(gate_line)
            if op is not None:
                circuit.ops.append(ConditionalOp(
                    classical_bit=0, expected_value=cond_val, op=op,
                ))
            continue

        # Measurement: measure q[0] -> c[0];
        meas_match = re.match(r'measure\s+\w+\[(\d+)\]\s*->\s*\w+\[(\d+)\]\s*;', line)
        if meas_match:
            circuit.ops.append(MeasureOp(
                qubit=int(meas_match.group(1)),
                classical_bit=int(meas_match.group(2)),
            ))
            continue

        op = _parse_gate_line(line)
        if op is not None:
            circuit.ops.append(op)

    return circuit


def _parse_gate_line(line: str) -> GateOp | None:
    """Parse a single gate instruction line into a GateOp."""
    line = line.strip().rstrip(';').strip()
    if not line:
        return None

    paren_match = re.match(r'(\w+)\s*\(([^)]*)\)\s+(.+)', line)
    if paren_match:
        gate_name = paren_match.group(1).strip()
        params = _parse_params(paren_match.group(2))
        qubit_str = paren_match.group(3)
    else:
        parts = line.split(None, 1)
        if len(parts) < 2:
            return None
        gate_name = parts[0].strip()
        params = ()
        qubit_str = parts[1]

    qubits = _parse_qubits(qubit_str)
    if not qubits:
        return None

    qforge_name = _GATE_MAP.get(gate_name.lower())
    if qforge_name is None:
        qforge_name = gate_name.upper()

    return GateOp(name=qforge_name, qubits=tuple(qubits), params=params)


# ============================================================
# QASM 3.0 importer
# ============================================================

def qasm3_to_circuit(qasm_str: str) -> Circuit:
    """Parse an OpenQASM 3.0 string into a qforge Circuit.

    Args:
        qasm_str: OpenQASM 3.0 source.

    Returns:
        A :class:`~qforge.ir.Circuit`.
    """
    lines = qasm_str.strip().split('\n')
    n_qubits = None

    for line in lines:
        line = line.strip()
        m = re.match(r'qubit\[(\d+)\]\s+\w+\s*;', line)
        if m:
            n_qubits = int(m.group(1))
            break

    if n_qubits is None:
        raise ValueError("No qubit declaration found in QASM 3.0 string")

    circuit = Circuit(n_qubits)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith('//'):
            continue
        if line.startswith(('OPENQASM', 'include', 'qubit', 'bit', 'barrier')):
            continue

        # QASM 3.0 measurement: c[i] = measure q[j];
        meas_match = re.match(
            r'\w+\[(\d+)\]\s*=\s*measure\s+\w+\[(\d+)\]\s*;', line
        )
        if meas_match:
            circuit.ops.append(MeasureOp(
                qubit=int(meas_match.group(2)),
                classical_bit=int(meas_match.group(1)),
            ))
            continue

        # QASM 3.0 conditional: if (c[bit] == val) { ... }
        cond_match = re.match(
            r'if\s*\(\s*\w+\[(\d+)\]\s*==\s*(\d+)\s*\)\s*\{', line
        )
        if cond_match:
            cbit = int(cond_match.group(1))
            cval = int(cond_match.group(2))
            while i < len(lines):
                inner = lines[i].strip()
                i += 1
                if inner == '}':
                    break
                op = _parse_gate_line(inner)
                if op is not None:
                    circuit.ops.append(ConditionalOp(
                        classical_bit=cbit, expected_value=cval, op=op,
                    ))
            continue

        op = _parse_gate_line(line)
        if op is not None:
            circuit.ops.append(op)

    return circuit
