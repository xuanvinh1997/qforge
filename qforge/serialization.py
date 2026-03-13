# -*- coding: utf-8 -*-
# author: vinhpx
"""Circuit serialization to JSON and binary formats."""
from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp


# ============================================================
# JSON serialization
# ============================================================

def circuit_to_json(circuit: Circuit) -> str:
    """Serialize a circuit to a JSON string."""
    data = _circuit_to_dict(circuit)
    return json.dumps(data, indent=2)


def circuit_from_json(json_str: str) -> Circuit:
    """Deserialize a circuit from a JSON string."""
    data = json.loads(json_str)
    return _dict_to_circuit(data)


def _circuit_to_dict(circuit: Circuit) -> dict[str, Any]:
    ops = []
    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            ops.append({'type': 'measure', 'qubit': op.qubit, 'classical_bit': op.classical_bit})
        elif isinstance(op, ConditionalOp):
            ops.append({
                'type': 'conditional', 'classical_bit': op.classical_bit,
                'expected_value': op.expected_value, 'op': _gate_to_dict(op.op),
            })
        elif isinstance(op, GateOp):
            ops.append(_gate_to_dict(op))
    return {'n_qubits': circuit.n_qubits, 'ops': ops}


def _gate_to_dict(op: GateOp) -> dict[str, Any]:
    d: dict[str, Any] = {'type': 'gate', 'name': op.name, 'qubits': list(op.qubits)}
    if op.params:
        d['params'] = list(op.params)
    if op.is_adjoint:
        d['is_adjoint'] = True
    if op.controls:
        d['controls'] = list(op.controls)
    if op.matrix is not None:
        d['matrix_real'] = op.matrix.real.tolist()
        d['matrix_imag'] = op.matrix.imag.tolist()
    return d


def _dict_to_circuit(data: dict) -> Circuit:
    circuit = Circuit(data['n_qubits'])
    for op_data in data['ops']:
        op_type = op_data.get('type', 'gate')
        if op_type == 'measure':
            circuit.ops.append(MeasureOp(qubit=op_data['qubit'], classical_bit=op_data['classical_bit']))
        elif op_type == 'conditional':
            inner = _dict_to_gate(op_data['op'])
            circuit.ops.append(ConditionalOp(
                classical_bit=op_data['classical_bit'],
                expected_value=op_data['expected_value'], op=inner,
            ))
        else:
            circuit.ops.append(_dict_to_gate(op_data))
    return circuit


def _dict_to_gate(d: dict) -> GateOp:
    matrix = None
    if 'matrix_real' in d:
        real = np.array(d['matrix_real'])
        imag = np.array(d['matrix_imag'])
        matrix = real + 1j * imag
    return GateOp(
        name=d['name'], qubits=tuple(d['qubits']),
        params=tuple(d.get('params', ())), matrix=matrix,
        is_adjoint=d.get('is_adjoint', False),
        controls=tuple(d.get('controls', ())),
    )


# ============================================================
# Binary serialization
# ============================================================

_MAGIC = b'QFGE'
_VERSION = 1


def circuit_to_binary(circuit: Circuit) -> bytes:
    """Serialize a circuit to a compact binary format."""
    buf = bytearray()
    buf.extend(_MAGIC)
    buf.extend(struct.pack('<HHI', _VERSION, circuit.n_qubits, len(circuit.ops)))
    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            buf.append(1)
            buf.extend(struct.pack('<HH', op.qubit, op.classical_bit))
        elif isinstance(op, ConditionalOp):
            buf.append(2)
            buf.extend(struct.pack('<HH', op.classical_bit, op.expected_value))
            _encode_gate(buf, op.op)
        elif isinstance(op, GateOp):
            buf.append(0)
            _encode_gate(buf, op)
    return bytes(buf)


def _encode_gate(buf: bytearray, op: GateOp) -> None:
    name_bytes = op.name.encode('utf-8')
    buf.append(len(name_bytes))
    buf.extend(name_bytes)
    buf.append(len(op.qubits))
    for q in op.qubits:
        buf.extend(struct.pack('<H', q))
    buf.append(len(op.params))
    for p in op.params:
        buf.extend(struct.pack('<d', p))
    flags = 0
    if op.is_adjoint:
        flags |= 1
    if op.matrix is not None:
        flags |= 2
    if op.controls:
        flags |= 4
    buf.append(flags)
    if flags & 2:
        flat = op.matrix.flatten()
        buf.extend(struct.pack('<I', len(flat)))
        for v in flat:
            buf.extend(struct.pack('<dd', v.real, v.imag))
    if flags & 4:
        buf.append(len(op.controls))
        for c in op.controls:
            buf.extend(struct.pack('<H', c))


def circuit_from_binary(data: bytes) -> Circuit:
    """Deserialize a circuit from binary format."""
    if data[:4] != _MAGIC:
        raise ValueError("Invalid binary format: bad magic bytes")
    offset = 4
    version, n_qubits, n_ops = struct.unpack_from('<HHI', data, offset)
    offset += 8
    if version != _VERSION:
        raise ValueError(f"Unsupported binary format version: {version}")
    circuit = Circuit(n_qubits)
    for _ in range(n_ops):
        op_type = data[offset]; offset += 1
        if op_type == 1:
            qubit, cbit = struct.unpack_from('<HH', data, offset); offset += 4
            circuit.ops.append(MeasureOp(qubit=qubit, classical_bit=cbit))
        elif op_type == 2:
            cbit, cval = struct.unpack_from('<HH', data, offset); offset += 4
            gate, offset = _decode_gate(data, offset)
            circuit.ops.append(ConditionalOp(classical_bit=cbit, expected_value=cval, op=gate))
        elif op_type == 0:
            gate, offset = _decode_gate(data, offset)
            circuit.ops.append(gate)
    return circuit


def _decode_gate(data: bytes, offset: int) -> tuple[GateOp, int]:
    name_len = data[offset]; offset += 1
    name = data[offset:offset + name_len].decode('utf-8'); offset += name_len
    n_q = data[offset]; offset += 1
    qubits = []
    for _ in range(n_q):
        q, = struct.unpack_from('<H', data, offset); offset += 2; qubits.append(q)
    n_p = data[offset]; offset += 1
    params = []
    for _ in range(n_p):
        p, = struct.unpack_from('<d', data, offset); offset += 8; params.append(p)
    flags = data[offset]; offset += 1
    matrix = None
    if flags & 2:
        flat_len, = struct.unpack_from('<I', data, offset); offset += 4
        flat = []
        for _ in range(flat_len):
            re_part, im_part = struct.unpack_from('<dd', data, offset); offset += 16
            flat.append(complex(re_part, im_part))
        side = int(np.sqrt(flat_len))
        matrix = np.array(flat).reshape(side, side)
    controls = ()
    if flags & 4:
        n_c = data[offset]; offset += 1
        ctrl_list = []
        for _ in range(n_c):
            c, = struct.unpack_from('<H', data, offset); offset += 2; ctrl_list.append(c)
        controls = tuple(ctrl_list)
    return GateOp(
        name=name, qubits=tuple(qubits), params=tuple(params),
        matrix=matrix, is_adjoint=bool(flags & 1), controls=controls,
    ), offset
