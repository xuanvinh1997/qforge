# -*- coding: utf-8 -*-
"""Optimise consecutive single-qubit rotation gates.

- Merge same-axis rotations: ``RX(a) + RX(b) → RX(a+b)``
- Remove rotations with angle ≈ 0 (mod 2π)
- Merge RZ-RY-RZ sequences into a single ZYZ decomposition
"""
from __future__ import annotations

import math

from qforge.ir import GateOp
from qforge.transpiler.dag import DAGCircuit, DAGNode
from qforge.transpiler.pass_manager import TranspilerPass

_SINGLE_QUBIT_ROTATIONS = {'RX', 'RY', 'RZ', 'Phase'}

_EPS = 1e-9


def _is_1q_rotation(op: GateOp) -> bool:
    return op.name in _SINGLE_QUBIT_ROTATIONS and len(op.qubits) == 1


def _angle_is_zero(angle: float) -> bool:
    """Check if angle is effectively zero (mod 2pi)."""
    a = angle % (2 * math.pi)
    return a < _EPS or (2 * math.pi - a) < _EPS


def _make_rotation(name: str, qubit: int, angle: float) -> GateOp | None:
    """Create a rotation GateOp, returning None if angle ≈ 0."""
    if _angle_is_zero(angle):
        return None
    return GateOp(name=name, qubits=(qubit,), params=(angle,))


class Optimize1qRotations(TranspilerPass):
    """Merge and simplify consecutive single-qubit rotation gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for q in range(dag.n_qubits):
            self._optimize_wire(dag, q)
        return dag

    def _optimize_wire(self, dag: DAGCircuit, qubit: int) -> None:
        """Optimize rotations on a single qubit wire."""
        changed = True
        while changed:
            changed = False

            # Pass 1: merge consecutive same-axis rotations
            wire = dag.nodes_on_wire(qubit)
            i = 0
            while i < len(wire) - 1:
                n1 = wire[i]
                n2 = wire[i + 1]
                if n1.id not in dag._nodes or n2.id not in dag._nodes:
                    i += 1
                    continue

                op1, op2 = n1.op, n2.op
                if (_is_1q_rotation(op1) and _is_1q_rotation(op2)
                        and op1.name == op2.name
                        and op1.qubits == op2.qubits):
                    merged_angle = op1.params[0] + op2.params[0]
                    replacement = _make_rotation(op1.name, qubit, merged_angle)
                    # Remove both, optionally insert merged
                    dag.remove(n2.id)
                    if replacement is not None:
                        dag.replace(n1.id, [replacement])
                    else:
                        dag.remove(n1.id)
                    changed = True
                    wire = dag.nodes_on_wire(qubit)
                    continue
                i += 1

            # Pass 2: remove near-zero rotations
            wire = dag.nodes_on_wire(qubit)
            for node in wire:
                if node.id not in dag._nodes:
                    continue
                op = node.op
                if _is_1q_rotation(op) and _angle_is_zero(op.params[0]):
                    dag.remove(node.id)
                    changed = True

            # Pass 3: merge RZ-RY-RZ into ZYZ decomposition
            wire = dag.nodes_on_wire(qubit)
            i = 0
            while i < len(wire) - 2:
                n1, n2, n3 = wire[i], wire[i + 1], wire[i + 2]
                if (n1.id not in dag._nodes or n2.id not in dag._nodes
                        or n3.id not in dag._nodes):
                    i += 1
                    continue

                o1, o2, o3 = n1.op, n2.op, n3.op
                if (o1.name == 'RZ' and o2.name == 'RY' and o3.name == 'RZ'
                        and o1.qubits == o2.qubits == o3.qubits == (qubit,)):
                    # These three form a ZYZ decomposition — keep as-is but
                    # check if the total rotation is trivial
                    alpha = o1.params[0]
                    beta = o2.params[0]
                    gamma = o3.params[0]

                    # If beta ≈ 0, the whole thing is RZ(alpha + gamma)
                    if _angle_is_zero(beta):
                        total = alpha + gamma
                        replacement = _make_rotation('RZ', qubit, total)
                        dag.remove(n2.id)
                        dag.remove(n3.id)
                        if replacement is not None:
                            dag.replace(n1.id, [replacement])
                        else:
                            dag.remove(n1.id)
                        changed = True
                        wire = dag.nodes_on_wire(qubit)
                        continue
                i += 1
