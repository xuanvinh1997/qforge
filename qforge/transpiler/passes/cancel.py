# -*- coding: utf-8 -*-
"""Cancel consecutive inverse gate pairs on each qubit wire.

Recognised patterns:
- Self-inverse pairs: H-H, X-X, Y-Y, Z-Z, CNOT-CNOT (same qubits)
- Rotation cancellation: RX(a) + RX(-a), RY(a) + RY(-a), RZ(a) + RZ(-a)
"""
from __future__ import annotations

import math

from qforge.ir import GateOp
from qforge.transpiler.dag import DAGCircuit, DAGNode
from qforge.transpiler.pass_manager import TranspilerPass

_SELF_INVERSE = {'H', 'X', 'Y', 'Z', 'CNOT', 'SWAP'}

_ROTATION_GATES = {'RX', 'RY', 'RZ', 'Phase', 'CRX', 'CRY', 'CRZ', 'CPhase', 'CP'}

_EPS = 1e-9


def _are_inverses(op1: GateOp, op2: GateOp) -> bool:
    """Return True if *op1* followed by *op2* is identity."""
    if op1.name != op2.name:
        return False
    if op1.qubits != op2.qubits:
        return False
    if op1.controls != op2.controls:
        return False

    if op1.name in _SELF_INVERSE:
        return True

    if op1.name in _ROTATION_GATES:
        if len(op1.params) == 1 and len(op2.params) == 1:
            total = op1.params[0] + op2.params[0]
            # Check if total angle is a multiple of 2*pi (effectively zero)
            return abs(total % (2 * math.pi)) < _EPS or abs(total % (2 * math.pi) - 2 * math.pi) < _EPS

    return False


class CancelInverses(TranspilerPass):
    """Remove consecutive pairs of inverse gates on each qubit wire."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        while changed:
            changed = False
            for q in range(dag.n_qubits):
                wire_nodes = dag.nodes_on_wire(q)
                i = 0
                while i < len(wire_nodes) - 1:
                    n1 = wire_nodes[i]
                    n2 = wire_nodes[i + 1]
                    # Verify they are still in the DAG
                    if n1.id not in dag._nodes or n2.id not in dag._nodes:
                        i += 1
                        continue
                    # For multi-qubit gates, ensure n2 is an immediate
                    # successor of n1 on ALL shared wires (not just this one)
                    op1, op2 = n1.op, n2.op
                    if _are_inverses(op1, op2):
                        # For 2q gates, verify adjacency on all wires
                        all_qubits = set(op1.qubits) | set(op1.controls)
                        adjacent_on_all = True
                        for qq in all_qubits:
                            wn = dag.nodes_on_wire(qq)
                            ids = [nd.id for nd in wn]
                            if n1.id in ids and n2.id in ids:
                                i1 = ids.index(n1.id)
                                i2 = ids.index(n2.id)
                                if i2 != i1 + 1:
                                    adjacent_on_all = False
                                    break
                            else:
                                adjacent_on_all = False
                                break

                        if adjacent_on_all:
                            dag.remove(n1.id)
                            dag.remove(n2.id)
                            changed = True
                            # Re-fetch wire after modification
                            wire_nodes = dag.nodes_on_wire(q)
                            continue
                    i += 1
        return dag
