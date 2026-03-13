# -*- coding: utf-8 -*-
"""Commutation analysis pass.

Identifies pairs of gates that commute and reorders them to cluster
cancellable / mergeable operations together, enabling subsequent
optimisation passes.
"""
from __future__ import annotations

import math
from typing import FrozenSet

from qforge.ir import GateOp
from qforge.transpiler.dag import DAGCircuit, DAGNode
from qforge.transpiler.pass_manager import TranspilerPass


# ============================================================
# Commutation rules
# ============================================================

# Two gates commute if their combined action is order-independent.
# We encode this as a lookup of known commuting pairs.

# Diagonal gates commute with each other
_DIAGONAL_GATES = {'Z', 'S', 'T', 'RZ', 'Phase', 'CPhase', 'CP'}

# Gates that commute with Z-type (diagonal) gates when acting on the
# control qubit of a CNOT
_Z_COMMUTE_CONTROL = {'Z', 'S', 'T', 'RZ', 'Phase'}

# Self-commuting: any gate commutes with itself if it's diagonal
# CNOT commutes with CZ on control qubit
# Z-type on target commutes with CNOT if Z-type on control


def _gate_qubits(op: GateOp) -> set[int]:
    """All qubits touched by *op*."""
    return set(op.qubits) | set(op.controls)


def _commutes(op1: GateOp, op2: GateOp) -> bool:
    """Return True if *op1* and *op2* commute.

    This is a conservative check — returns False when unsure.
    """
    q1 = _gate_qubits(op1)
    q2 = _gate_qubits(op2)

    # Non-overlapping qubits always commute
    if not q1 & q2:
        return True

    # Both diagonal => commute
    if op1.name in _DIAGONAL_GATES and op2.name in _DIAGONAL_GATES:
        return True

    # Z-type gate on control of CNOT commutes
    if op1.name == 'CNOT' and op2.name in _Z_COMMUTE_CONTROL:
        ctrl = op1.qubits[0]
        if op2.qubits == (ctrl,) and op1.qubits[1] not in q2:
            return True
    if op2.name == 'CNOT' and op1.name in _Z_COMMUTE_CONTROL:
        ctrl = op2.qubits[0]
        if op1.qubits == (ctrl,) and op2.qubits[1] not in q1:
            return True

    # X commutes with CNOT on the target
    if op1.name == 'CNOT' and op2.name == 'X':
        tgt = op1.qubits[1]
        if op2.qubits == (tgt,) and op1.qubits[0] not in q2:
            return True
    if op2.name == 'CNOT' and op1.name == 'X':
        tgt = op2.qubits[1]
        if op1.qubits == (tgt,) and op2.qubits[0] not in q1:
            return True

    # RX commutes with X
    if op1.name == 'X' and op2.name == 'RX' and op1.qubits == op2.qubits:
        return True
    if op2.name == 'X' and op1.name == 'RX' and op1.qubits == op2.qubits:
        return True

    # RZ commutes with Z
    if op1.name == 'Z' and op2.name == 'RZ' and op1.qubits == op2.qubits:
        return True
    if op2.name == 'Z' and op1.name == 'RZ' and op1.qubits == op2.qubits:
        return True

    # Same single-qubit rotation axis commute
    if (op1.name == op2.name and op1.qubits == op2.qubits
            and op1.name in {'RX', 'RY', 'RZ', 'Phase'}):
        return True

    # H commutes with itself (it's self-inverse, and HH = I anyway)
    if op1.name == 'H' and op2.name == 'H' and op1.qubits == op2.qubits:
        return True

    return False


# ============================================================
# Pass
# ============================================================

class CommutationAnalysis(TranspilerPass):
    """Reorder commuting gates to cluster cancellable / mergeable operations.

    For each qubit wire, try to swap adjacent commuting gates so that
    same-type gates become neighbours (enabling CancelInverses and
    Optimize1qRotations to be more effective).
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        changed = True
        max_iterations = 50  # prevent infinite loops
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for q in range(dag.n_qubits):
                if self._reorder_wire(dag, q):
                    changed = True
        return dag

    def _reorder_wire(self, dag: DAGCircuit, qubit: int) -> bool:
        """Bubble-sort adjacent ops on *qubit* if they commute and swapping
        brings same-name gates closer together."""
        wire = dag.nodes_on_wire(qubit)
        if len(wire) < 2:
            return False

        changed = False
        # Single bubble pass
        for i in range(len(wire) - 1):
            n1 = wire[i]
            n2 = wire[i + 1]
            if n1.id not in dag._nodes or n2.id not in dag._nodes:
                continue

            op1, op2 = n1.op, n2.op

            # Check if swapping would be beneficial
            if not self._should_swap(op1, op2, wire, i):
                continue

            # Verify they commute
            if not _commutes(op1, op2):
                continue

            # For multi-qubit gates, ensure they are adjacent on ALL shared wires
            shared = _gate_qubits(op1) & _gate_qubits(op2)
            can_swap = True
            for qq in shared:
                wn = dag.nodes_on_wire(qq)
                ids = [nd.id for nd in wn]
                if n1.id in ids and n2.id in ids:
                    i1 = ids.index(n1.id)
                    i2 = ids.index(n2.id)
                    if i2 != i1 + 1:
                        can_swap = False
                        break
                else:
                    can_swap = False
                    break

            if not can_swap:
                continue

            # Perform the swap in wire lists
            self._swap_nodes(dag, n1.id, n2.id)
            changed = True

        return changed

    def _should_swap(self, op1: GateOp, op2: GateOp,
                     wire: list[DAGNode], idx: int) -> bool:
        """Heuristic: swap if op2 has the same name as op1's predecessor or
        op1 has the same name as op2's successor (clustering same-type gates)."""
        # Prefer grouping same-name gates
        if op1.name == op2.name:
            return False  # Already adjacent, no benefit

        # Check if op2 matches something before idx
        if idx > 0 and wire[idx - 1].op.name == op2.name:
            return True
        # Check if op1 matches something after idx+1
        if idx + 2 < len(wire) and wire[idx + 2].op.name == op1.name:
            return True

        return False

    def _swap_nodes(self, dag: DAGCircuit, id1: int, id2: int) -> None:
        """Swap the positions of two nodes in the DAG wire lists and edges."""
        # Swap in all wire lists
        node1 = dag._nodes[id1]
        node2 = dag._nodes[id2]
        all_qubits = _gate_qubits(node1.op) | _gate_qubits(node2.op)

        for q in all_qubits:
            wire = dag._wire_nodes.get(q, [])
            if id1 in wire and id2 in wire:
                i1 = wire.index(id1)
                i2 = wire.index(id2)
                wire[i1], wire[i2] = wire[i2], wire[i1]

        # Swap edges: id1's predecessors become id2's and vice versa
        # (only for the edge between them)
        preds1 = dag._predecessors.get(id1, [])
        succs1 = dag._successors.get(id1, [])
        preds2 = dag._predecessors.get(id2, [])
        succs2 = dag._successors.get(id2, [])

        # Remove mutual edges
        if id2 in succs1:
            dag._successors[id1] = [s for s in succs1 if s != id2]
        if id1 in preds2:
            dag._predecessors[id2] = [p for p in preds2 if p != id1]

        # Add reversed edge
        if id1 not in dag._successors.get(id2, []):
            dag._successors.setdefault(id2, []).append(id1)
        if id2 not in dag._predecessors.get(id1, []):
            dag._predecessors.setdefault(id1, []).append(id2)
