# -*- coding: utf-8 -*-
"""Directed Acyclic Graph representation of a quantum circuit.

Converts a linear ``Circuit`` (list of ``GateOp``) into a DAG where nodes are
gate operations and edges encode qubit-wire data dependencies.  This enables
analysis and rewriting passes that respect the partial ordering of gates.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterator

from qforge.ir import Circuit, GateOp


# ============================================================
# DAG node
# ============================================================

@dataclass
class DAGNode:
    """A single node in the DAG, wrapping a :class:`GateOp`."""
    id: int
    op: GateOp

    def __repr__(self) -> str:
        return f"DAGNode(id={self.id}, op={self.op.name}({self.op.qubits}))"


# ============================================================
# DAG circuit
# ============================================================

class DAGCircuit:
    """DAG-based representation of a quantum circuit.

    Each ``GateOp`` becomes a node.  Directed edges represent qubit-wire
    dependencies: an edge from *u* to *v* on qubit *q* means *u* must execute
    before *v* because both touch *q* and *u* comes first in program order.

    The implementation uses simple adjacency lists and requires no external
    graph library.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._nodes: dict[int, DAGNode] = {}
        # Adjacency lists
        self._successors: dict[int, list[int]] = defaultdict(list)
        self._predecessors: dict[int, list[int]] = defaultdict(list)
        # Per-wire ordered list of node ids
        self._wire_nodes: dict[int, list[int]] = defaultdict(list)
        self._next_id: int = 0

    # ---- construction ------------------------------------------------

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> DAGCircuit:
        """Build a :class:`DAGCircuit` from a :class:`Circuit`."""
        dag = cls(circuit.n_qubits)
        for op in circuit.ops:
            dag._append_op(op)
        return dag

    def _all_qubits(self, op: GateOp) -> set[int]:
        """Return every qubit index touched by *op* (qubits + controls)."""
        return set(op.qubits) | set(op.controls)

    def _append_op(self, op: GateOp) -> int:
        """Append *op* to the DAG and wire up dependency edges."""
        node_id = self._next_id
        self._next_id += 1
        node = DAGNode(id=node_id, op=op)
        self._nodes[node_id] = node

        for q in self._all_qubits(op):
            wire = self._wire_nodes[q]
            if wire:
                pred_id = wire[-1]
                # Avoid duplicate edges between the same pair
                if node_id not in self._successors[pred_id]:
                    self._successors[pred_id].append(node_id)
                if pred_id not in self._predecessors[node_id]:
                    self._predecessors[node_id].append(pred_id)
            wire.append(node_id)

        return node_id

    # ---- queries -----------------------------------------------------

    def nodes(self) -> list[DAGNode]:
        """Return all nodes in insertion order."""
        return list(self._nodes.values())

    def node(self, node_id: int) -> DAGNode:
        """Return the node with the given *node_id*."""
        return self._nodes[node_id]

    def nodes_on_wire(self, qubit: int) -> list[DAGNode]:
        """Return nodes acting on *qubit*, in wire order."""
        return [self._nodes[nid] for nid in self._wire_nodes.get(qubit, [])
                if nid in self._nodes]

    def predecessors(self, node_id: int) -> list[DAGNode]:
        """Return immediate predecessors of *node_id*."""
        return [self._nodes[pid] for pid in self._predecessors.get(node_id, [])
                if pid in self._nodes]

    def successors(self, node_id: int) -> list[DAGNode]:
        """Return immediate successors of *node_id*."""
        return [self._nodes[sid] for sid in self._successors.get(node_id, [])
                if sid in self._nodes]

    def num_nodes(self) -> int:
        return len(self._nodes)

    # ---- mutation ----------------------------------------------------

    def remove(self, node_id: int) -> None:
        """Remove *node_id* from the DAG and reconnect edges."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        preds = list(self._predecessors.get(node_id, []))
        succs = list(self._successors.get(node_id, []))

        # Re-wire: connect each predecessor to each successor on shared wires
        for pid in preds:
            if pid in self._successors:
                self._successors[pid] = [s for s in self._successors[pid] if s != node_id]
        for sid in succs:
            if sid in self._predecessors:
                self._predecessors[sid] = [p for p in self._predecessors[sid] if p != node_id]

        # For each qubit wire touched by this node, reconnect neighbours
        for q in self._all_qubits(node.op):
            wire = self._wire_nodes.get(q, [])
            if node_id in wire:
                idx = wire.index(node_id)
                pred_on_wire = wire[idx - 1] if idx > 0 else None
                succ_on_wire = wire[idx + 1] if idx < len(wire) - 1 else None
                wire.remove(node_id)
                if pred_on_wire is not None and succ_on_wire is not None:
                    if pred_on_wire in self._nodes and succ_on_wire in self._nodes:
                        if succ_on_wire not in self._successors.get(pred_on_wire, []):
                            self._successors[pred_on_wire].append(succ_on_wire)
                        if pred_on_wire not in self._predecessors.get(succ_on_wire, []):
                            self._predecessors[succ_on_wire].append(pred_on_wire)

        # Delete the node itself
        del self._nodes[node_id]
        self._successors.pop(node_id, None)
        self._predecessors.pop(node_id, None)

    def replace(self, node_id: int, ops: list[GateOp]) -> list[int]:
        """Replace *node_id* with a sequence of *ops*.

        The new ops are inserted into the DAG in order, inheriting the
        predecessor / successor relationships of the original node on a
        per-wire basis.

        Returns the ids of the newly inserted nodes.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not in DAG")

        old_node = self._nodes[node_id]
        old_preds = {pid for pid in self._predecessors.get(node_id, []) if pid in self._nodes}
        old_succs = {sid for sid in self._successors.get(node_id, []) if sid in self._nodes}

        # Disconnect old node from predecessors / successors
        for pid in old_preds:
            self._successors[pid] = [s for s in self._successors[pid] if s != node_id]
        for sid in old_succs:
            self._predecessors[sid] = [p for p in self._predecessors[sid] if p != node_id]

        # Remove old node from wire lists
        for q in self._all_qubits(old_node.op):
            wire = self._wire_nodes.get(q, [])
            if node_id in wire:
                wire.remove(node_id)

        del self._nodes[node_id]
        self._successors.pop(node_id, None)
        self._predecessors.pop(node_id, None)

        # Track the "last node placed on each wire" so we can build internal
        # edges among the replacement ops and connect to predecessors/successors.
        # Start with the old node's predecessors, per qubit.
        last_on_wire: dict[int, int | None] = {}
        for q in self._all_qubits(old_node.op):
            wire = self._wire_nodes.get(q, [])
            # Find the predecessor on this wire (the last id before insertion point)
            # The predecessor should be one of old_preds that is on this wire
            last_on_wire[q] = None
            for pid in reversed(wire):
                if pid in old_preds or pid in self._nodes:
                    # Find the last node on this wire that was a predecessor
                    last_on_wire[q] = pid if pid in old_preds else None
                    break
            # Actually find the tail of the wire correctly
            # After removal, the wire contains existing nodes. We need the
            # last node on this wire that was a predecessor of old_node.
            for pid in old_preds:
                if pid in self._nodes and q in self._all_qubits(self._nodes[pid].op):
                    last_on_wire[q] = pid

        new_ids: list[int] = []
        for op in ops:
            nid = self._next_id
            self._next_id += 1
            new_node = DAGNode(id=nid, op=op)
            self._nodes[nid] = new_node
            new_ids.append(nid)

            for q in self._all_qubits(op):
                # Find insertion point on the wire
                wire = self._wire_nodes.get(q, [])
                pred = last_on_wire.get(q)
                if pred is not None and pred in self._nodes:
                    # Insert after pred
                    if pred in self._nodes:
                        idx = wire.index(pred) + 1 if pred in wire else len(wire)
                    else:
                        idx = 0
                    wire.insert(idx, nid)
                    # Add edge pred -> nid
                    if nid not in self._successors.get(pred, []):
                        self._successors[pred].append(nid)
                    if pred not in self._predecessors.get(nid, []):
                        self._predecessors[nid].append(pred)
                else:
                    # Prepend
                    wire.insert(0, nid)

                last_on_wire[q] = nid

        # Connect last placed nodes to old successors
        for sid in old_succs:
            if sid not in self._nodes:
                continue
            succ_node = self._nodes[sid]
            for q in self._all_qubits(succ_node.op):
                last = last_on_wire.get(q)
                if last is not None and last in self._nodes:
                    if sid not in self._successors.get(last, []):
                        self._successors[last].append(sid)
                    if last not in self._predecessors.get(sid, []):
                        self._predecessors[sid].append(last)

        return new_ids

    # ---- conversion --------------------------------------------------

    def topological_ops(self) -> list[GateOp]:
        """Return ops in topological (dependency-respecting) order."""
        return [node.op for node in self._topological_sort()]

    def to_circuit(self) -> Circuit:
        """Convert back to a :class:`Circuit`."""
        circ = Circuit(self.n_qubits)
        circ.ops = self.topological_ops()
        return circ

    def _topological_sort(self) -> list[DAGNode]:
        """Kahn's algorithm for topological sorting.

        Ties are broken by node ID (insertion order) to ensure a stable,
        deterministic output that preserves the original program order for
        independent operations.
        """
        import heapq

        in_degree: dict[int, int] = {}
        for nid in self._nodes:
            in_degree[nid] = len([p for p in self._predecessors.get(nid, [])
                                  if p in self._nodes])

        heap: list[int] = []
        for nid, deg in in_degree.items():
            if deg == 0:
                heapq.heappush(heap, nid)

        result: list[DAGNode] = []
        while heap:
            nid = heapq.heappop(heap)
            result.append(self._nodes[nid])
            for sid in self._successors.get(nid, []):
                if sid in in_degree:
                    in_degree[sid] -= 1
                    if in_degree[sid] == 0:
                        heapq.heappush(heap, sid)

        if len(result) != len(self._nodes):
            raise RuntimeError("DAG contains a cycle — topological sort failed")
        return result

    # ---- utilities ---------------------------------------------------

    def count_ops(self) -> dict[str, int]:
        """Return a gate-name -> count mapping."""
        counts: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            counts[node.op.name] += 1
        return dict(counts)

    def depth(self) -> int:
        """Longest path length in the DAG (circuit depth)."""
        if not self._nodes:
            return 0
        dist: dict[int, int] = {}
        for node in self._topological_sort():
            preds = [p for p in self._predecessors.get(node.id, []) if p in dist]
            dist[node.id] = (max(dist[p] for p in preds) + 1) if preds else 1
        return max(dist.values()) if dist else 0

    def __repr__(self) -> str:
        return (f"DAGCircuit(n_qubits={self.n_qubits}, "
                f"nodes={len(self._nodes)}, depth={self.depth()})")
