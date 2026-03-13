# -*- coding: utf-8 -*-
"""Tests for the qforge transpiler module."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qforge.ir import Circuit, GateOp
from qforge.transpiler.dag import DAGCircuit, DAGNode
from qforge.transpiler.pass_manager import PassManager, TranspilerPass
from qforge.transpiler.passes.decompose import Decompose
from qforge.transpiler.passes.cancel import CancelInverses
from qforge.transpiler.passes.optimize_1q import Optimize1qRotations
from qforge.transpiler.passes.commute import CommutationAnalysis
from qforge.transpiler.passes.basis import BasisTranslator


# ============================================================
# Helper: simulate a circuit and return the statevector
# ============================================================

def _simulate(circuit: Circuit) -> np.ndarray:
    """Simulate a circuit by matrix multiplication and return the statevector."""
    n = circuit.n_qubits
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |00...0>

    for op in circuit.ops:
        mat = _gate_matrix(op, n)
        state = mat @ state
    return state


def _gate_matrix(op: GateOp, n_qubits: int) -> np.ndarray:
    """Build the full 2^n x 2^n matrix for a GateOp."""
    dim = 2 ** n_qubits
    local = _local_matrix(op)
    if local is None:
        return np.eye(dim, dtype=complex)

    qubits = list(op.qubits)
    return _embed_matrix(local, qubits, n_qubits)


def _local_matrix(op: GateOp) -> np.ndarray | None:
    """Return the local unitary matrix for a gate."""
    name = op.name
    sq2 = 1 / math.sqrt(2)

    if name == 'H':
        return np.array([[sq2, sq2], [sq2, -sq2]], dtype=complex)
    if name == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == 'Y':
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == 'S':
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == 'T':
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
    if name == 'Xsquare':
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    if name == 'RX':
        t = op.params[0] / 2
        return np.array([[math.cos(t), -1j * math.sin(t)],
                         [-1j * math.sin(t), math.cos(t)]], dtype=complex)
    if name == 'RY':
        t = op.params[0] / 2
        return np.array([[math.cos(t), -math.sin(t)],
                         [math.sin(t), math.cos(t)]], dtype=complex)
    if name == 'RZ':
        t = op.params[0] / 2
        return np.array([[np.exp(-1j * t), 0],
                         [0, np.exp(1j * t)]], dtype=complex)
    if name == 'Phase':
        phi = op.params[0]
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    if name == 'CNOT':
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    if name == 'SWAP':
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    if name == 'CCNOT':
        m = np.eye(8, dtype=complex)
        m[6, 6] = 0; m[7, 7] = 0; m[6, 7] = 1; m[7, 6] = 1
        return m
    if name == 'CSWAP':
        m = np.eye(8, dtype=complex)
        m[5, 5] = 0; m[6, 6] = 0; m[5, 6] = 1; m[6, 5] = 1
        return m
    return None


def _embed_matrix(local: np.ndarray, qubits: list[int], n_qubits: int) -> np.ndarray:
    """Embed a local gate matrix into the full 2^n Hilbert space.

    Uses the standard tensor-product embedding with qubit ordering
    where qubit 0 is the most significant bit.
    """
    dim = 2 ** n_qubits
    n_gate = len(qubits)
    dim_gate = 2 ** n_gate
    result = np.zeros((dim, dim), dtype=complex)

    for col in range(dim):
        for row in range(dim):
            # Extract the bits for the gate qubits
            col_bits = [(col >> (n_qubits - 1 - q)) & 1 for q in qubits]
            row_bits = [(row >> (n_qubits - 1 - q)) & 1 for q in qubits]

            # The non-gate qubits must match
            match = True
            for q in range(n_qubits):
                if q not in qubits:
                    if ((col >> (n_qubits - 1 - q)) & 1) != ((row >> (n_qubits - 1 - q)) & 1):
                        match = False
                        break
            if not match:
                continue

            # Compute the gate matrix element
            col_idx = sum(b << (n_gate - 1 - i) for i, b in enumerate(col_bits))
            row_idx = sum(b << (n_gate - 1 - i) for i, b in enumerate(row_bits))
            result[row, col] = local[row_idx, col_idx]

    return result


def _states_equivalent(s1: np.ndarray, s2: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if two state vectors are equal up to global phase."""
    if s1.shape != s2.shape:
        return False
    # Find first nonzero element to determine phase
    for i in range(len(s1)):
        if abs(s1[i]) > tol:
            if abs(s2[i]) < tol:
                return False
            phase = s2[i] / s1[i]
            return np.allclose(s1 * phase, s2, atol=tol)
    # s1 is all zeros
    return np.allclose(s2, 0, atol=tol)


# ============================================================
# Tests: DAG round-trip
# ============================================================

class TestDAGCircuit:
    def test_roundtrip_preserves_ops(self):
        """Circuit -> DAG -> Circuit preserves all ops."""
        c = Circuit(3)
        c.h(0).cnot(0, 1).rx(2, 0.5).cnot(1, 2)

        dag = DAGCircuit.from_circuit(c)
        c2 = dag.to_circuit()

        assert len(c2.ops) == len(c.ops)
        for orig, restored in zip(c.ops, c2.ops):
            assert orig.name == restored.name
            assert orig.qubits == restored.qubits
            assert orig.params == restored.params

    def test_roundtrip_single_qubit(self):
        c = Circuit(1)
        c.h(0).x(0).rz(0, 1.0)
        dag = DAGCircuit.from_circuit(c)
        c2 = dag.to_circuit()
        assert len(c2.ops) == 3
        assert c2.ops[0].name == 'H'
        assert c2.ops[1].name == 'X'
        assert c2.ops[2].name == 'RZ'

    def test_nodes_on_wire(self):
        c = Circuit(2)
        c.h(0).cnot(0, 1).x(1)
        dag = DAGCircuit.from_circuit(c)

        wire0 = dag.nodes_on_wire(0)
        assert [n.op.name for n in wire0] == ['H', 'CNOT']

        wire1 = dag.nodes_on_wire(1)
        assert [n.op.name for n in wire1] == ['CNOT', 'X']

    def test_remove_node(self):
        c = Circuit(1)
        c.h(0).x(0).z(0)
        dag = DAGCircuit.from_circuit(c)
        nodes = dag.nodes()
        assert len(nodes) == 3

        dag.remove(nodes[1].id)  # remove X
        c2 = dag.to_circuit()
        assert len(c2.ops) == 2
        assert c2.ops[0].name == 'H'
        assert c2.ops[1].name == 'Z'

    def test_replace_node(self):
        c = Circuit(1)
        c.h(0)
        dag = DAGCircuit.from_circuit(c)
        nid = dag.nodes()[0].id
        dag.replace(nid, [
            GateOp(name='RZ', qubits=(0,), params=(math.pi / 2,)),
            GateOp(name='Xsquare', qubits=(0,)),
            GateOp(name='RZ', qubits=(0,), params=(math.pi / 2,)),
        ])
        c2 = dag.to_circuit()
        assert len(c2.ops) == 3
        assert c2.ops[0].name == 'RZ'
        assert c2.ops[1].name == 'Xsquare'
        assert c2.ops[2].name == 'RZ'

    def test_predecessors_successors(self):
        c = Circuit(2)
        c.h(0).cnot(0, 1).x(1)
        dag = DAGCircuit.from_circuit(c)
        nodes = dag.nodes()
        # CNOT is successor of H
        cnot = [n for n in nodes if n.op.name == 'CNOT'][0]
        preds = dag.predecessors(cnot.id)
        assert any(p.op.name == 'H' for p in preds)
        succs = dag.successors(cnot.id)
        assert any(s.op.name == 'X' for s in succs)

    def test_count_ops(self):
        c = Circuit(2)
        c.h(0).h(1).cnot(0, 1)
        dag = DAGCircuit.from_circuit(c)
        counts = dag.count_ops()
        assert counts['H'] == 2
        assert counts['CNOT'] == 1

    def test_depth(self):
        c = Circuit(2)
        c.h(0).h(1).cnot(0, 1)
        dag = DAGCircuit.from_circuit(c)
        assert dag.depth() == 2  # H parallel, then CNOT


# ============================================================
# Tests: Decompose pass
# ============================================================

class TestDecompose:
    def test_ccnot_decomposed(self):
        """CCNOT should be decomposed into elementary gates."""
        c = Circuit(3)
        c.ccnot(0, 1, 2)
        pm = PassManager([Decompose()])
        c2 = pm.run(c)

        gate_names = {op.name for op in c2.ops}
        assert 'CCNOT' not in gate_names
        # Should contain CNOT and single-qubit gates
        assert 'CNOT' in gate_names

    def test_cswap_decomposed(self):
        c = Circuit(3)
        c.cswap(0, 1, 2)
        pm = PassManager([Decompose()])
        c2 = pm.run(c)
        gate_names = {op.name for op in c2.ops}
        assert 'CSWAP' not in gate_names
        assert 'CNOT' in gate_names

    def test_elementary_gates_unchanged(self):
        c = Circuit(2)
        c.h(0).cnot(0, 1).rx(1, 0.5)
        pm = PassManager([Decompose()])
        c2 = pm.run(c)
        assert len(c2.ops) == len(c.ops)

    def test_mcx_decomposed(self):
        """MCX with 2 controls should decompose."""
        c = Circuit(3)
        c.ops.append(GateOp(name='MCX', qubits=(2,), controls=(0, 1)))
        pm = PassManager([Decompose()])
        c2 = pm.run(c)
        gate_names = {op.name for op in c2.ops}
        assert 'MCX' not in gate_names


# ============================================================
# Tests: CancelInverses pass
# ============================================================

class TestCancelInverses:
    def test_hh_cancelled(self):
        """H followed by H should cancel."""
        c = Circuit(1)
        c.h(0).h(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_xx_cancelled(self):
        c = Circuit(1)
        c.x(0).x(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_cnot_cnot_cancelled(self):
        c = Circuit(2)
        c.cnot(0, 1).cnot(0, 1)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_rx_inverse_cancelled(self):
        c = Circuit(1)
        c.rx(0, 0.5).rx(0, -0.5)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_non_inverse_not_cancelled(self):
        c = Circuit(1)
        c.h(0).x(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 2

    def test_triple_h_leaves_one(self):
        """H H H -> one H remains."""
        c = Circuit(1)
        c.h(0).h(0).h(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)
        assert len(c2.ops) == 1
        assert c2.ops[0].name == 'H'


# ============================================================
# Tests: Optimize1qRotations pass
# ============================================================

class TestOptimize1qRotations:
    def test_merge_rx(self):
        """RX(a) + RX(b) should become RX(a+b)."""
        c = Circuit(1)
        c.rx(0, 0.3).rx(0, 0.7)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)
        assert len(c2.ops) == 1
        assert c2.ops[0].name == 'RX'
        assert abs(c2.ops[0].params[0] - 1.0) < 1e-9

    def test_merge_rz(self):
        c = Circuit(1)
        c.rz(0, 0.1).rz(0, 0.2).rz(0, 0.3)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)
        assert len(c2.ops) == 1
        assert abs(c2.ops[0].params[0] - 0.6) < 1e-9

    def test_zero_rotation_removed(self):
        c = Circuit(1)
        c.rx(0, 0.0)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_2pi_rotation_removed(self):
        c = Circuit(1)
        c.rz(0, 2 * math.pi)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)
        assert len(c2.ops) == 0

    def test_different_axes_not_merged(self):
        c = Circuit(1)
        c.rx(0, 0.5).rz(0, 0.5)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)
        assert len(c2.ops) == 2


# ============================================================
# Tests: BasisTranslator pass
# ============================================================

class TestBasisTranslator:
    def test_target_basis_only(self):
        """After basis translation, only target basis gates remain."""
        c = Circuit(2)
        c.h(0).s(0).cnot(0, 1).t(1)
        pm = PassManager([BasisTranslator()])
        c2 = pm.run(c)
        target = {'CNOT', 'RZ', 'Xsquare', 'X'}
        for op in c2.ops:
            assert op.name in target, f"Unexpected gate: {op.name}"

    def test_h_decomposed(self):
        c = Circuit(1)
        c.h(0)
        pm = PassManager([BasisTranslator()])
        c2 = pm.run(c)
        assert all(op.name in {'RZ', 'Xsquare', 'X'} for op in c2.ops)

    def test_cnot_preserved(self):
        c = Circuit(2)
        c.cnot(0, 1)
        pm = PassManager([BasisTranslator()])
        c2 = pm.run(c)
        assert len(c2.ops) == 1
        assert c2.ops[0].name == 'CNOT'


# ============================================================
# Tests: PassManager preset levels
# ============================================================

class TestPassManager:
    def test_preset_level_0(self):
        pm = PassManager.preset(0)
        assert len(pm.passes) == 0

    def test_preset_level_1(self):
        pm = PassManager.preset(1)
        assert len(pm.passes) == 2

    def test_preset_level_2(self):
        pm = PassManager.preset(2)
        assert len(pm.passes) == 4

    def test_preset_level_3(self):
        pm = PassManager.preset(3)
        assert len(pm.passes) >= 5

    def test_preset_runs(self):
        """All preset levels should run without error."""
        c = Circuit(2)
        c.h(0).cnot(0, 1).rx(1, 0.5)
        for level in range(4):
            pm = PassManager.preset(level)
            c2 = pm.run(c)
            assert c2.n_qubits == 2

    def test_custom_pass(self):
        """A custom pass can be added."""
        class NoopPass(TranspilerPass):
            def run(self, dag):
                return dag

        pm = PassManager([NoopPass()])
        c = Circuit(1)
        c.h(0)
        c2 = pm.run(c)
        assert len(c2.ops) == 1


# ============================================================
# Tests: Circuit equivalence via statevector simulation
# ============================================================

class TestCircuitEquivalence:
    def test_hh_identity(self):
        """H·H on a qubit should be identity."""
        c = Circuit(1)
        c.h(0).h(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)

        s_orig = _simulate(c)
        s_opt = _simulate(c2)
        assert _states_equivalent(s_orig, s_opt)

    def test_rx_merge_equivalence(self):
        """Merging RX(a)+RX(b) should give same state."""
        c = Circuit(1)
        c.rx(0, 0.3).rx(0, 0.7)
        pm = PassManager([Optimize1qRotations()])
        c2 = pm.run(c)

        s_orig = _simulate(c)
        s_opt = _simulate(c2)
        assert _states_equivalent(s_orig, s_opt)

    def test_cancel_preserves_state(self):
        """Cancel pass on H X X H should give identity."""
        c = Circuit(1)
        c.h(0).x(0).x(0).h(0)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)

        s_orig = _simulate(c)
        s_opt = _simulate(c2)
        assert _states_equivalent(s_orig, s_opt)

    def test_cnot_cancel_preserves_state(self):
        c = Circuit(2)
        c.h(0).cnot(0, 1).cnot(0, 1)
        pm = PassManager([CancelInverses()])
        c2 = pm.run(c)

        s_orig = _simulate(c)
        s_opt = _simulate(c2)
        assert _states_equivalent(s_orig, s_opt)
