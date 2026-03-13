# -*- coding: utf-8 -*-
"""Transpiler pass manager and base pass class.

The :class:`PassManager` converts a :class:`Circuit` to a DAG, runs a
sequence of :class:`TranspilerPass` instances, and converts back.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from qforge.ir import Circuit
from qforge.transpiler.dag import DAGCircuit


# ============================================================
# Base pass
# ============================================================

class TranspilerPass(ABC):
    """Abstract base class for all transpiler passes."""

    @abstractmethod
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute this pass on *dag* and return the (possibly modified) DAG."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============================================================
# Pass manager
# ============================================================

class PassManager:
    """Orchestrates a sequence of :class:`TranspilerPass` runs.

    Usage::

        pm = PassManager([Decompose(), CancelInverses()])
        optimized = pm.run(circuit)
    """

    def __init__(self, passes: list[TranspilerPass] | None = None) -> None:
        self.passes: list[TranspilerPass] = passes or []

    def append(self, p: TranspilerPass) -> None:
        """Add a pass to the end of the pipeline."""
        self.passes.append(p)

    def run(self, circuit: Circuit) -> Circuit:
        """Convert *circuit* to DAG, run all passes, convert back."""
        dag = DAGCircuit.from_circuit(circuit)
        for p in self.passes:
            dag = p.run(dag)
        return dag.to_circuit()

    @classmethod
    def preset(cls, level: int = 1) -> PassManager:
        """Return a preset :class:`PassManager` for a given optimization level.

        Levels:
            0 — No optimisation (empty pipeline).
            1 — Decompose + cancel inverses.
            2 — Level 1 + 1q rotation merging + basis translation.
            3 — Level 2 + commutation analysis (repeat cancel/merge).
        """
        from qforge.transpiler.passes.decompose import Decompose
        from qforge.transpiler.passes.cancel import CancelInverses
        from qforge.transpiler.passes.optimize_1q import Optimize1qRotations
        from qforge.transpiler.passes.basis import BasisTranslator
        from qforge.transpiler.passes.commute import CommutationAnalysis

        if level <= 0:
            return cls([])
        if level == 1:
            return cls([Decompose(), CancelInverses()])
        if level == 2:
            return cls([
                Decompose(),
                CancelInverses(),
                Optimize1qRotations(),
                BasisTranslator(),
            ])
        # level >= 3
        return cls([
            Decompose(),
            CancelInverses(),
            CommutationAnalysis(),
            CancelInverses(),
            Optimize1qRotations(),
            BasisTranslator(),
        ])

    def __repr__(self) -> str:
        names = [p.name for p in self.passes]
        return f"PassManager({names})"
