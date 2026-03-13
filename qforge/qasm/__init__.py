# -*- coding: utf-8 -*-
# author: vinhpx
"""qforge.qasm --- OpenQASM import/export.

Provides round-trip conversion between qforge circuits and OpenQASM 2.0/3.0.
"""
from __future__ import annotations

from qforge.qasm.exporter import circuit_to_qasm2, circuit_to_qasm3
from qforge.qasm.importer import qasm2_to_circuit, qasm3_to_circuit

__all__ = [
    "circuit_to_qasm2",
    "circuit_to_qasm3",
    "qasm2_to_circuit",
    "qasm3_to_circuit",
]
