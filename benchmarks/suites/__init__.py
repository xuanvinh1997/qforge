# -*- coding: utf-8 -*-
"""Registry of benchmark suites."""
from __future__ import annotations

from benchmarks.suites.correctness import CorrectnessSuite
from benchmarks.suites.gate_perf import GatePerfSuite
from benchmarks.suites.framework_compare import FrameworkCompareSuite
from benchmarks.suites.backend_compare import BackendCompareSuite
from benchmarks.suites.qml_algos import QMLAlgosSuite
from benchmarks.suites.gradient import GradientSuite
from benchmarks.suites.vqe import VQESuite

ALL_SUITES = {
    "correctness": CorrectnessSuite,
    "gate_perf": GatePerfSuite,
    "framework": FrameworkCompareSuite,
    "backend": BackendCompareSuite,
    "qml": QMLAlgosSuite,
    "gradient": GradientSuite,
    "vqe": VQESuite,
}
