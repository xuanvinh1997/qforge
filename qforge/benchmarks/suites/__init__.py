# -*- coding: utf-8 -*-
"""Registry of all built-in benchmark suites."""
from __future__ import annotations

from qforge.benchmarks.suites.gates import GatesBenchmarkSuite
from qforge.benchmarks.suites.circuits import CircuitsBenchmarkSuite
from qforge.benchmarks.suites.vqe import VQEBenchmarkSuite
from qforge.benchmarks.suites.qaoa import QAOABenchmarkSuite
from qforge.benchmarks.suites.gradient import GradientBenchmarkSuite
from qforge.benchmarks.suites.measurement import MeasurementBenchmarkSuite
from qforge.benchmarks.suites.scaling import ScalingBenchmarkSuite
from qforge.benchmarks.suites.accuracy import AccuracyBenchmarkSuite
from qforge.benchmarks.suites.memory import MemoryBenchmarkSuite
from qforge.benchmarks.suites.mps import MPSBenchmarkSuite
from qforge.benchmarks.suites.dmrg import DMRGBenchmarkSuite

ALL_SUITES = {
    "gates": GatesBenchmarkSuite,
    "circuits": CircuitsBenchmarkSuite,
    "vqe": VQEBenchmarkSuite,
    "qaoa": QAOABenchmarkSuite,
    "gradient": GradientBenchmarkSuite,
    "measurement": MeasurementBenchmarkSuite,
    "scaling": ScalingBenchmarkSuite,
    "accuracy": AccuracyBenchmarkSuite,
    "memory": MemoryBenchmarkSuite,
    "mps": MPSBenchmarkSuite,
    "dmrg": DMRGBenchmarkSuite,
}
