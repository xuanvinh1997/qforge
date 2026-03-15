# -*- coding: utf-8 -*-
"""Core benchmark infrastructure: config, timing, base class, framework loaders."""
from __future__ import annotations

import sys
import time
import subprocess
import tracemalloc
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """Global benchmark configuration."""
    n_warmup: int = 2
    n_runs: int = 5
    seed: int = 42
    max_qubits: int = 20
    shift: float = np.pi / 2
    lr: float = 0.1
    steps: int = 50
    output_dir: str = "benchmark_results"
    frameworks: List[str] = field(default_factory=lambda: ["qforge", "pennylane", "qiskit"])


BenchResult = namedtuple("BenchResult", ["median", "min", "max", "result"])

# ---------------------------------------------------------------------------
# Timing & memory utilities
# ---------------------------------------------------------------------------

def bench(fn: Callable, n_warmup: int = 2, n_runs: int = 5) -> BenchResult:
    """Run *fn()* with warmup, return BenchResult(median, min, max, last_result)."""
    result = None
    for _ in range(n_warmup):
        result = fn()
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return BenchResult(
        median=float(np.median(times)),
        min=times[0],
        max=times[-1],
        result=result,
    )


def measure_memory(fn: Callable) -> float:
    """Run *fn()* and return peak memory in MB (Python-side only)."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Framework loaders (cached at module level)
# ---------------------------------------------------------------------------

_qml = None
_has_qiskit = False


def _pip_install(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def load_pennylane():
    """Import PennyLane, installing if needed. Returns module or None."""
    global _qml
    if _qml is not None:
        return _qml
    try:
        import pennylane as qml
        _qml = qml
        return qml
    except ImportError:
        pass
    try:
        print("  Installing PennyLane ...", end=" ", flush=True)
        _pip_install("pennylane")
        import pennylane as qml
        _qml = qml
        print("done")
        return qml
    except Exception as e:
        print(f"failed ({e})")
        return None


def load_qiskit() -> bool:
    """Import Qiskit, installing if needed. Returns True/False."""
    global _has_qiskit
    if _has_qiskit:
        return True
    try:
        import qiskit  # noqa: F401
        _has_qiskit = True
        return True
    except ImportError:
        pass
    try:
        print("  Installing Qiskit ...", end=" ", flush=True)
        _pip_install("qiskit")
        import qiskit  # noqa: F401
        _has_qiskit = True
        print("done")
        return True
    except Exception as e:
        print(f"failed ({e})")
        return False


def get_pennylane():
    """Return cached PennyLane module or None."""
    return _qml


def get_qiskit_available() -> bool:
    """Return whether Qiskit is available."""
    return _has_qiskit


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

_WIDTH = 72


def section(title: str, desc: str = "") -> None:
    print("\n" + "=" * _WIDTH)
    print(f"  {title}")
    if desc:
        print(f"  {desc}")
    print("=" * _WIDTH)


def table(headers: List[str], rows: List[list], col_widths: Optional[List[int]] = None) -> None:
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for r in rows:
                if i < len(r):
                    w = max(w, len(str(r[i])))
            col_widths.append(w + 2)
    fmt = "  ".join(
        f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}"
        for i, w in enumerate(col_widths)
    )
    print(fmt.format(*headers))
    print("-" * _WIDTH)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


# ---------------------------------------------------------------------------
# Base benchmark suite
# ---------------------------------------------------------------------------

class BaseBenchmarkSuite:
    """Base class for all benchmark suites.

    Subclasses must set *name* and *description* and implement *run()*.
    """
    name: str = ""
    description: str = ""

    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self._results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute all benchmarks in this suite. Returns results dict."""
        raise NotImplementedError

    def _store(self, key: str, data: dict) -> None:
        self._results[key] = data

    @property
    def results(self) -> Dict[str, Any]:
        return self._results

    def print_table(self) -> None:
        """Override to print text summary after run()."""
        pass
