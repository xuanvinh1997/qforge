# -*- coding: utf-8 -*-
"""Qforge benchmark framework — modular, extensible, with chart generation."""
from __future__ import annotations

from qforge.benchmarks.core import BenchConfig, BenchResult, BaseBenchmarkSuite, bench, measure_memory
from qforge.benchmarks.results import BenchmarkResults

__all__ = [
    "BenchConfig",
    "BenchResult",
    "BaseBenchmarkSuite",
    "BenchmarkResults",
    "bench",
    "measure_memory",
    "run_benchmarks",
]


def run_benchmarks(
    suites: list[str] | None = None,
    config: BenchConfig | None = None,
    charts: bool = True,
) -> BenchmarkResults:
    """Run selected benchmark suites and optionally generate charts/report.

    Args:
        suites:  List of suite names, or None for all.
        config:  Benchmark configuration. Uses defaults if None.
        charts:  Whether to generate matplotlib charts and HTML report.

    Returns:
        BenchmarkResults with all collected data.
    """
    from qforge.benchmarks.suites import ALL_SUITES
    from qforge.benchmarks.core import load_pennylane, load_qiskit, section
    from pathlib import Path

    if config is None:
        config = BenchConfig()

    # Load external frameworks
    print("\n  Loading external frameworks ...")
    qml = load_pennylane()
    has_qk = load_qiskit()
    if qml:
        print(f"    PennyLane v{qml.__version__}")
    if has_qk:
        import qiskit
        print(f"    Qiskit v{qiskit.__version__}")

    # Select suites
    if suites is None or "all" in suites:
        selected = ALL_SUITES
    else:
        selected = {k: v for k, v in ALL_SUITES.items() if k in suites}

    # Run each suite
    all_suite_results = {}
    for name, cls in selected.items():
        try:
            suite = cls(config)
            suite.run()
            all_suite_results[name] = suite.results
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")

    # Package results
    results = BenchmarkResults.create(config, all_suite_results)

    # Generate charts + report
    if charts:
        try:
            from qforge.benchmarks.report import generate_report
            output_dir = Path(config.output_dir)
            generate_report(results, output_dir)
            print(f"\n  Report generated at {output_dir}/report.html")
        except ImportError:
            print("\n  matplotlib not installed — skipping chart generation.")
        except Exception as e:
            print(f"\n  Chart generation failed: {e}")

    return results
