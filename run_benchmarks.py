#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qforge Benchmark Suite — CLI Entry Point
=========================================

Comprehensive benchmarks: Qforge vs PennyLane vs Qiskit with chart generation.

Usage:
    python run_benchmarks.py                                # all suites
    python run_benchmarks.py --suites gates scaling mps     # select suites
    python run_benchmarks.py --max-qubits 16 --runs 3       # configure
    python run_benchmarks.py --output benchmark_results/    # output dir
    python run_benchmarks.py --no-charts                    # text tables only
    python run_benchmarks.py --from-json results.json       # regenerate charts from saved data
    python run_benchmarks.py --list                         # list available suites

Available suites:
    gates, circuits, vqe, qaoa, gradient, measurement,
    scaling, accuracy, memory, mps, dmrg
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Qforge Benchmark Suite: Qforge vs PennyLane vs Qiskit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suites", nargs="*", default=["all"],
        help="Benchmark suites to run (default: all). Use --list to see options.",
    )
    parser.add_argument("--max-qubits", type=int, default=20,
                        help="Maximum qubit count (default: 20)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of benchmark runs for median (default: 5)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Optimization steps for VQE/QAOA (default: 50)")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory for charts and report (default: benchmark_results)")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to specific JSON path")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation (text tables only)")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Regenerate charts from previously saved JSON results")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmark suites and exit")
    args = parser.parse_args()

    # List suites
    if args.list:
        from qforge.benchmarks.suites import ALL_SUITES
        print("\nAvailable benchmark suites:")
        print("-" * 50)
        for name, cls in ALL_SUITES.items():
            print(f"  {name:15s}  {cls.description}")
        print()
        return

    # Regenerate charts from JSON
    if args.from_json:
        from qforge.benchmarks.results import BenchmarkResults
        from qforge.benchmarks.report import generate_report

        print(f"Loading results from {args.from_json} ...")
        results = BenchmarkResults.load_json(Path(args.from_json))
        output_dir = Path(args.output)
        generate_report(results, output_dir)
        print(f"Report regenerated at {output_dir}/report.html")
        return

    # Run benchmarks
    from qforge.benchmarks.core import BenchConfig, section
    from qforge.benchmarks import run_benchmarks

    config = BenchConfig(
        n_runs=args.runs,
        max_qubits=args.max_qubits,
        steps=args.steps,
        output_dir=args.output,
    )

    # Header
    print("=" * 72)
    print("  QFORGE BENCHMARK SUITE")
    print("  Comprehensive Comparison: Qforge vs PennyLane vs Qiskit")
    print("=" * 72)

    from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
    print(f"  Qforge backends — CPU:{_HAS_CPP}  CUDA:{_HAS_CUDA}  Metal:{_HAS_METAL}")
    print(f"  Max qubits: {config.max_qubits}  |  Runs: {config.n_runs}  |  Steps: {config.steps}")
    print(f"  Suites: {', '.join(args.suites)}")

    results = run_benchmarks(
        suites=args.suites,
        config=config,
        charts=not args.no_charts,
    )

    # Optional extra JSON export
    if args.json:
        results.save_json(Path(args.json))
        print(f"\n  Results exported to {args.json}")

    # Summary
    section("SUMMARY", "")
    _print_summary(results)
    print("\n  Done.")


def _print_summary(results):
    """Print a summary of which framework was fastest."""
    from qforge.benchmarks.report import _compute_summary
    summary = _compute_summary(results)

    print(f"  Suites run: {summary['n_suites']}")
    print(f"  Total tests: {summary['n_tests']}")
    if summary['n_comparable'] > 0:
        pct = summary['qf_wins'] / summary['n_comparable'] * 100
        print(f"  Qforge fastest: {summary['qf_wins']}/{summary['n_comparable']} ({pct:.0f}%)")

    for suite_name, suite_data in results.suites.items():
        qf_wins = 0
        total = 0
        for test_data in suite_data.values():
            if not isinstance(test_data, dict):
                continue
            qf_t = test_data.get("qforge") or test_data.get("time")
            pl_t = test_data.get("pennylane")
            qk_t = test_data.get("qiskit")
            times = {k: v for k, v in [("qforge", qf_t), ("pennylane", pl_t), ("qiskit", qk_t)]
                     if v is not None and isinstance(v, (int, float))}
            if len(times) > 1:
                total += 1
                if min(times, key=times.get) == "qforge":
                    qf_wins += 1
        if total > 0:
            print(f"    {suite_name:15s}  Qforge fastest in {qf_wins}/{total} tests")


if __name__ == "__main__":
    main()
