#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qforge Benchmark Runner — version-aware, organized results.

Usage:
    python -m benchmarks.run                           # all suites, auto-detect backends
    python -m benchmarks.run --suites gate_perf vqe    # select suites
    python -m benchmarks.run --backends cpu metal       # select backends
    python -m benchmarks.run --tag experiment1          # tag this run
    python -m benchmarks.run --list                    # list suites
    python -m benchmarks.run --history                 # show past runs
    python -m benchmarks.run --compare v3.0.0 v3.1.0   # compare versions

Available suites:
    correctness     Gate/measurement/entropy correctness tests
    gate_perf       Gate-level circuit and single-gate timing
    framework       Qforge vs Qiskit vs PennyLane
    backend         CPU vs Metal vs CUDA comparison
    qml             QML classifier accuracy & training time
    gradient        Parameter-shift gradient scaling
    vqe             VQE/QAOA convergence and timing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `python -m benchmarks.run` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Qforge Benchmark Runner (version-managed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--suites", nargs="*", default=None,
                        help="Suites to run (default: all). Use --list to see options.")
    parser.add_argument("--backends", nargs="*", default=None,
                        help="Backends to benchmark: cpu, metal, cuda (auto-detect if omitted)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for this benchmark run (appears in directory name)")
    parser.add_argument("--list", action="store_true",
                        help="List available suites")
    parser.add_argument("--history", action="store_true",
                        help="Show past benchmark runs")
    parser.add_argument("--compare", nargs=2, metavar=("V1", "V2"),
                        help="Compare two versions (e.g., --compare v3.0.0 v3.1.0)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")

    # Config overrides
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-qubits", type=int, default=22)

    args = parser.parse_args()

    # --list
    if args.list:
        from benchmarks.suites import ALL_SUITES
        print("\nAvailable benchmark suites:")
        print("-" * 55)
        for name, cls in ALL_SUITES.items():
            print(f"  {name:18s} {cls.description}")
        print()
        return

    # --history
    if args.history:
        from benchmarks.version import list_versions, list_runs
        versions = list_versions()
        if not versions:
            print("No benchmark results found.")
            return
        print(f"\nBenchmark history ({len(versions)} versions):")
        print("=" * 70)
        for v in versions:
            runs = list_runs(v)
            print(f"\n  {v} ({len(runs)} runs)")
            for r in runs:
                suites_str = ", ".join(r["suites"]) if r["suites"] else "?"
                print(f"    {r['name']:30s}  [{suites_str}]")
        print()
        return

    # --compare
    if args.compare:
        from benchmarks.compare import compare_versions
        compare_versions(args.compare[0], args.compare[1], charts=not args.no_charts)
        return

    # ---- Run benchmarks ----
    from benchmarks.suites import ALL_SUITES
    from benchmarks.version import get_system_info, get_qforge_version, make_run_dir

    from qforge import _HAS_CPP, _HAS_METAL, _HAS_CUDA

    # Detect backends
    if args.backends:
        backends = args.backends
    else:
        backends = ['cpu']
        if _HAS_METAL:
            backends.append('metal')
        if _HAS_CUDA:
            backends.append('cuda')

    # Select suites
    if args.suites:
        selected = {k: v for k, v in ALL_SUITES.items() if k in args.suites}
    else:
        selected = ALL_SUITES

    # Build config
    qubit_range = list(range(2, args.max_qubits + 1, 2))
    config = {
        "depth": args.depth,
        "repeats": args.repeats,
        "steps": args.steps,
        "qubit_range": qubit_range,
        "grad_qubits": [2, 4, 6, 8, 10],
        "qml_qubits": [2, 4, 8],
        "vqe_qubits": [2, 4, 6],
        "qaoa_nodes": [3, 4, 5, 6],
        "tol": 1e-6 if 'metal' in backends else 1e-12,
    }

    version = get_qforge_version()
    sys_info = get_system_info()

    # Header
    print("=" * 70)
    print(f"  QFORGE BENCHMARK v{version}")
    print("=" * 70)
    print(f"  Backends:  {', '.join(backends)}")
    print(f"  Suites:    {', '.join(selected.keys())}")
    print(f"  Qubits:    2-{args.max_qubits}  |  Depth: {args.depth}  |  Steps: {args.steps}")
    print(f"  Platform:  {sys_info['platform']}")
    print("=" * 70)

    # Run suites
    all_results = {}
    total_t0 = time.perf_counter()

    for suite_name, suite_cls in selected.items():
        print(f"\n{'─'*70}")
        print(f"  Running: {suite_name} — {suite_cls.description}")
        print(f"{'─'*70}")

        suite = suite_cls(config)
        t0 = time.perf_counter()
        try:
            result = suite.run(backends=backends)
            elapsed = time.perf_counter() - t0
            all_results[suite_name] = result
            print(f"  ✓ {suite_name} completed in {elapsed:.2f}s")
        except Exception as e:
            print(f"  ✗ {suite_name} failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.perf_counter() - total_t0

    # Save results
    run_dir = make_run_dir(tag=args.tag)

    # Meta
    meta = {
        "version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suites": list(all_results.keys()),
        "backends": backends,
        "config": config,
        "system_info": sys_info,
        "total_time_seconds": total_time,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Full results
    with open(run_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    # Generate charts
    if not args.no_charts:
        try:
            from benchmarks.charts import generate_charts
            generate_charts(all_results, meta, run_dir)
            print(f"\n  Charts saved to {run_dir}/")
        except Exception as e:
            print(f"\n  Chart generation failed: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  COMPLETE — v{version}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results:    {run_dir}")
    print(f"{'='*70}")

    # Quick summary of key metrics
    if "gate_perf" in all_results:
        for backend, data in all_results["gate_perf"].items():
            circuit = data.get("circuit", {})
            if circuit:
                last_q = sorted(circuit.keys(), key=int)[-1]
                t = circuit[last_q]["time"]
                print(f"  {backend:6s} circuit {last_q}q: {t:.4f}s ({circuit[last_q]['throughput']:.0f} gates/s)")

    if "correctness" in all_results:
        for backend, data in all_results["correctness"].items():
            print(f"  {backend:6s} correctness: {data['n_pass']}/{len(data['tests'])} pass")


if __name__ == "__main__":
    main()
