# -*- coding: utf-8 -*-
"""Cross-version benchmark comparison."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.version import RESULTS_DIR, get_latest_run


def load_run(run_dir: Path) -> tuple[dict, dict]:
    """Load meta and results from a run directory."""
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    return meta, results


def compare_versions(v1: str, v2: str, charts: bool = True):
    """Compare benchmark results between two versions."""
    # Ensure v-prefix
    if not v1.startswith("v"):
        v1 = f"v{v1}"
    if not v2.startswith("v"):
        v2 = f"v{v2}"

    run1 = get_latest_run(v1)
    run2 = get_latest_run(v2)

    if run1 is None:
        print(f"No benchmark results found for {v1}")
        return
    if run2 is None:
        print(f"No benchmark results found for {v2}")
        return

    meta1, res1 = load_run(run1)
    meta2, res2 = load_run(run2)

    print("=" * 70)
    print(f"  BENCHMARK COMPARISON: {v1} vs {v2}")
    print("=" * 70)
    print(f"  {v1}: {meta1.get('timestamp', '?')}")
    print(f"  {v2}: {meta2.get('timestamp', '?')}")
    print()

    # Find common suites
    common = set(res1.keys()) & set(res2.keys())
    if not common:
        print("  No common suites to compare.")
        return

    # Compare gate_perf
    if "gate_perf" in common:
        print("─" * 70)
        print("  Gate-level Circuit Performance")
        print("─" * 70)
        for backend in set(res1["gate_perf"].keys()) & set(res2["gate_perf"].keys()):
            c1 = res1["gate_perf"][backend].get("circuit", {})
            c2 = res2["gate_perf"][backend].get("circuit", {})
            common_q = sorted(set(c1.keys()) & set(c2.keys()), key=int)
            if not common_q:
                continue
            print(f"\n  [{backend}]")
            print(f"  {'Qubits':>8} {v1+' (s)':>14} {v2+' (s)':>14} {'Speedup':>10}")
            print(f"  {'-'*50}")
            for q in common_q:
                t1 = c1[q]["time"]
                t2 = c2[q]["time"]
                speedup = t1 / t2 if t2 > 0 else float('inf')
                arrow = "▲" if speedup > 1.05 else ("▼" if speedup < 0.95 else "─")
                print(f"  {q:>8} {t1:>14.6f} {t2:>14.6f} {speedup:>8.2f}x {arrow}")

    # Compare correctness
    if "correctness" in common:
        print(f"\n{'─'*70}")
        print("  Correctness")
        print(f"{'─'*70}")
        for backend in set(res1["correctness"].keys()) & set(res2["correctness"].keys()):
            p1 = res1["correctness"][backend]["n_pass"]
            t1 = len(res1["correctness"][backend]["tests"])
            p2 = res2["correctness"][backend]["n_pass"]
            t2 = len(res2["correctness"][backend]["tests"])
            print(f"  {backend:>8}:  {v1} {p1}/{t1}    {v2} {p2}/{t2}")

    # Compare VQE
    if "vqe" in common:
        print(f"\n{'─'*70}")
        print("  VQE Training Time")
        print(f"{'─'*70}")
        for backend in set(res1["vqe"].get("vqe", {}).keys()) & set(res2["vqe"].get("vqe", {}).keys()):
            vqe1 = res1["vqe"]["vqe"][backend]
            vqe2 = res2["vqe"]["vqe"][backend]
            common_q = sorted(set(vqe1.keys()) & set(vqe2.keys()), key=int)
            if not common_q:
                continue
            print(f"\n  [{backend}]")
            print(f"  {'Qubits':>8} {v1+' (s)':>14} {v2+' (s)':>14} {'Speedup':>10}")
            print(f"  {'-'*50}")
            for q in common_q:
                t1 = vqe1[q]["time"]
                t2 = vqe2[q]["time"]
                speedup = t1 / t2 if t2 > 0 else float('inf')
                arrow = "▲" if speedup > 1.05 else ("▼" if speedup < 0.95 else "─")
                print(f"  {q:>8} {t1:>14.4f} {t2:>14.4f} {speedup:>8.2f}x {arrow}")

    # Compare QML accuracy
    if "qml" in common:
        print(f"\n{'─'*70}")
        print("  QML Classifier Accuracy")
        print(f"{'─'*70}")
        acc1 = res1["qml"].get("accuracy", {})
        acc2 = res2["qml"].get("accuracy", {})
        algos = set()
        for a in list(acc1.values()) + list(acc2.values()):
            algos.update(a.keys())
        for q in sorted(set(acc1.keys()) & set(acc2.keys()), key=int):
            print(f"\n  {q} qubits:")
            for algo in sorted(algos):
                a1 = acc1[q].get(algo, "n/a")
                a2 = acc2[q].get(algo, "n/a")
                if isinstance(a1, (int, float)) and isinstance(a2, (int, float)):
                    diff = a2 - a1
                    arrow = "▲" if diff > 0.01 else ("▼" if diff < -0.01 else "─")
                    print(f"    {algo:15s}  {a1:.2%} → {a2:.2%}  ({diff:+.1%}) {arrow}")
                else:
                    print(f"    {algo:15s}  {a1} → {a2}")

    # Generate comparison chart
    if charts:
        try:
            _generate_comparison_chart(v1, v2, meta1, meta2, res1, res2, run2.parent)
        except Exception as e:
            print(f"\n  Comparison chart failed: {e}")

    print(f"\n{'='*70}")


def _generate_comparison_chart(v1, v2, meta1, meta2, res1, res2, output_dir):
    """Generate a side-by-side comparison chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.suptitle(f'Benchmark Comparison: {v1} vs {v2}', fontsize=16, fontweight='bold')

    C1, C2 = '#1E88E5', '#E53935'

    # Plot 1: Gate circuit time
    ax = axes[0]
    if "gate_perf" in res1 and "gate_perf" in res2:
        for backend in set(res1["gate_perf"].keys()) & set(res2["gate_perf"].keys()):
            c1 = res1["gate_perf"][backend].get("circuit", {})
            c2 = res2["gate_perf"][backend].get("circuit", {})
            common_q = sorted(set(c1.keys()) & set(c2.keys()), key=int)
            qs = [int(q) for q in common_q]
            t1 = [c1[q]["time"] for q in common_q]
            t2 = [c2[q]["time"] for q in common_q]
            ax.semilogy(qs, t1, 'o-', color=C1, linewidth=2, label=f'{v1} ({backend})')
            ax.semilogy(qs, t2, 's--', color=C2, linewidth=2, label=f'{v2} ({backend})')
    ax.set_xlabel('Qubits'); ax.set_ylabel('Time (s)')
    ax.set_title('Gate Circuit Time'); ax.legend(); ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Speedup
    ax = axes[1]
    if "gate_perf" in res1 and "gate_perf" in res2:
        for backend in set(res1["gate_perf"].keys()) & set(res2["gate_perf"].keys()):
            c1 = res1["gate_perf"][backend].get("circuit", {})
            c2 = res2["gate_perf"][backend].get("circuit", {})
            common_q = sorted(set(c1.keys()) & set(c2.keys()), key=int)
            qs = [int(q) for q in common_q]
            speedups = [c1[q]["time"] / c2[q]["time"] if c2[q]["time"] > 0 else 1
                        for q in common_q]
            ax.plot(qs, speedups, 'D-', color=C2, linewidth=2, label=f'{backend}')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Qubits'); ax.set_ylabel(f'Speedup ({v2} / {v1})')
    ax.set_title(f'{v2} Speedup vs {v1}'); ax.legend(); ax.grid(True, alpha=0.3)

    path = output_dir / f"compare_{v1}_vs_{v2}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Comparison chart: {path}")
