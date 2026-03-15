# -*- coding: utf-8 -*-
"""Chart generation for benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#1E88E5', '#E53935', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1']


def generate_charts(results: dict, meta: dict, output_dir: Path):
    """Generate all applicable charts from benchmark results."""
    charts = []

    if "gate_perf" in results:
        charts.append(_chart_gate_perf(results["gate_perf"], output_dir))

    if "correctness" in results:
        charts.append(_chart_correctness(results["correctness"], output_dir))

    if "gradient" in results:
        charts.append(_chart_gradient(results["gradient"], output_dir))

    if "vqe" in results:
        charts.append(_chart_vqe(results["vqe"], output_dir))

    if "qml" in results:
        charts.append(_chart_qml(results["qml"], output_dir))

    if "framework" in results:
        charts.append(_chart_framework(results["framework"], output_dir))

    if "backend" in results:
        charts.append(_chart_backend(results["backend"], output_dir))

    # Summary chart combining key metrics
    if len(charts) >= 2:
        _chart_summary(results, meta, output_dir)

    for c in charts:
        if c:
            print(f"    Chart: {c}")


def _chart_gate_perf(data: dict, output_dir: Path) -> str | None:
    """Gate-level circuit performance chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('Gate Circuit Performance', fontsize=14, fontweight='bold')

    # Circuit timing
    ax = axes[0]
    for i, (backend, bdata) in enumerate(data.items()):
        circuit = bdata.get("circuit", {})
        if not circuit:
            continue
        qs = sorted(circuit.keys(), key=int)
        x = [int(q) for q in qs]
        y = [circuit[q]["time"] for q in qs]
        ax.semilogy(x, y, 'o-', color=COLORS[i % len(COLORS)], linewidth=2,
                    label=backend, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('Circuit Build + Execute')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Throughput
    ax = axes[1]
    for i, (backend, bdata) in enumerate(data.items()):
        circuit = bdata.get("circuit", {})
        if not circuit:
            continue
        qs = sorted(circuit.keys(), key=int)
        x = [int(q) for q in qs]
        y = [circuit[q].get("throughput", 0) for q in qs]
        ax.plot(x, y, 's-', color=COLORS[i % len(COLORS)], linewidth=2,
                label=backend, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Gates/s')
    ax.set_title('Gate Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "gate_perf.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_correctness(data: dict, output_dir: Path) -> str | None:
    """Correctness pass/fail bar chart."""
    backends = list(data.keys())
    if not backends:
        return None

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    fig.suptitle('Correctness Tests', fontsize=14, fontweight='bold')

    x = np.arange(len(backends))
    passes = [data[b]["n_pass"] for b in backends]
    totals = [len(data[b]["tests"]) for b in backends]
    fails = [t - p for p, t in zip(passes, totals)]

    ax.bar(x, passes, color='#43A047', label='Pass')
    ax.bar(x, fails, bottom=passes, color='#E53935', label='Fail')
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.set_ylabel('Tests')
    ax.legend()

    for i, (p, t) in enumerate(zip(passes, totals)):
        ax.text(i, t + 0.3, f'{p}/{t}', ha='center', fontsize=10, fontweight='bold')

    path = output_dir / "correctness.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_gradient(data: dict, output_dir: Path) -> str | None:
    """Gradient computation scaling chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('Parameter-Shift Gradient Scaling', fontsize=14, fontweight='bold')

    # Time vs qubits
    ax = axes[0]
    for i, (backend, bdata) in enumerate(data.items()):
        qs = sorted(bdata.keys(), key=int)
        x = [int(q) for q in qs]
        y = [bdata[q]["time"] for q in qs]
        ax.semilogy(x, y, 'o-', color=COLORS[i % len(COLORS)], linewidth=2,
                    label=backend, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('Gradient Computation Time')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Time per eval
    ax = axes[1]
    for i, (backend, bdata) in enumerate(data.items()):
        qs = sorted(bdata.keys(), key=int)
        x = [int(q) for q in qs]
        y = [bdata[q]["time"] / bdata[q]["evals"] for q in qs]
        ax.semilogy(x, y, 's-', color=COLORS[i % len(COLORS)], linewidth=2,
                    label=backend, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time per evaluation (s)')
    ax.set_title('Cost per Circuit Evaluation')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    path = output_dir / "gradient.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_vqe(data: dict, output_dir: Path) -> str | None:
    """VQE/QAOA convergence and timing chart."""
    has_vqe = bool(data.get("vqe"))
    has_qaoa = bool(data.get("qaoa"))
    ncols = int(has_vqe) + int(has_qaoa)
    if ncols == 0:
        return None

    fig, axes = plt.subplots(1, max(ncols, 1), figsize=(7 * ncols, 5), facecolor='white')
    if ncols == 1:
        axes = [axes]
    fig.suptitle('VQE / QAOA', fontsize=14, fontweight='bold')

    idx = 0
    if has_vqe:
        ax = axes[idx]; idx += 1
        for i, (backend, bdata) in enumerate(data["vqe"].items()):
            qs = sorted(bdata.keys(), key=int)
            x = [int(q) for q in qs]
            y = [bdata[q]["time"] for q in qs]
            ax.bar([xi + i * 0.25 for xi in range(len(x))], y, width=0.25,
                   color=COLORS[i % len(COLORS)], label=backend)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(xi) for xi in x])
        ax.set_xlabel('Qubits')
        ax.set_ylabel('Time (s)')
        ax.set_title('VQE Training Time')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    if has_qaoa:
        ax = axes[idx]
        for i, (backend, bdata) in enumerate(data["qaoa"].items()):
            ns = sorted(bdata.keys(), key=int)
            x = [int(n) for n in ns]
            y = [bdata[n]["time"] for n in ns]
            ax.bar([xi + i * 0.25 for xi in range(len(x))], y, width=0.25,
                   color=COLORS[i % len(COLORS)], label=backend)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(xi) for xi in x])
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Time (s)')
        ax.set_title('QAOA Training Time')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    path = output_dir / "vqe_qaoa.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_qml(data: dict, output_dir: Path) -> str | None:
    """QML classifier accuracy and training time chart."""
    acc = data.get("accuracy", {})
    tt = data.get("training_time", {})
    if not acc:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('QML Classifiers', fontsize=14, fontweight='bold')

    # Accuracy grouped by qubit count
    ax = axes[0]
    qubit_keys = sorted(acc.keys(), key=int)
    algos = sorted(set().union(*(acc[q].keys() for q in qubit_keys)))
    x = np.arange(len(qubit_keys))
    width = 0.8 / max(len(algos), 1)
    for j, algo in enumerate(algos):
        vals = [acc[q].get(algo, 0) for q in qubit_keys]
        ax.bar(x + j * width, vals, width, label=algo, color=COLORS[j % len(COLORS)])
    ax.set_xticks(x + width * len(algos) / 2)
    ax.set_xticklabels([f'{q}q' for q in qubit_keys])
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Training time
    ax = axes[1]
    if tt:
        for j, algo in enumerate(algos):
            vals = [tt.get(q, {}).get(algo, 0) for q in qubit_keys]
            ax.bar(x + j * width, vals, width, label=algo, color=COLORS[j % len(COLORS)])
        ax.set_xticks(x + width * len(algos) / 2)
        ax.set_xticklabels([f'{q}q' for q in qubit_keys])
        ax.set_ylabel('Time (s)')
        ax.set_title('Training Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    path = output_dir / "qml.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_framework(data: dict, output_dir: Path) -> str | None:
    """Framework comparison chart (Qforge vs Qiskit vs PennyLane)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('Framework Comparison', fontsize=14, fontweight='bold')

    frameworks = list(data.keys())
    if not frameworks:
        return None

    # Circuit time
    ax = axes[0]
    for i, fw in enumerate(frameworks):
        circuit = data[fw].get("circuit", {})
        if not circuit:
            continue
        qs = sorted(circuit.keys(), key=int)
        x = [int(q) for q in qs]
        y = [circuit[q]["time"] for q in qs]
        ax.semilogy(x, y, 'o-', color=COLORS[i % len(COLORS)], linewidth=2,
                    label=fw, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('Circuit Build + Execute')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Speedup vs first framework
    ax = axes[1]
    ref = frameworks[0] if frameworks else None
    ref_circuit = data.get(ref, {}).get("circuit", {}) if ref else {}
    for i, fw in enumerate(frameworks[1:], 1):
        circuit = data[fw].get("circuit", {})
        common_q = sorted(set(ref_circuit.keys()) & set(circuit.keys()), key=int)
        if not common_q:
            continue
        x = [int(q) for q in common_q]
        y = [circuit[q]["time"] / ref_circuit[q]["time"] if ref_circuit[q]["time"] > 0 else 1
             for q in common_q]
        ax.plot(x, y, 's-', color=COLORS[i % len(COLORS)], linewidth=2,
                label=f'{fw} / {ref}', markersize=5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Relative Time')
    ax.set_title(f'Slowdown vs {ref}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "framework_compare.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_backend(data: dict, output_dir: Path) -> str | None:
    """Backend comparison chart (CPU vs Metal vs CUDA)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('Backend Comparison', fontsize=14, fontweight='bold')

    backends = list(data.keys())
    if not backends:
        return None

    ax = axes[0]
    for i, backend in enumerate(backends):
        circuit = data[backend].get("circuit", {})
        if not circuit:
            continue
        qs = sorted(circuit.keys(), key=int)
        x = [int(q) for q in qs]
        y = [circuit[q]["time"] for q in qs]
        ax.semilogy(x, y, 'o-', color=COLORS[i % len(COLORS)], linewidth=2,
                    label=backend, markersize=5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('Circuit Time by Backend')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Speedup vs CPU
    ax = axes[1]
    cpu_data = data.get("cpu", {}).get("circuit", {})
    for i, backend in enumerate(backends):
        if backend == "cpu":
            continue
        circuit = data[backend].get("circuit", {})
        common_q = sorted(set(cpu_data.keys()) & set(circuit.keys()), key=int)
        if not common_q:
            continue
        x = [int(q) for q in common_q]
        y = [cpu_data[q]["time"] / circuit[q]["time"] if circuit[q]["time"] > 0 else 1
             for q in common_q]
        ax.plot(x, y, 's-', color=COLORS[i % len(COLORS)], linewidth=2,
                label=f'{backend} speedup', markersize=5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Speedup vs CPU')
    ax.set_title('Accelerator Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "backend_compare.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(path.name)


def _chart_summary(results: dict, meta: dict, output_dir: Path):
    """Generate a single-page summary of all benchmarks."""
    version = meta.get("version", "?")
    backends = meta.get("backends", [])
    suites = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
    ax.axis('off')

    lines = [
        f"Qforge v{version} Benchmark Summary",
        f"Date: {meta.get('timestamp', '?')}",
        f"Backends: {', '.join(backends)}",
        f"Suites: {', '.join(suites)}",
        f"Total time: {meta.get('total_time_seconds', 0):.1f}s",
    ]

    # Add key metrics
    if "gate_perf" in results:
        for backend, bdata in results["gate_perf"].items():
            circuit = bdata.get("circuit", {})
            if circuit:
                last_q = sorted(circuit.keys(), key=int)[-1]
                t = circuit[last_q]["time"]
                lines.append(f"{backend} {last_q}q circuit: {t:.4f}s")

    if "correctness" in results:
        for backend, bdata in results["correctness"].items():
            lines.append(f"{backend} correctness: {bdata['n_pass']}/{len(bdata['tests'])} pass")

    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    path = output_dir / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
