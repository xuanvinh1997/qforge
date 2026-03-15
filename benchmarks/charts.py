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
    """Framework comparison chart (fair benchmark).

    Data format: {simulate: {qubit: {fw: {median, q25, q75, ...}}},
                  gradient: {qubit: {fw: {method: {median, ...}}}},
                  methodology: {...}}
    """
    sim_data = data.get("simulate", {})
    grad_data = data.get("gradient", {})
    if not sim_data and not grad_data:
        return None

    FW_COLORS = {'qforge': '#1E88E5', 'qiskit': '#E53935', 'pennylane': '#43A047'}
    FW_MARKERS = {'qforge': 'o', 'qiskit': 's', 'pennylane': 'D'}
    META_KEYS = {'n_qubits', 'depth', 'n_params'}
    charts_made = []

    # ---- Chart 1: Simulation time (median + IQR) ----
    if sim_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        fig.suptitle('Framework Comparison — Simulation (median ± IQR, with warm-up)',
                     fontsize=13, fontweight='bold')

        qubit_keys = sorted(sim_data.keys(), key=int)
        frameworks = sorted(set().union(*(
            {k for k in sim_data[q] if k not in META_KEYS}
            for q in qubit_keys
        )))

        # Plot 1a: Absolute time
        ax = axes[0]
        for fw in frameworks:
            qs, meds, lo, hi = [], [], [], []
            for q in qubit_keys:
                entry = sim_data[q].get(fw)
                if isinstance(entry, dict) and 'median' in entry:
                    qs.append(int(q))
                    meds.append(entry['median'])
                    lo.append(entry['q25'])
                    hi.append(entry['q75'])
            if qs:
                meds, lo, hi = np.array(meds), np.array(lo), np.array(hi)
                ax.semilogy(qs, meds, f'{FW_MARKERS.get(fw, "o")}-',
                            color=FW_COLORS.get(fw, '#888'), linewidth=2,
                            label=fw, markersize=6)
                ax.fill_between(qs, lo, hi, alpha=0.15,
                                color=FW_COLORS.get(fw, '#888'))
        ax.set_xlabel('Qubits')
        ax.set_ylabel('Time (s)')
        ax.set_title('Simulation Time')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Plot 1b: Slowdown vs qforge
        ax = axes[1]
        ref_fw = 'qforge'
        for fw in frameworks:
            if fw == ref_fw:
                continue
            qs, ratios = [], []
            for q in qubit_keys:
                r = sim_data[q].get(ref_fw)
                f = sim_data[q].get(fw)
                if (isinstance(r, dict) and isinstance(f, dict)
                        and 'median' in r and 'median' in f and r['median'] > 0):
                    qs.append(int(q))
                    ratios.append(f['median'] / r['median'])
            if qs:
                ax.plot(qs, ratios, f'{FW_MARKERS.get(fw, "s")}-',
                        color=FW_COLORS.get(fw, '#888'), linewidth=2,
                        label=f'{fw} / {ref_fw}', markersize=6)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Qubits')
        ax.set_ylabel(f'Slowdown vs {ref_fw}')
        ax.set_title('Relative Simulation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = output_dir / "framework_simulate.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        charts_made.append(str(path.name))

    # ---- Chart 2: Gradient time by method ----
    if grad_data:
        # Discover all (framework, method) combinations
        qubit_keys = sorted(grad_data.keys(), key=int)
        fw_methods = set()
        for q in qubit_keys:
            for fw, fdata in grad_data[q].items():
                if fw in META_KEYS:
                    continue
                if isinstance(fdata, dict):
                    for method in fdata:
                        if isinstance(fdata[method], dict) and 'median' in fdata[method]:
                            fw_methods.add((fw, method))

        if fw_methods:
            METHOD_STYLES = {
                'parameter_shift': '-',
                'adjoint': '--',
                'backprop': ':',
            }

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
            fig.suptitle('Framework Comparison — Gradient Methods (median ± IQR)',
                         fontsize=13, fontweight='bold')

            # Plot 2a: Absolute gradient time
            ax = axes[0]
            for fw, method in sorted(fw_methods):
                qs, meds, lo, hi = [], [], [], []
                for q in qubit_keys:
                    entry = grad_data[q].get(fw, {})
                    if isinstance(entry, dict):
                        m = entry.get(method)
                        if isinstance(m, dict) and 'median' in m:
                            qs.append(int(q))
                            meds.append(m['median'])
                            lo.append(m['q25'])
                            hi.append(m['q75'])
                if qs:
                    meds, lo, hi = np.array(meds), np.array(lo), np.array(hi)
                    ls = METHOD_STYLES.get(method, '-')
                    ax.semilogy(qs, meds,
                                f'{FW_MARKERS.get(fw, "o")}{ls}',
                                color=FW_COLORS.get(fw, '#888'), linewidth=2,
                                label=f'{fw} ({method})', markersize=5)
                    ax.fill_between(qs, lo, hi, alpha=0.1,
                                    color=FW_COLORS.get(fw, '#888'))
            ax.set_xlabel('Qubits')
            ax.set_ylabel('Time (s)')
            ax.set_title('Gradient Computation Time')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')

            # Plot 2b: Best gradient method per framework
            ax = axes[1]
            for fw in sorted(set(f for f, _ in fw_methods)):
                qs, best_times = [], []
                for q in qubit_keys:
                    entry = grad_data[q].get(fw, {})
                    if not isinstance(entry, dict):
                        continue
                    best = float('inf')
                    for method, m in entry.items():
                        if isinstance(m, dict) and 'median' in m:
                            best = min(best, m['median'])
                    if best < float('inf'):
                        qs.append(int(q))
                        best_times.append(best)
                if qs:
                    ax.semilogy(qs, best_times,
                                f'{FW_MARKERS.get(fw, "o")}-',
                                color=FW_COLORS.get(fw, '#888'), linewidth=2,
                                label=f'{fw} (best)', markersize=6)
            ax.set_xlabel('Qubits')
            ax.set_ylabel('Time (s)')
            ax.set_title('Best Gradient Method per Framework')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')

            path = output_dir / "framework_gradient.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            charts_made.append(str(path.name))

    return ", ".join(charts_made) if charts_made else None


def _chart_backend(data: dict, output_dir: Path) -> str | None:
    """Backend comparison chart (CPU vs Metal vs CUDA).

    Data format: {backends: [...], circuit: {qubit: {backend: time}},
                  single_gate: {qubit: {backend: us}}, metal_crossover_qubits: int|None}
    """
    circuit_data = data.get("circuit", {})
    single_gate = data.get("single_gate", {})
    backend_list = data.get("backends", [])
    if not circuit_data:
        return None

    BK_COLORS = {'cpu': '#1E88E5', 'metal': '#FB8C00', 'cuda': '#43A047'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.suptitle('Backend Comparison', fontsize=14, fontweight='bold')

    # Plot 1: Circuit time per backend
    ax = axes[0]
    qubit_keys = sorted(circuit_data.keys(), key=int)
    for i, backend in enumerate(backend_list):
        qs, times = [], []
        for q in qubit_keys:
            t = circuit_data[q].get(backend)
            if t is not None:
                qs.append(int(q))
                times.append(t)
        if qs:
            ax.semilogy(qs, times, 'o-', color=BK_COLORS.get(backend, COLORS[i]),
                        linewidth=2, label=backend, markersize=5)
    crossover = data.get("metal_crossover_qubits")
    if crossover:
        ax.axvline(x=crossover, color='#FB8C00', linestyle='--', alpha=0.6,
                   label=f'Metal crossover ({crossover}q)')
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Time (s)')
    ax.set_title('Circuit Time by Backend')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 2: Speedup vs CPU
    ax = axes[1]
    for i, backend in enumerate(backend_list):
        if backend == 'cpu':
            continue
        qs, speedups = [], []
        for q in qubit_keys:
            t_cpu = circuit_data[q].get('cpu')
            t_bk = circuit_data[q].get(backend)
            if t_cpu and t_bk and t_bk > 0:
                qs.append(int(q))
                speedups.append(t_cpu / t_bk)
        if qs:
            ax.plot(qs, speedups, 's-', color=BK_COLORS.get(backend, COLORS[i]),
                    linewidth=2, label=f'{backend} speedup', markersize=5)
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
