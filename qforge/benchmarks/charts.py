# -*- coding: utf-8 -*-
"""Chart generation for benchmark results using matplotlib.

All functions accept results dicts and produce PNG files.
matplotlib is imported lazily so benchmarks can run without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "qforge":      "#2196F3",
    "qforge_cpu":  "#2196F3",
    "qforge_cuda": "#1565C0",
    "qforge_metal": "#0D47A1",
    "pennylane":   "#4CAF50",
    "qiskit":      "#FF9800",
    "mps_chi64":   "#9C27B0",
    "mps_chi128":  "#7B1FA2",
    "statevector":  "#F44336",
    "theoretical":  "#9E9E9E",
    "cpp":         "#2196F3",
    "python":      "#FF5722",
}

FRAMEWORK_ORDER = ["qforge", "pennylane", "qiskit"]


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------

def grouped_bar_chart(
    data: Dict[str, Dict[str, Optional[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    frameworks: List[str] = None,
    log_scale: bool = False,
    figsize: tuple = None,
) -> Path:
    """Grouped bar chart comparing frameworks across test cases.

    data: {"H_4q": {"qforge": 1.2, "pennylane": 5.3, "qiskit": 4.1}, ...}
    """
    plt = _get_plt()

    if frameworks is None:
        frameworks = FRAMEWORK_ORDER
    labels = list(data.keys())
    n_groups = len(labels)
    n_bars = len(frameworks)

    if figsize is None:
        figsize = (max(8, n_groups * 1.2), 5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    width = 0.8 / max(n_bars, 1)

    for i, fw in enumerate(frameworks):
        vals = []
        for label in labels:
            v = data[label].get(fw)
            vals.append(v if v is not None else 0)
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=fw.capitalize(),
                      color=COLORS.get(fw, f"C{i}"), alpha=0.85)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Benchmark")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Scaling line chart
# ---------------------------------------------------------------------------

def scaling_line_chart(
    data: Dict[str, Dict[str, Optional[float]]],
    x_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    series_keys: List[str] = None,
    log_scale: bool = False,
    figsize: tuple = (10, 6),
) -> Path:
    """Line chart showing scaling behavior.

    data: {"4q": {"qubits": 4, "qforge": 0.5, "pennylane": 2.1}, ...}
    """
    plt = _get_plt()

    if series_keys is None:
        series_keys = FRAMEWORK_ORDER

    # Extract x values and series
    entries = sorted(data.values(), key=lambda d: d.get(x_key, 0))
    x_vals = [d[x_key] for d in entries]

    fig, ax = plt.subplots(figsize=figsize)

    for fw in series_keys:
        y_vals = [d.get(fw) for d in entries]
        # Filter None values
        valid = [(xv, yv) for xv, yv in zip(x_vals, y_vals) if yv is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", label=fw.capitalize(),
                    color=COLORS.get(fw, None), linewidth=2, markersize=6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Convergence chart
# ---------------------------------------------------------------------------

def convergence_chart(
    series: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> Path:
    """Line chart of convergence (e.g., energy vs sweep).

    series: {"chi=32": [e1, e2, ...], "chi=64": [...]}
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, values) in enumerate(series.items()):
        ax.plot(range(1, len(values) + 1), values, "o-",
                label=name, linewidth=2, markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Memory chart (bars + theoretical line)
# ---------------------------------------------------------------------------

def memory_chart(
    data: Dict[str, Dict[str, Optional[float]]],
    output_path: Path,
    figsize: tuple = (10, 6),
) -> Path:
    """Bar chart of memory usage with theoretical baseline."""
    plt = _get_plt()

    entries = sorted(data.values(), key=lambda d: d.get("qubits", 0))
    qubits = [d["qubits"] for d in entries]
    theoretical = [d.get("theoretical", 0) for d in entries]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(qubits))
    width = 0.2

    for i, fw in enumerate(["qforge", "pennylane", "qiskit"]):
        vals = [d.get(fw, 0) or 0 for d in entries]
        ax.bar(x + (i - 1) * width, vals, width, label=fw.capitalize(),
               color=COLORS.get(fw), alpha=0.85)

    ax.plot(x, theoretical, "k--o", label="Theoretical", linewidth=2,
            markersize=5, color=COLORS["theoretical"])

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Usage: Frameworks vs Theoretical", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(qubits)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    if max(theoretical) > 0:
        ax.set_yscale("log")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Accuracy heatmap
# ---------------------------------------------------------------------------

def accuracy_heatmap(
    data: Dict[str, Dict[str, Optional[float]]],
    output_path: Path,
    figsize: tuple = (8, 5),
) -> Path:
    """Heatmap of max amplitude differences between frameworks."""
    plt = _get_plt()

    tests = list(data.keys())
    pairs = ["qf_pl", "qf_qk", "pl_qk"]
    pair_labels = ["QF vs PL", "QF vs QK", "PL vs QK"]

    matrix = np.zeros((len(tests), len(pairs)))
    for i, test in enumerate(tests):
        for j, pair in enumerate(pairs):
            v = data[test].get(pair)
            matrix[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=figsize)
    # Use log scale for the color map
    with np.errstate(divide="ignore", invalid="ignore"):
        log_matrix = np.log10(matrix + 1e-20)

    im = ax.imshow(log_matrix, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels(tests)

    # Annotate cells
    for i in range(len(tests)):
        for j in range(len(pairs)):
            v = matrix[i, j]
            text = f"{v:.1e}" if not np.isnan(v) else "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    ax.set_title("Accuracy: Max |Amplitude Difference|", fontweight="bold")
    fig.colorbar(im, ax=ax, label="log10(error)")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Bond dimension chart (for MPS/DMRG)
# ---------------------------------------------------------------------------

def bond_dim_chart(
    data: Dict[str, Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (8, 5),
) -> Path:
    """Chart for bond dimension related results (MPS accuracy, DMRG convergence)."""
    plt = _get_plt()

    entries = sorted(data.values(), key=lambda d: d.get(x_key, 0))
    x_vals = [d[x_key] for d in entries]
    y_vals = [d.get(y_key, 0) for d in entries]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_vals, y_vals, "o-", color=COLORS["mps_chi64"], linewidth=2, markersize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Suite-level chart dispatcher
# ---------------------------------------------------------------------------

def generate_suite_charts(suite_name: str, results: Dict, output_dir: Path) -> List[Path]:
    """Generate all charts for a given suite. Returns list of chart paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "gates": _charts_gates,
        "circuits": _charts_circuits,
        "vqe": _charts_comparison,
        "qaoa": _charts_comparison,
        "gradient": _charts_comparison,
        "measurement": _charts_comparison,
        "scaling": _charts_scaling,
        "accuracy": _charts_accuracy,
        "memory": _charts_memory,
        "mps": _charts_mps,
        "dmrg": _charts_dmrg,
    }

    fn = dispatch.get(suite_name, _charts_generic)
    try:
        return fn(suite_name, results, output_dir)
    except Exception as e:
        print(f"  Warning: chart generation failed for {suite_name}: {e}")
        return []


def _charts_gates(name, results, output_dir):
    """Generate per-qubit grouped bar charts for gates."""
    paths = []
    # Group by qubit count
    by_qubit: Dict[int, Dict[str, Dict]] = {}
    for key, data in results.items():
        nq = data.get("qubits", 0)
        if nq not in by_qubit:
            by_qubit[nq] = {}
        gate = data.get("gate", key)
        by_qubit[nq][gate] = {
            fw: data.get(fw) for fw in FRAMEWORK_ORDER
        }

    for nq in sorted(by_qubit.keys()):
        p = grouped_bar_chart(
            by_qubit[nq],
            title=f"Gate Performance — {nq} Qubits",
            ylabel="Time (us/gate)",
            output_path=output_dir / f"gates_{nq}q.png",
            log_scale=True,
        )
        paths.append(p)
    return paths


def _charts_circuits(name, results, output_dir):
    """Generate per-pattern grouped bar charts for circuits."""
    paths = []
    by_pattern: Dict[str, Dict[str, Dict]] = {}
    for key, data in results.items():
        pat = data.get("pattern", key)
        nq = data.get("qubits", 0)
        if pat not in by_pattern:
            by_pattern[pat] = {}
        by_pattern[pat][f"{nq}q"] = {
            fw: data.get(fw) for fw in FRAMEWORK_ORDER
        }

    for pat in by_pattern:
        p = grouped_bar_chart(
            by_pattern[pat],
            title=f"Circuit Pattern: {pat}",
            ylabel="Time (ms)",
            output_path=output_dir / f"circuits_{pat.replace(' ', '_').replace('(', '').replace(')', '')}.png",
            log_scale=True,
        )
        paths.append(p)
    return paths


def _charts_comparison(name, results, output_dir):
    """Generic framework comparison bar chart."""
    # Extract time data
    chart_data = {}
    for key, data in results.items():
        fw = data.get("framework", key)
        prob = data.get("problem", key)
        t = data.get("time")
        if t is not None:
            if prob not in chart_data:
                chart_data[prob] = {}
            # Normalize framework name
            fw_clean = fw.replace("qforge_", "qforge ").split()[0] if "qforge" in fw else fw
            chart_data[prob][fw_clean] = t

    if chart_data:
        p = grouped_bar_chart(
            chart_data,
            title=f"{name.upper()} — Total Time Comparison",
            ylabel="Time (s)",
            output_path=output_dir / f"{name}_comparison.png",
        )
        return [p]
    return []


def _charts_scaling(name, results, output_dir):
    """Scaling line chart."""
    return [scaling_line_chart(
        results,
        x_key="qubits",
        title="Scalability: HEA Forward Pass",
        xlabel="Qubits",
        ylabel="Time (ms)",
        output_path=output_dir / "scaling.png",
        log_scale=True,
    )]


def _charts_accuracy(name, results, output_dir):
    """Accuracy heatmap."""
    return [accuracy_heatmap(results, output_dir / "accuracy_heatmap.png")]


def _charts_memory(name, results, output_dir):
    """Memory comparison chart."""
    return [memory_chart(results, output_dir / "memory.png")]


def _charts_mps(name, results, output_dir):
    """MPS-specific charts."""
    paths = []

    # Gate application: MPS vs statevector
    gate_data = {k: v for k, v in results.items() if k.startswith("gate_ghz_")}
    if gate_data:
        chart_data = {}
        for key, data in gate_data.items():
            nq = data.get("qubits", 0)
            chart_data[f"{nq}q"] = {
                "statevector": data.get("statevector"),
                "mps_chi64": data.get("mps_chi64"),
                "mps_chi128": data.get("mps_chi128"),
            }
        p = grouped_bar_chart(
            chart_data,
            title="MPS vs StateVector: GHZ Circuit",
            ylabel="Time (ms)",
            output_path=output_dir / "mps_vs_statevec.png",
            frameworks=["statevector", "mps_chi64", "mps_chi128"],
            log_scale=True,
        )
        paths.append(p)

    # Qubit scaling
    qubit_data = {k: v for k, v in results.items() if k.startswith("qubit_scale_")}
    if qubit_data:
        p = scaling_line_chart(
            qubit_data,
            x_key="qubits",
            title="MPS Qubit Scaling (chi=64)",
            xlabel="Qubits",
            ylabel="Time (ms)",
            output_path=output_dir / "mps_qubit_scaling.png",
            series_keys=["time_ms"],
        )
        paths.append(p)

    # Bond dim scaling
    bond_data = {k: v for k, v in results.items() if k.startswith("bonddim_")}
    if bond_data:
        p = bond_dim_chart(
            bond_data, x_key="max_chi", y_key="time_ms",
            title="MPS: Time vs Bond Dimension (30q)",
            xlabel="Max Bond Dimension (chi)",
            ylabel="Time (ms)",
            output_path=output_dir / "mps_bonddim_scaling.png",
        )
        paths.append(p)

    # Accuracy vs bond dim
    acc_data = {k: v for k, v in results.items() if k.startswith("accuracy_chi")}
    if acc_data:
        p = bond_dim_chart(
            acc_data, x_key="chi", y_key="fidelity",
            title="MPS Fidelity vs Bond Dimension",
            xlabel="Max Bond Dimension (chi)",
            ylabel="Fidelity",
            output_path=output_dir / "mps_fidelity.png",
        )
        paths.append(p)

    return paths


def _charts_dmrg(name, results, output_dir):
    """DMRG-specific charts."""
    paths = []

    # Heisenberg scaling
    heis_data = {k: v for k, v in results.items() if k.startswith("heisenberg_")}
    if heis_data:
        p = scaling_line_chart(
            heis_data,
            x_key="n_sites",
            title="DMRG: Heisenberg Chain Scaling",
            xlabel="Number of Sites",
            ylabel="Time (s)",
            output_path=output_dir / "dmrg_heisenberg_scaling.png",
            series_keys=["time_s"],
        )
        paths.append(p)

    # Bond dim convergence
    bond_data = {k: v for k, v in results.items() if k.startswith("bondconv_")}
    if bond_data:
        p = bond_dim_chart(
            bond_data, x_key="chi", y_key="energy",
            title="DMRG: Energy vs Bond Dimension (20-site Heisenberg)",
            xlabel="Max Bond Dimension (chi)",
            ylabel="Ground State Energy",
            output_path=output_dir / "dmrg_bonddim_convergence.png",
        )
        paths.append(p)

    # Sweep convergence
    sweep_data = results.get("sweep_convergence")
    if sweep_data and "energies" in sweep_data:
        p = convergence_chart(
            {"Energy": sweep_data["energies"]},
            title="DMRG: Sweep Convergence (20-site Heisenberg, chi=32)",
            xlabel="Sweep",
            ylabel="Energy",
            output_path=output_dir / "dmrg_sweep_convergence.png",
        )
        paths.append(p)

    # Ising phase diagram
    ising_data = {k: v for k, v in results.items() if k.startswith("ising_h")}
    if ising_data:
        p = bond_dim_chart(
            ising_data, x_key="h", y_key="energy_per_site",
            title="Transverse-Field Ising: E/site vs h/J",
            xlabel="Transverse Field h/J",
            ylabel="Energy per Site",
            output_path=output_dir / "dmrg_ising_phase.png",
        )
        paths.append(p)

    return paths


def _charts_generic(name, results, output_dir):
    """Fallback for unknown suite types."""
    return []
