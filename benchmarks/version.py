# -*- coding: utf-8 -*-
"""Version detection and result directory management."""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"


def get_system_info() -> dict[str, str]:
    """Collect system and framework metadata."""
    info: dict[str, str] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    try:
        import qforge
        info["qforge"] = getattr(qforge, "__version__", "dev")
        info["cpp"] = str(getattr(qforge, "_HAS_CPP", False))
        info["metal"] = str(getattr(qforge, "_HAS_METAL", False))
        info["cuda"] = str(getattr(qforge, "_HAS_CUDA", False))
    except ImportError:
        info["qforge"] = "not installed"
    try:
        import pennylane
        info["pennylane"] = pennylane.__version__
    except ImportError:
        info["pennylane"] = "n/a"
    try:
        import qiskit
        info["qiskit"] = qiskit.__version__
    except ImportError:
        info["qiskit"] = "n/a"
    return info


def get_qforge_version() -> str:
    try:
        import qforge
        return getattr(qforge, "__version__", "dev")
    except ImportError:
        return "dev"


def make_run_dir(tag: str | None = None) -> Path:
    """Create a timestamped result directory under results/v{version}/.

    Structure:
        benchmarks/results/v3.0.0/2026-03-15_143022/
        benchmarks/results/v3.0.0/2026-03-15_143022_mytag/

    Returns the created directory path.
    """
    version = get_qforge_version()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dirname = f"{ts}_{tag}" if tag else ts
    run_dir = RESULTS_DIR / f"v{version}" / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_versions() -> list[str]:
    """List all versions that have benchmark results."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()],
        key=lambda v: [int(x) for x in v.lstrip("v").split(".") if x.isdigit()],
    )


def list_runs(version: str | None = None) -> list[dict[str, Any]]:
    """List all benchmark runs, optionally filtered by version.

    Returns list of {version, timestamp, tag, path, suites}.
    """
    versions = [version] if version else list_versions()
    runs = []
    for v in versions:
        v_dir = RESULTS_DIR / v
        if not v_dir.is_dir():
            continue
        for run_dir in sorted(v_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            runs.append({
                "version": v,
                "name": run_dir.name,
                "path": str(run_dir),
                "suites": meta.get("suites", []),
                "timestamp": meta.get("timestamp", run_dir.name[:19]),
            })
    return runs


def get_latest_run(version: str | None = None) -> Path | None:
    """Get the path to the most recent benchmark run."""
    if version is None:
        versions = list_versions()
        if not versions:
            return None
        version = versions[-1]
    v_dir = RESULTS_DIR / version
    if not v_dir.exists():
        return None
    runs = sorted([d for d in v_dir.iterdir() if d.is_dir()])
    return runs[-1] if runs else None
