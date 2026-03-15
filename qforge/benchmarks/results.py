# -*- coding: utf-8 -*-
"""Benchmark results collection, serialization, and loading."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class BenchmarkResults:
    """Aggregated results from one or more benchmark suites."""

    timestamp: str
    config: Dict[str, Any]
    suites: Dict[str, Dict[str, Any]]
    system_info: Dict[str, str]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, config, suite_results: Dict[str, Dict[str, Any]]) -> "BenchmarkResults":
        """Build results from a BenchConfig and suite output dicts."""
        from qforge.benchmarks.core import BenchConfig

        sys_info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
        }
        try:
            import qforge
            sys_info["qforge_version"] = getattr(qforge, "__version__", "dev")
            sys_info["qforge_cpp"] = str(getattr(qforge, "_HAS_CPP", False))
            sys_info["qforge_cuda"] = str(getattr(qforge, "_HAS_CUDA", False))
        except Exception:
            pass

        try:
            qml = __import__("pennylane")
            sys_info["pennylane_version"] = qml.__version__
        except Exception:
            pass

        try:
            qiskit = __import__("qiskit")
            sys_info["qiskit_version"] = qiskit.__version__
        except Exception:
            pass

        cfg_dict = {
            "n_warmup": config.n_warmup,
            "n_runs": config.n_runs,
            "seed": config.seed,
            "max_qubits": config.max_qubits,
            "steps": config.steps,
            "lr": config.lr,
        }

        return cls(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            config=cfg_dict,
            suites=suite_results,
            system_info=sys_info,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: Path) -> "BenchmarkResults":
        with open(path) as f:
            d = json.load(f)
        return cls(
            timestamp=d["timestamp"],
            config=d["config"],
            suites=d["suites"],
            system_info=d["system_info"],
        )

    def merge(self, other: "BenchmarkResults") -> None:
        """Merge results from another run into this one."""
        for suite_name, suite_data in other.suites.items():
            if suite_name not in self.suites:
                self.suites[suite_name] = suite_data
            else:
                self.suites[suite_name].update(suite_data)

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "system_info": self.system_info,
            "suites": self.suites,
        }
