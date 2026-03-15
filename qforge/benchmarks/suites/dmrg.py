# -*- coding: utf-8 -*-
"""Category 11: DMRG ground-state solver benchmarks.

Qforge-specific: measures DMRG performance characteristics including
site scaling, bond dimension convergence, and sweep convergence.
"""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
)


class DMRGBenchmarkSuite(BaseBenchmarkSuite):
    name = "dmrg"
    description = "DMRG Ground-State Solver Benchmarks"

    def run(self):
        section(f"CATEGORY 11: {self.description}", "")
        self._bench_heisenberg_scaling()
        self._bench_bond_dim_convergence()
        self._bench_sweep_convergence()
        self._bench_ising_phase()
        self._bench_backend_comparison()
        return self._results

    # ── 1. Heisenberg chain scaling ────────────────────────────────────

    def _bench_heisenberg_scaling(self):
        """DMRG time for Heisenberg chain vs system size."""
        from qforge.dmrg import DMRG

        print("\n  --- Heisenberg Chain: Time vs System Size (chi=32, 10 sweeps) ---")
        headers = ["Sites", "Time(s)", "Energy", "E/site"]
        rows = []

        for n_sites in [10, 20, 40, 60, 80, 100]:
            def run(n=n_sites):
                dmrg = DMRG.heisenberg(n_sites=n, max_bond_dim=32)
                energy, _ = dmrg.run(n_sweeps=10)
                return energy

            try:
                r = bench(run, n_warmup=0, n_runs=3)
                energy = float(r.result)
                rows.append([n_sites, f"{r.median:.3f}", f"{energy:.6f}", f"{energy / n_sites:.6f}"])
                self._store(f"heisenberg_{n_sites}", {
                    "n_sites": n_sites, "time_s": r.median,
                    "energy": energy, "energy_per_site": energy / n_sites,
                })
            except Exception as e:
                rows.append([n_sites, "ERR", "ERR", str(e)[:30]])

        table(headers, rows)

    # ── 2. Bond dimension convergence ──────────────────────────────────

    def _bench_bond_dim_convergence(self):
        """Energy convergence vs max_bond_dim for 20-site Heisenberg."""
        from qforge.dmrg import DMRG

        n_sites = 20
        print(f"\n  --- Bond Dim Convergence: {n_sites}-site Heisenberg, 20 sweeps ---")
        headers = ["Max chi", "Energy", "Time(s)", "Final max chi"]
        rows = []

        for chi in [4, 8, 16, 32, 64, 128]:
            def run(chi=chi):
                dmrg = DMRG.heisenberg(n_sites=n_sites, max_bond_dim=chi)
                energy, psi = dmrg.run(n_sweeps=20)
                max_chi_actual = max(psi.bond_dimensions())
                return (energy, max_chi_actual)

            try:
                r = bench(run, n_warmup=0, n_runs=3)
                energy, max_chi_actual = r.result
                rows.append([chi, f"{energy:.8f}", f"{r.median:.3f}", max_chi_actual])
                self._store(f"bondconv_{chi}", {
                    "chi": chi, "energy": float(energy),
                    "time_s": r.median, "actual_chi": int(max_chi_actual),
                })
            except Exception as e:
                rows.append([chi, "ERR", "ERR", str(e)[:30]])

        table(headers, rows)

    # ── 3. Sweep convergence ───────────────────────────────────────────

    def _bench_sweep_convergence(self):
        """Energy per sweep for 20-site Heisenberg (chi=32)."""
        from qforge.dmrg import DMRG

        n_sites = 20
        print(f"\n  --- Sweep Convergence: {n_sites}-site Heisenberg, chi=32 ---")
        headers = ["Sweep", "Energy", "Delta E"]
        rows = []

        dmrg = DMRG.heisenberg(n_sites=n_sites, max_bond_dim=32)
        energy, _ = dmrg.run(n_sweeps=30)
        history = dmrg.energy_history

        for i, e in enumerate(history):
            delta = abs(e - history[i - 1]) if i > 0 else float('inf')
            rows.append([i + 1, f"{e:.10f}", f"{delta:.2e}" if i > 0 else "-"])

        # Store for chart generation
        self._store("sweep_convergence", {
            "n_sites": n_sites, "chi": 32,
            "energies": [float(e) for e in history],
            "final_energy": float(energy),
        })

        table(headers, rows)

    # ── 4. Ising model phase diagram ───────────────────────────────────

    def _bench_ising_phase(self):
        """Ground state energy at various transverse field strengths."""
        from qforge.dmrg import DMRG

        n_sites = 20
        print(f"\n  --- Transverse-Field Ising: {n_sites} sites, chi=32 ---")
        headers = ["h/J", "Energy", "E/site", "Time(s)"]
        rows = []

        for h in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            def run(h=h):
                dmrg = DMRG.ising(n_sites=n_sites, J=1.0, h=h, max_bond_dim=32)
                energy, _ = dmrg.run(n_sweeps=15)
                return energy

            try:
                r = bench(run, n_warmup=0, n_runs=3)
                energy = float(r.result)
                rows.append([f"{h:.1f}", f"{energy:.6f}", f"{energy / n_sites:.6f}", f"{r.median:.3f}"])
                self._store(f"ising_h{h:.1f}", {
                    "h": h, "energy": energy,
                    "energy_per_site": energy / n_sites, "time_s": r.median,
                })
            except Exception as e:
                rows.append([f"{h:.1f}", "ERR", "ERR", str(e)[:20]])

        table(headers, rows)

    # ── 5. C++ vs Python backend comparison ────────────────────────────

    def _bench_backend_comparison(self):
        """Compare C++ and Python DMRG backends."""
        from qforge.dmrg import DMRG, _HAS_MPS_CPP

        if not _HAS_MPS_CPP:
            print("\n  --- Backend Comparison: SKIPPED (no C++ backend) ---")
            return

        n_sites = 10  # Keep small so Python backend doesn't take forever
        print(f"\n  --- C++ vs Python Backend: {n_sites}-site Heisenberg, chi=16 ---")
        headers = ["Backend", "Time(s)", "Energy", "Speedup"]
        rows = []

        # C++
        def run_cpp():
            dmrg = DMRG.heisenberg(n_sites=n_sites, max_bond_dim=16, backend="cpu")
            energy, _ = dmrg.run(n_sweeps=10)
            return energy

        try:
            r_cpp = bench(run_cpp, n_warmup=0, n_runs=3)
            t_cpp = r_cpp.median
            rows.append(["C++ (cpu)", f"{t_cpp:.3f}", f"{r_cpp.result:.8f}", "-"])
        except Exception as e:
            t_cpp = None
            rows.append(["C++ (cpu)", "ERR", str(e)[:30], "-"])

        # Python
        def run_py():
            dmrg = DMRG.heisenberg(n_sites=n_sites, max_bond_dim=16, backend="python")
            energy, _ = dmrg.run(n_sweeps=10)
            return energy

        try:
            r_py = bench(run_py, n_warmup=0, n_runs=3)
            t_py = r_py.median
            speedup = f"{t_py / t_cpp:.1f}x" if t_cpp else "N/A"
            rows.append(["Python", f"{t_py:.3f}", f"{r_py.result:.8f}", speedup])
        except Exception as e:
            rows.append(["Python", "ERR", str(e)[:30], "N/A"])

        self._store("backend_comparison", {
            "n_sites": n_sites, "chi": 16,
            "cpp_time": t_cpp, "python_time": r_py.median if 'r_py' in locals() else None,
        })

        table(headers, rows)
