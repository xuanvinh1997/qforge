# -*- coding: utf-8 -*-
"""Category 10: Matrix Product State (MPS) benchmarks.

Qforge-specific: compares MPS backend vs statevector and measures scaling
characteristics (qubit count, bond dimension, accuracy).
"""
from __future__ import annotations

import numpy as np

from qforge.benchmarks.core import (
    BaseBenchmarkSuite, bench, section, table,
)


class MPSBenchmarkSuite(BaseBenchmarkSuite):
    name = "mps"
    description = "Matrix Product State (MPS) Benchmarks"

    def run(self):
        section(f"CATEGORY 10: {self.description}", "")
        self._bench_gate_application()
        self._bench_qubit_scaling()
        self._bench_bond_dim_scaling()
        self._bench_accuracy_vs_bond_dim()
        self._bench_entanglement_entropy()
        return self._results

    # ── 1. MPS vs StateVector gate application ─────────────────────────

    def _bench_gate_application(self):
        """GHZ circuit: H on q0, then CNOT chain."""
        from qforge import _HAS_CPP
        from qforge.mps import MatrixProductState, _HAS_MPS_CPP

        print("\n  --- MPS vs StateVector: GHZ circuit ---")
        headers = ["Qubits", "StateVec(ms)", "MPS chi=64(ms)", "MPS chi=128(ms)", "Speedup(64)"]
        rows = []

        qubit_list = [n for n in [8, 12, 16, 20] if n <= self.config.max_qubits]
        if _HAS_MPS_CPP:
            qubit_list += [n for n in [30, 50] if n > self.config.max_qubits]

        for nq in qubit_list:
            t_sv = t_mps64 = t_mps128 = None

            # StateVector (only feasible up to ~24 qubits)
            if _HAS_CPP and nq <= min(self.config.max_qubits, 24):
                from qforge.circuit import Qubit
                from qforge.gates import H, CNOT
                def run_sv(nq=nq):
                    wf = Qubit(nq, backend="cpu")
                    H(wf, 0)
                    for q in range(nq - 1): CNOT(wf, q, q + 1)
                    _ = wf.amplitude[0]
                try:
                    t_sv = bench(run_sv, n_warmup=2, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass

            # MPS chi=64
            if _HAS_MPS_CPP:
                from qforge.gates import H, CNOT
                def run_mps64(nq=nq):
                    psi = MatrixProductState(nq, max_bond_dim=64)
                    H(psi, 0)
                    for q in range(nq - 1): CNOT(psi, q, q + 1)
                    _ = psi.bond_dimensions()
                try:
                    t_mps64 = bench(run_mps64, n_warmup=2, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass

            # MPS chi=128
            if _HAS_MPS_CPP:
                def run_mps128(nq=nq):
                    psi = MatrixProductState(nq, max_bond_dim=128)
                    H(psi, 0)
                    for q in range(nq - 1): CNOT(psi, q, q + 1)
                    _ = psi.bond_dimensions()
                try:
                    t_mps128 = bench(run_mps128, n_warmup=2, n_runs=self.config.n_runs).median * 1000
                except Exception:
                    pass

            speedup = f"{t_sv / t_mps64:.1f}x" if t_sv and t_mps64 else "N/A"
            rows.append([
                nq,
                f"{t_sv:.3f}" if t_sv else "N/A",
                f"{t_mps64:.3f}" if t_mps64 else "N/A",
                f"{t_mps128:.3f}" if t_mps128 else "N/A",
                speedup,
            ])
            self._store(f"gate_ghz_{nq}q", {
                "qubits": nq, "statevector": t_sv,
                "mps_chi64": t_mps64, "mps_chi128": t_mps128,
            })
        table(headers, rows)

    # ── 2. MPS qubit scaling (low-entanglement) ────────────────────────

    def _bench_qubit_scaling(self):
        """Time for a low-entanglement circuit as qubit count grows."""
        from qforge.mps import MatrixProductState, _HAS_MPS_CPP
        if not _HAS_MPS_CPP:
            print("\n  --- MPS Qubit Scaling: SKIPPED (no C++ MPS backend) ---")
            return

        from qforge.gates import H, CNOT, RY

        print("\n  --- MPS Qubit Scaling (chi=64, layered RY+CNOT) ---")
        headers = ["Qubits", "Time(ms)", "Max bond dim"]
        rows = []

        for nq in [20, 50, 100, 200, 500]:
            def run_mps(nq=nq):
                psi = MatrixProductState(nq, max_bond_dim=64)
                # Layer of RY + nearest-neighbor CNOT
                for q in range(nq): RY(psi, q, 0.3)
                for q in range(0, nq - 1, 2): CNOT(psi, q, q + 1)
                for q in range(1, nq - 1, 2): CNOT(psi, q, q + 1)
                return max(psi.bond_dimensions())

            try:
                r = bench(run_mps, n_warmup=1, n_runs=max(3, self.config.n_runs))
                t_ms = r.median * 1000
                max_chi = r.result
                rows.append([nq, f"{t_ms:.2f}", max_chi])
                self._store(f"qubit_scale_{nq}q", {"qubits": nq, "time_ms": t_ms, "max_chi": int(max_chi)})
            except Exception as e:
                rows.append([nq, "ERR", str(e)[:30]])

        table(headers, rows)

    # ── 3. Bond dimension scaling ──────────────────────────────────────

    def _bench_bond_dim_scaling(self):
        """Time vs max_bond_dim for a fixed 30-qubit circuit."""
        from qforge.mps import MatrixProductState, _HAS_MPS_CPP
        if not _HAS_MPS_CPP:
            print("\n  --- Bond Dim Scaling: SKIPPED ---")
            return

        from qforge.gates import H, CNOT, RY

        nq = min(30, max(self.config.max_qubits, 30))
        print(f"\n  --- Bond Dimension Scaling ({nq}q, layered circuit) ---")
        headers = ["Max chi", "Time(ms)", "Actual max chi"]
        rows = []

        for chi in [4, 8, 16, 32, 64, 128, 256]:
            def run(chi=chi):
                psi = MatrixProductState(nq, max_bond_dim=chi)
                for q in range(nq): H(psi, q)
                for q in range(nq - 1): CNOT(psi, q, q + 1)
                return max(psi.bond_dimensions())

            try:
                r = bench(run, n_warmup=1, n_runs=max(3, self.config.n_runs))
                t_ms = r.median * 1000
                actual_chi = r.result
                rows.append([chi, f"{t_ms:.2f}", actual_chi])
                self._store(f"bonddim_{chi}", {"max_chi": chi, "time_ms": t_ms, "actual_chi": int(actual_chi)})
            except Exception as e:
                rows.append([chi, "ERR", str(e)[:30]])

        table(headers, rows)

    # ── 4. Accuracy vs bond dimension ──────────────────────────────────

    def _bench_accuracy_vs_bond_dim(self):
        """Fidelity of MPS vs exact statevector for small circuits."""
        from qforge import _HAS_CPP
        from qforge.mps import MatrixProductState, _HAS_MPS_CPP
        if not _HAS_CPP or not _HAS_MPS_CPP:
            print("\n  --- MPS Accuracy: SKIPPED ---")
            return

        from qforge.circuit import Qubit
        from qforge.gates import H, CNOT, RY

        nq = min(12, self.config.max_qubits)
        print(f"\n  --- MPS Accuracy vs Bond Dim ({nq}q random circuit) ---")

        # Build exact statevector
        rng = np.random.RandomState(self.config.seed)
        wf = Qubit(nq, backend="cpu")
        for q in range(nq): H(wf, q)
        for q in range(nq - 1): CNOT(wf, q, q + 1)
        for q in range(nq): RY(wf, q, rng.uniform(0, 2 * np.pi))
        for q in range(nq - 1): CNOT(wf, q, q + 1)
        exact = wf.amplitude.copy()

        headers = ["Max chi", "Fidelity", "Max |error|"]
        rows = []

        for chi in [2, 4, 8, 16, 32, 64]:
            psi = MatrixProductState(nq, max_bond_dim=chi)
            for q in range(nq): H(psi, q)
            for q in range(nq - 1): CNOT(psi, q, q + 1)
            rng2 = np.random.RandomState(self.config.seed)
            for q in range(nq): RY(psi, q, rng2.uniform(0, 2 * np.pi))
            for q in range(nq - 1): CNOT(psi, q, q + 1)

            mps_amp = psi.amplitude
            fidelity = float(np.abs(np.vdot(exact, mps_amp)) ** 2)
            max_err = float(np.max(np.abs(exact - mps_amp)))

            rows.append([chi, f"{fidelity:.8f}", f"{max_err:.2e}"])
            self._store(f"accuracy_chi{chi}", {"chi": chi, "fidelity": fidelity, "max_error": max_err})

        table(headers, rows)

    # ── 5. Entanglement entropy computation ────────────────────────────

    def _bench_entanglement_entropy(self):
        """Time to compute entanglement entropy at all bonds."""
        from qforge.mps import MatrixProductState, _HAS_MPS_CPP
        if not _HAS_MPS_CPP:
            print("\n  --- Entanglement Entropy: SKIPPED ---")
            return

        from qforge.gates import H, CNOT

        print("\n  --- Entanglement Entropy Computation Time ---")
        headers = ["Qubits", "chi", "Entropy time(us)", "Max entropy(bits)"]
        rows = []

        for nq in [20, 50, 100]:
            psi = MatrixProductState(nq, max_bond_dim=64)
            H(psi, 0)
            for q in range(nq - 1): CNOT(psi, q, q + 1)

            def run_ee(psi=psi, nq=nq):
                return max(psi.entanglement_entropy(b) for b in range(nq - 1))

            try:
                r = bench(run_ee, n_warmup=2, n_runs=self.config.n_runs)
                t_us = r.median * 1e6
                max_ee = r.result
                rows.append([nq, 64, f"{t_us:.1f}", f"{max_ee:.4f}"])
                self._store(f"entropy_{nq}q", {"qubits": nq, "time_us": t_us, "max_entropy": float(max_ee)})
            except Exception as e:
                rows.append([nq, 64, "ERR", str(e)[:30]])

        table(headers, rows)
