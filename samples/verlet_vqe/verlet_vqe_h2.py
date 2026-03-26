# -*- coding: utf-8 -*-
"""Reproduce arXiv:2603.09862 — Velocity Verlet VQE optimizer for H2.

Paper: "Velocity Verlet-Based Optimization for Variational Quantum
        Eigensolvers" by Rinka Miura (March 2026).

This script compares the velocity Verlet optimizer against Adam,
L-BFGS-B (with analytical gradient), and COBYLA on the H2 ground-state
problem (4 qubits, STO-3G, Jordan-Wigner) using qforge.

Ansatz: depth-4 hardware-efficient with RY/RZ rotation layers + CZ ladder,
matching the paper's circuit architecture.

Usage:
    python samples/verlet_vqe/verlet_vqe_h2.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as sp_minimize

from qforge.algo import Adam
from qforge.algo.gradient import parameter_shift
from qforge.algo.vqa import VQA
from qforge.chem import Molecule
from qforge.gates import RY, RZ, CPhase


# ---------------------------------------------------------------------------
# Velocity Verlet optimizer (arXiv:2603.09862)
# ---------------------------------------------------------------------------

class VelocityVerlet:
    """Velocity Verlet integrator adapted for VQE optimization.

    Update equations (Miura 2026):
        v(t+dt/2) = v(t) + dt * F(t) / (2m)
        theta(t+dt) = theta(t) + dt * v(t+dt/2)
        v(t+dt) = v(t+dt/2) + dt * F(t+dt) / (2m)
        v(t+dt) *= (1 - gamma)

    where F(theta) = -grad E(theta).
    """

    def __init__(self, dt: float = 0.1, mass: float = 1.0, gamma: float = 0.1):
        self.dt = dt
        self.mass = mass
        self.gamma = gamma
        self._velocity = None
        self._prev_force = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        force = -np.asarray(grad, dtype=float)

        if self._velocity is None:
            self._velocity = np.zeros_like(params)
            self._prev_force = force.copy()

        v_half = self._velocity + 0.5 * self.dt * self._prev_force / self.mass
        new_params = params + self.dt * v_half
        self._velocity = v_half + 0.5 * self.dt * force / self.mass
        self._velocity *= (1.0 - self.gamma)
        self._prev_force = force.copy()

        return new_params


# ---------------------------------------------------------------------------
# Paper ansatz: depth-4 RY/RZ + CZ ladder (arXiv:2603.09862)
# ---------------------------------------------------------------------------

def paper_ansatz(wf, params: np.ndarray, n_layers: int = 4) -> None:
    """Hardware-efficient ansatz matching the paper's description.

    Each layer: RY(q) + RZ(q) for all qubits, then CZ ladder on neighbors.
    Final layer: RY(q) + RZ(q) without trailing CZ.

    n_params = n_qubits * (n_layers + 1) * 2
    """
    n_qubits = len(wf.state[0])
    idx = 0
    for layer in range(n_layers + 1):
        for q in range(n_qubits):
            RY(wf, q, params[idx]); idx += 1
            RZ(wf, q, params[idx]); idx += 1
        if layer < n_layers:
            for q in range(n_qubits - 1):
                CPhase(wf, q, q + 1, np.pi)  # CZ = CPhase(pi)


def paper_n_params(n_qubits: int, n_layers: int = 4) -> int:
    return n_qubits * (n_layers + 1) * 2


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

H2_BOND_LENGTH = 0.74   # angstrom
N_LAYERS = 4
MAX_STEPS = 200
CHEMICAL_ACCURACY = 1.6e-3  # Hartree
SEED = 42

# Exact electronic ground-state energy (from diagonalization)
EXACT_ENERGY = -1.83043


def _print_result(name: str, history: list[float], e_nuc: float):
    """Print a one-line result summary for an optimizer run."""
    final = history[-1]
    error = abs(final - EXACT_ENERGY)
    chem_step = next(
        (i for i, e in enumerate(history) if abs(e - EXACT_ENERGY) < CHEMICAL_ACCURACY),
        None,
    )
    total = final + e_nuc
    print(f"  [{name:25s}] E_elec={final:+.10f}  E_tot={total:+.8f}  "
          f"err={error:.2e}  evals={len(history):>5d}  "
          f"chem_acc@{'step ' + str(chem_step) if chem_step is not None else '---'}")


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(results: dict, exact: float, save_path: str | None = None):
    if save_path is None:
        save_path = str(_SCRIPT_DIR / "verlet_vqe_convergence.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        return

    colors = {
        "VV (grid-tuned)": "#1f77b4",
        "Velocity Verlet": "#aec7e8",
        "Adam": "#ff7f0e",
        "L-BFGS-B": "#2ca02c",
        "COBYLA": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in results.items():
        errors = [abs(e - exact) for e in hist]
        style = {"linewidth": 2.0, "alpha": 1.0} if name == "VV (grid-tuned)" else {"linewidth": 1.2, "alpha": 0.8}
        ax.semilogy(range(len(errors)), errors, label=name,
                     color=colors.get(name), **style)

    ax.axhline(CHEMICAL_ACCURACY, color="gray", linestyle="--", alpha=0.6,
               label=f"Chemical accuracy ({CHEMICAL_ACCURACY} Ha)")
    ax.set_xlabel("Function evaluation", fontsize=12)
    ax.set_ylabel("|E - E_exact| (Ha)", fontsize=12)
    ax.set_title("VQE Optimizer Comparison — H$_2$ (4 qubits, STO-3G)\n"
                 "Ansatz: depth-4 RY/RZ + CZ ladder (arXiv:2603.09862)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nConvergence plot saved to {save_path}")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main():
    mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, H2_BOND_LENGTH))])
    hamiltonian = mol.hamiltonian()
    n_qubits = mol.n_qubits
    n_params = paper_n_params(n_qubits, N_LAYERS)
    E_NUC = mol.nuclear_repulsion

    print(f"H2 ({n_qubits} qubits) | {n_params} params | depth-{N_LAYERS} RY/RZ+CZ")
    print(f"Nuclear repulsion: {E_NUC:.6f} Ha")
    print(f"Exact electronic energy: {EXACT_ENERGY:.5f} Ha")
    print(f"Exact total energy: {EXACT_ENERGY + E_NUC:.5f} Ha")
    print(f"Chemical accuracy threshold: {CHEMICAL_ACCURACY} Ha")
    print("=" * 70)

    np.random.seed(SEED)
    init_params = np.random.uniform(-np.pi, np.pi, n_params)

    def make_vqa():
        return VQA(
            n_qubits=n_qubits,
            circuit_fn=lambda wf, p: paper_ansatz(wf, p, N_LAYERS),
            cost_fn=hamiltonian.expectation,
        )

    def run_step_opt(optimizer, name, steps=MAX_STEPS):
        vqa = make_vqa()
        _, history = vqa.optimize(init_params.copy(), optimizer=optimizer, steps=steps)
        _print_result(name, history, E_NUC)
        return history

    def run_lbfgs(name="L-BFGS-B", maxiter=MAX_STEPS):
        vqa = make_vqa()
        history = []
        def cost_and_grad(p):
            e = vqa._evaluate(p)
            g = parameter_shift(vqa._evaluate, p)
            history.append(e)
            return e, g
        res = sp_minimize(
            cost_and_grad, init_params.copy(), method="L-BFGS-B", jac=True,
            options={"maxiter": maxiter},
        )
        if len(history) == 0 or history[-1] != res.fun:
            history.append(res.fun)
        _print_result(name, history, E_NUC)
        return history

    def run_cobyla(name="COBYLA", maxiter=MAX_STEPS * 5):
        vqa = make_vqa()
        history = []
        def cost_fn(p):
            e = vqa._evaluate(p)
            history.append(e)
            return e
        sp_minimize(cost_fn, init_params.copy(), method="COBYLA",
                    options={"maxiter": maxiter})
        _print_result(name, history, E_NUC)
        return history

    # --- Run all optimizers with default hyperparameters ---
    print("\n--- Phase 1: Default hyperparameters ---")
    results = {}

    opt_vv = VelocityVerlet(dt=0.2, mass=1.0, gamma=0.2)
    results["Velocity Verlet"] = run_step_opt(opt_vv, "Velocity Verlet")

    opt_adam = Adam(lr=0.05)
    results["Adam"] = run_step_opt(opt_adam, "Adam")

    results["L-BFGS-B"] = run_lbfgs()

    results["COBYLA"] = run_cobyla()

    # --- Grid search for Velocity Verlet hyperparameters ---
    print("\n--- Phase 2: Velocity Verlet grid search ---")

    dt_range = [0.10, 0.20, 0.30]
    gamma_range = [0.10, 0.20, 0.30]
    mass_range = [0.5, 1.0]

    best_error = float("inf")
    best_cfg = None
    grid_results = []

    GRID_STEPS = 100
    for dt, gamma, mass in itertools.product(dt_range, gamma_range, mass_range):
        vqa = make_vqa()
        opt = VelocityVerlet(dt=dt, mass=mass, gamma=gamma)
        _, hist = vqa.optimize(init_params.copy(), optimizer=opt, steps=GRID_STEPS)
        err = abs(hist[-1] - EXACT_ENERGY)
        chem = next(
            (i for i, e in enumerate(hist) if abs(e - EXACT_ENERGY) < CHEMICAL_ACCURACY),
            None,
        )
        grid_results.append((dt, gamma, mass, err, chem, hist[-1]))
        if err < best_error:
            best_error = err
            best_cfg = (dt, gamma, mass)

    grid_results.sort(key=lambda x: x[3])
    print(f"\n  Top-5 configurations (out of {len(grid_results)}):")
    print(f"  {'dt':>6s}  {'gamma':>6s}  {'mass':>5s}  {'error':>10s}  {'chem@step':>10s}  {'E_elec':>14s}")
    for dt, gamma, mass, err, chem, e_final in grid_results[:5]:
        cs = str(chem) if chem is not None else "---"
        print(f"  {dt:6.2f}  {gamma:6.2f}  {mass:5.1f}  {err:10.2e}  {cs:>10s}  {e_final:+14.10f}")

    print(f"\n  Best: dt={best_cfg[0]}, gamma={best_cfg[1]}, mass={best_cfg[2]} "
          f"-> error={best_error:.2e}")

    # --- Re-run best config with full steps ---
    print(f"\n--- Phase 3: Re-run best VV config with {MAX_STEPS} steps ---")
    opt_best = VelocityVerlet(dt=best_cfg[0], mass=best_cfg[2], gamma=best_cfg[1])
    results["VV (grid-tuned)"] = run_step_opt(
        opt_best, f"VV(dt={best_cfg[0]},g={best_cfg[1]},m={best_cfg[2]})", MAX_STEPS
    )

    # --- Convergence plot ---
    plot_convergence(results, EXACT_ENERGY)


if __name__ == "__main__":
    main()
