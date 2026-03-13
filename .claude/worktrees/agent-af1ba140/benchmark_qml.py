"""
QML Algorithm Benchmark: Qforge vs PennyLane vs Qiskit
=====================================================

Compares VQE and QAOA across three frameworks on identical problems:
  - Same Hamiltonian / graph
  - Same ansatz structure
  - Same optimizer (GradientDescent lr=0.1)
  - Same gradient method (parameter-shift, shift=pi/2)
  - Same number of steps (50)

Missing packages are auto-installed on first run.

Usage:
    python benchmark_qml.py
"""

import sys
import time
import warnings
import subprocess
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Shared benchmark settings
# ─────────────────────────────────────────────────────────────────────────────

STEPS    = 50        # optimisation steps per run
N_RUNS   = 3        # repeat runs for median timing
LR       = 0.1      # GradientDescent learning rate
SHIFT    = np.pi / 2

# VQE: 2-qubit H₂-like Hamiltonian
#   H = -1.0523·I + 0.3979·Z₀ - 0.3979·Z₁ - 0.0112·Z₀Z₁ + 0.1809·X₀X₁
VQE_COEFFS = [-1.0523, 0.3979, -0.3979, -0.0112, 0.1809]
VQE_N_QUBITS = 2

# QAOA: 4-node ring graph  0-1-2-3-0
QAOA_EDGES    = [(0, 1), (1, 2), (2, 3), (3, 0)]
QAOA_N_QUBITS = 4
QAOA_P        = 1


# ─────────────────────────────────────────────────────────────────────────────
# Package management
# ─────────────────────────────────────────────────────────────────────────────

def _pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _load_pennylane():
    """Import PennyLane, installing if needed. Returns module or None."""
    try:
        import pennylane as qml
        return qml
    except ImportError:
        pass
    try:
        print("  Installing PennyLane ...", end=" ", flush=True)
        _pip_install("pennylane")
        import pennylane as qml
        print("done")
        return qml
    except Exception as e:
        print(f"failed ({e})")
        return None


def _load_qiskit():
    """Import Qiskit stack, installing if needed. Returns True or False."""
    try:
        import qiskit                           # noqa: F401
        import qiskit_algorithms               # noqa: F401
        return True
    except ImportError:
        pass
    try:
        print("  Installing Qiskit ...", end=" ", flush=True)
        _pip_install("qiskit", "qiskit-aer", "qiskit-algorithms")
        import qiskit                           # noqa: F401
        import qiskit_algorithms               # noqa: F401
        print("done")
        return True
    except Exception as e:
        print(f"failed ({e})")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def _median_time(fn, n_runs=N_RUNS):
    """Run fn() n_runs times and return (median_seconds, last_result)."""
    times, result = [], None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


# ─────────────────────────────────────────────────────────────────────────────
# VQE implementations
# ─────────────────────────────────────────────────────────────────────────────

def _vqe_qforge(backend):
    from Qforge.algo import Hamiltonian, VQE, GradientDescent

    H = Hamiltonian(
        coeffs=VQE_COEFFS,
        terms=[[], [('Z', 0)], [('Z', 1)], [('Z', 0), ('Z', 1)], [('X', 0), ('X', 1)]],
    )
    n_p = VQE.n_params_hardware_efficient(VQE_N_QUBITS, n_layers=1)

    def run():
        vqe = VQE(n_qubits=VQE_N_QUBITS, hamiltonian=H, n_layers=1, backend=backend)
        _, history = vqe.optimize(
            np.zeros(n_p), optimizer=GradientDescent(lr=LR), steps=STEPS
        )
        return history[-1]

    return _median_time(run)


def _vqe_pennylane(qml):
    pnp = qml.numpy

    dev = qml.device("default.qubit", wires=VQE_N_QUBITS)

    # Build Hamiltonian (compatible with PennyLane 0.29+)
    obs = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
    ]
    try:
        H_pl = qml.Hamiltonian(pnp.array(VQE_COEFFS, requires_grad=False), obs)
    except Exception:
        H_pl = qml.ops.LinearCombination(VQE_COEFFS, obs)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        # 1-layer hardware-efficient ansatz (matches Qforge)
        qml.RY(params[0], wires=0); qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0); qml.RY(params[3], wires=1)
        return qml.expval(H_pl)

    opt = qml.GradientDescentOptimizer(stepsize=LR)

    def run():
        params = pnp.zeros(4, requires_grad=True)
        cost = None
        for _ in range(STEPS):
            params, cost = opt.step_and_cost(circuit, params)
        return float(cost)

    return _median_time(run)


def _vqe_qiskit():
    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    # Qiskit uses LSB ordering: qubit 0 is the rightmost character
    # Z₀ → "IZ", Z₁ → "ZI", Z₀Z₁ → "ZZ", X₀X₁ → "XX"
    H_qk = SparsePauliOp.from_list([
        ("II", VQE_COEFFS[0]),
        ("IZ", VQE_COEFFS[1]),   # Z₀
        ("ZI", VQE_COEFFS[2]),   # Z₁
        ("ZZ", VQE_COEFFS[3]),
        ("XX", VQE_COEFFS[4]),
    ])

    # 1-layer TwoLocal: RY-CNOT-RY on 2 qubits → 4 parameters (matches Qforge)
    ansatz = TwoLocal(
        VQE_N_QUBITS, rotation_blocks=["ry"], entanglement_blocks="cx",
        reps=1, entanglement="linear"
    )

    estimator = StatevectorEstimator()

    def evaluate(params):
        job = estimator.run([(ansatz, H_qk, params)])
        return float(job.result()[0].data.evs)

    def grad_param_shift(params):
        g = np.zeros_like(params)
        denom = 2.0 * np.sin(SHIFT)
        for i in range(len(params)):
            pp = params.copy(); pp[i] += SHIFT
            pm = params.copy(); pm[i] -= SHIFT
            g[i] = (evaluate(pp) - evaluate(pm)) / denom
        return g

    def run():
        params = np.zeros(ansatz.num_parameters)
        cost = None
        for _ in range(STEPS):
            cost = evaluate(params)
            params -= LR * grad_param_shift(params)
        return cost

    return _median_time(run)


# ─────────────────────────────────────────────────────────────────────────────
# QAOA implementations
# ─────────────────────────────────────────────────────────────────────────────

def _qaoa_qforge(backend):
    from Qforge.algo import QAOA, GradientDescent

    def run():
        qaoa = QAOA(n_qubits=QAOA_N_QUBITS, edges=QAOA_EDGES,
                    p_layers=QAOA_P, backend=backend)
        params0 = np.full(qaoa.n_params, 0.5)
        _, history = qaoa.optimize(
            params0, optimizer=GradientDescent(lr=LR), steps=STEPS
        )
        # Return the achieved cut value (expectation, not sampled)
        return -history[-1]   # stored as negative (minimisation)

    return _median_time(run)


def _qaoa_pennylane(qml):
    pnp = qml.numpy

    dev = qml.device("default.qubit", wires=QAOA_N_QUBITS)

    # Cost Hamiltonian H_C = 2·I - 0.5·(Z0Z1 + Z1Z2 + Z2Z3 + Z3Z0)
    obs_qaoa = [
        qml.Identity(0),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2) @ qml.PauliZ(3),
        qml.PauliZ(3) @ qml.PauliZ(0),
    ]
    coeffs_qaoa = pnp.array([2.0, -0.5, -0.5, -0.5, -0.5], requires_grad=False)
    try:
        H_cost = qml.Hamiltonian(coeffs_qaoa, obs_qaoa)
    except Exception:
        H_cost = qml.ops.LinearCombination(coeffs_qaoa, obs_qaoa)

    @qml.qnode(dev, diff_method="parameter-shift")
    def qaoa_circuit(params):
        gamma, beta = params[0], params[1]
        # Initial superposition
        for w in range(QAOA_N_QUBITS):
            qml.Hadamard(wires=w)
        # Problem unitary (matches Qforge QAOA exactly)
        for (i, j) in QAOA_EDGES:
            qml.CNOT(wires=[i, j])
            qml.RZ(-gamma, wires=j)
            qml.CNOT(wires=[i, j])
        # Mixer unitary
        for w in range(QAOA_N_QUBITS):
            qml.RX(2.0 * beta, wires=w)
        return qml.expval(H_cost)

    # Wrap with negation so GradientDescentOptimizer MAXIMISES the cut value
    def neg_qaoa(params):
        return -qaoa_circuit(params)

    opt = qml.GradientDescentOptimizer(stepsize=LR)

    def run():
        params = pnp.array([0.5, 0.5], requires_grad=True)
        neg_cost = None
        for _ in range(STEPS):
            params, neg_cost = opt.step_and_cost(neg_qaoa, params)
        # neg_cost = -<H_C>; return actual cut expectation
        return float(-neg_cost)

    return _median_time(run)


def _qaoa_qiskit():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    # Cost Hamiltonian (4 qubits, Qiskit LSB: qubit 0 = rightmost char)
    # Z0Z1 → "IIZZ", Z1Z2 → "IZZI", Z2Z3 → "ZZII", Z3Z0 → "ZIIZ"
    H_cost_qk = SparsePauliOp.from_list([
        ("IIII", 2.0),
        ("IIZZ", -0.5),   # Z₀Z₁
        ("IZZI", -0.5),   # Z₁Z₂
        ("ZZII", -0.5),   # Z₂Z₃
        ("ZIIZ", -0.5),   # Z₃Z₀
    ])

    # Build QAOA circuit matching Qforge's structure exactly
    gamma = ParameterVector("γ", QAOA_P)
    beta  = ParameterVector("β", QAOA_P)
    params_list = [*gamma, *beta]

    qc = QuantumCircuit(QAOA_N_QUBITS)
    qc.h(range(QAOA_N_QUBITS))
    for l in range(QAOA_P):
        for (i, j) in QAOA_EDGES:
            qc.cx(i, j)
            qc.rz(-gamma[l], j)
            qc.cx(i, j)
        for w in range(QAOA_N_QUBITS):
            qc.rx(2.0 * beta[l], w)

    estimator = StatevectorEstimator()

    def evaluate(params):
        job = estimator.run([(qc, H_cost_qk, params)])
        return float(job.result()[0].data.evs)

    def grad_param_shift(params):
        g = np.zeros_like(params)
        denom = 2.0 * np.sin(SHIFT)
        for i in range(len(params)):
            pp = params.copy(); pp[i] += SHIFT
            pm = params.copy(); pm[i] -= SHIFT
            g[i] = (evaluate(pp) - evaluate(pm)) / denom
        return g

    def run():
        params = np.array([0.5, 0.5])   # [gamma, beta]
        cut = None
        for _ in range(STEPS):
            cut = evaluate(params)
            # Maximise <H_C> = cut expectation: step in +gradient direction
            params += LR * grad_param_shift(params)
        return cut

    return _median_time(run)


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

_COL = 18

def _header(title):
    width = 60
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'Framework':<{_COL}}  {'ms/step':>10}  {'total(s)':>10}  {'result':>12}")
    print("-" * width)

def _row(name, total_s, result, label=""):
    ms_step = total_s / STEPS * 1000
    result_str = f"{result:.4f}" if result is not None else "N/A"
    label_str = f"  {label}" if label else ""
    print(f"{name:<{_COL}}  {ms_step:>10.2f}  {total_s:>10.3f}  {result_str:>12}{label_str}")

def _row_na(name, reason=""):
    r = f"  ({reason})" if reason else ""
    print(f"{name:<{_COL}}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}{r}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def run_vqe_benchmark(qml, has_qiskit):
    from Qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL

    _header(f"VQE  |  {VQE_N_QUBITS} qubits, H₂-like Hamiltonian, {STEPS} steps, lr={LR}")

    # Qforge backends
    for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA), ("metal", _HAS_METAL)]:
        if not available:
            continue
        try:
            t, energy = _vqe_qforge(backend)
            _row(f"qforge ({backend})", t, energy)
        except Exception as e:
            _row_na(f"qforge ({backend})", str(e)[:40])

    # PennyLane
    if qml is not None:
        try:
            t, energy = _vqe_pennylane(qml)
            _row("PennyLane", t, energy, f"(v{qml.__version__})")
        except Exception as e:
            _row_na("PennyLane", str(e)[:40])
    else:
        _row_na("PennyLane", "not installed")

    # Qiskit
    if has_qiskit:
        try:
            t, energy = _vqe_qiskit()
            import qiskit
            _row("Qiskit", t, energy, f"(v{qiskit.__version__})")
        except Exception as e:
            _row_na("Qiskit", str(e)[:40])
    else:
        _row_na("Qiskit", "not installed")


def run_qaoa_benchmark(qml, has_qiskit):
    from Qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL

    _header(
        f"QAOA |  {QAOA_N_QUBITS}-node ring graph, p={QAOA_P}, {STEPS} steps, lr={LR}"
        f"\n       edges={QAOA_EDGES}"
    )

    for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA), ("metal", _HAS_METAL)]:
        if not available:
            continue
        try:
            t, cut = _qaoa_qforge(backend)
            _row(f"qforge ({backend})", t, cut, "(expected cut)")
        except Exception as e:
            _row_na(f"qforge ({backend})", str(e)[:40])

    if qml is not None:
        try:
            t, cut = _qaoa_pennylane(qml)
            _row("PennyLane", t, cut, f"(v{qml.__version__})")
        except Exception as e:
            _row_na("PennyLane", str(e)[:40])
    else:
        _row_na("PennyLane", "not installed")

    if has_qiskit:
        try:
            t, cut = _qaoa_qiskit()
            import qiskit
            _row("Qiskit", t, cut, f"(v{qiskit.__version__})")
        except Exception as e:
            _row_na("Qiskit", str(e)[:40])
    else:
        _row_na("Qiskit", "not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Scaling benchmark  (--scale flag)
# ─────────────────────────────────────────────────────────────────────────────

SCALE_QUBITS  = [4, 6, 8, 10, 12, 14, 16, 18, 20]
SCALE_NLAYERS = 1   # hardware_efficient_ansatz layers


def _one_vqe_step(n_qubits, backend):
    """Time a single VQE gradient + update step (parameter-shift, n_layers=1)."""
    from Qforge.algo import Hamiltonian, VQE, GradientDescent

    # Random-coefficient all-Z Hamiltonian — cost is fast, scales with qubits
    rng = np.random.default_rng(0)
    coeffs = rng.uniform(-1, 1, n_qubits).tolist()
    terms  = [[('Z', q)] for q in range(n_qubits)]
    H = Hamiltonian(coeffs, terms)

    n_p = VQE.n_params_hardware_efficient(n_qubits, SCALE_NLAYERS)
    vqe = VQE(n_qubits=n_qubits, hamiltonian=H,
              n_layers=SCALE_NLAYERS, backend=backend)

    params = np.zeros(n_p)
    # Warmup
    vqe._evaluate(params)
    vqe.gradient(params)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        vqe.gradient(params)      # 2*n_params circuit evaluations
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_scaling_benchmark():
    from Qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL

    width = 72
    print("\n" + "=" * width)
    print(f"SCALING  |  VQE, hardware-efficient ansatz (layers={SCALE_NLAYERS}), 1 gradient step")
    print(f"         |  n_params = n_qubits × {SCALE_NLAYERS+1}  (parameter-shift: 2×n_params evals/step)")
    print("=" * width)

    backends = [(b, a) for b, a in
                [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA), ("metal", _HAS_METAL)]
                if a]

    # Header
    hdr = f"{'qubits':>7}  {'n_params':>9}"
    for b, _ in backends:
        hdr += f"  {b+' (ms)':>12}"
    if len(backends) > 1:
        hdr += f"  {'speedup':>9}"
    print(hdr)
    print("-" * width)

    for nq in SCALE_QUBITS:
        n_p = nq * (SCALE_NLAYERS + 1)
        row = f"{nq:>7}  {n_p:>9}"
        times = {}
        for b, _ in backends:
            try:
                t = _one_vqe_step(nq, b)
                times[b] = t
                row += f"  {t*1000:>12.1f}"
            except Exception as e:
                row += f"  {'ERR':>12}"
                times[b] = None

        if len(backends) > 1:
            t_first = times.get(backends[0][0])
            t_last  = times.get(backends[-1][0])
            if t_first and t_last:
                row += f"  {t_first/t_last:>8.1f}x"

        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from Qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL

    ap = argparse.ArgumentParser(description="QML benchmark: Qforge vs PennyLane vs Qiskit")
    ap.add_argument("--scale", action="store_true",
                    help="Run multi-qubit scaling benchmark (Qforge backends only)")
    ap.add_argument("--all", action="store_true",
                    help="Run both framework comparison and scaling benchmark")
    args = ap.parse_args()

    print("=" * 60)
    print("QML Algorithm Benchmark: Qforge vs PennyLane vs Qiskit")
    print("=" * 60)
    print(f"Settings: steps={STEPS}, lr={LR}, grad=parameter-shift(π/2)")
    print(f"qforge backends — CPU:{_HAS_CPP}  CUDA:{_HAS_CUDA}  Metal:{_HAS_METAL}")

    if args.scale:
        run_scaling_benchmark()
    else:
        print()
        print("Checking external frameworks ...")
        qml        = _load_pennylane()
        has_qiskit = _load_qiskit()

        run_vqe_benchmark(qml, has_qiskit)
        run_qaoa_benchmark(qml, has_qiskit)

        if args.all:
            run_scaling_benchmark()

    print()
