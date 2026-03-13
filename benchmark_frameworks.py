"""
Comprehensive Benchmark Suite: Qforge vs PennyLane vs Qiskit
=============================================================

Deep comparison across 9 categories:
  1. Primitive Gate Operations
  2. Circuit Execution Patterns
  3. VQE Algorithm
  4. QAOA Algorithm
  5. Gradient Computation
  6. Measurement Operations
  7. Scalability
  8. Accuracy & Correctness
  9. Memory Usage

Usage:
    python benchmark_frameworks.py                        # run all
    python benchmark_frameworks.py --categories gates vqe # select categories
    python benchmark_frameworks.py --max-qubits 16        # limit qubit range
    python benchmark_frameworks.py --json results.json    # export JSON
"""

import sys
import time
import json
import warnings
import tracemalloc
import subprocess
import argparse
from collections import namedtuple
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_WARMUP = 2
N_RUNS   = 5
SEED     = 42
SHIFT    = np.pi / 2
LR       = 0.1
STEPS    = 50

BenchResult = namedtuple("BenchResult", ["median", "min", "max", "result"])

ALL_CATEGORIES = [
    "gates", "circuits", "vqe", "qaoa",
    "gradient", "measurement", "scaling",
    "accuracy", "memory",
]

import os
_BENCH_CUDA = os.environ.get("QFORGE_BENCH_CUDA", "") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# Package management
# ─────────────────────────────────────────────────────────────────────────────

def _pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

_qml = None
_has_qiskit = False
_has_lightning = False

def _load_pennylane():
    global _qml
    try:
        import pennylane as qml
        _qml = qml
        return qml
    except ImportError:
        pass
    try:
        print("  Installing PennyLane ...", end=" ", flush=True)
        _pip_install("pennylane")
        import pennylane as qml
        _qml = qml
        print("done")
        return qml
    except Exception as e:
        print(f"failed ({e})")
        return None

def _load_qiskit():
    global _has_qiskit
    try:
        import qiskit
        _has_qiskit = True
        return True
    except ImportError:
        pass
    try:
        print("  Installing Qiskit ...", end=" ", flush=True)
        _pip_install("qiskit")
        import qiskit
        _has_qiskit = True
        print("done")
        return True
    except Exception as e:
        print(f"failed ({e})")
        return False

def _check_lightning():
    global _has_lightning
    try:
        import pennylane_lightning  # noqa: F401
        _has_lightning = True
    except ImportError:
        _has_lightning = False

# ─────────────────────────────────────────────────────────────────────────────
# Timing & memory utilities
# ─────────────────────────────────────────────────────────────────────────────

def bench(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Run fn() with warmup, return BenchResult(median, min, max, last_result)."""
    result = None
    for _ in range(n_warmup):
        result = fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return BenchResult(
        median=float(np.median(times)),
        min=times[0],
        max=times[-1],
        result=result,
    )

def measure_memory(fn):
    """Run fn() and return peak memory in MB (Python-side only)."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)

# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

_WIDTH = 72
_results_json = {}

def _section(title, desc=""):
    print("\n" + "=" * _WIDTH)
    print(f"  {title}")
    if desc:
        print(f"  {desc}")
    print("=" * _WIDTH)

def _table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for r in rows:
                w = max(w, len(str(r[i])) if i < len(r) else 0)
            col_widths.append(w + 2)
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(col_widths))
    print(fmt.format(*headers))
    print("-" * _WIDTH)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

def _store(category, key, data):
    """Store result for JSON export."""
    if category not in _results_json:
        _results_json[category] = {}
    _results_json[category][key] = data


# ─────────────────────────────────────────────────────────────────────────────
# Category 1: Primitive Gate Operations
# ─────────────────────────────────────────────────────────────────────────────

def _gate_bench_qforge(gate_name, n_qubits, n_reps=1000):
    """Time a single gate applied n_reps times in qforge."""
    from qforge.circuit import Qubit
    from qforge import gates as G, _HAS_CPP

    if not _HAS_CPP:
        return None

    gate_fn, args = {
        "H":     (G.H,    lambda wf: (wf, 0)),
        "X":     (G.X,    lambda wf: (wf, 0)),
        "RX":    (G.RX,   lambda wf: (wf, 0, np.pi / 3)),
        "RZ":    (G.RZ,   lambda wf: (wf, 0, np.pi / 6)),
        "CNOT":  (G.CNOT, lambda wf: (wf, 0, 1)),
        "SWAP":  (G.SWAP, lambda wf: (wf, 0, 1)),
        "CCNOT": (G.CCNOT, lambda wf: (wf, 0, 1, 2)),
    }[gate_name]

    def run():
        wf = Qubit(n_qubits, backend="cpu")
        a = args(wf)
        for _ in range(n_reps):
            gate_fn(*a)
        # Force read to ensure computation is done
        _ = wf.amplitude[0]

    r = bench(run, n_warmup=2, n_runs=N_RUNS)
    return r.median / n_reps * 1e6  # microseconds per gate


def _gate_bench_pennylane(gate_name, n_qubits, n_reps=1000, device_name="default.qubit"):
    """Time a single gate in pennylane by building a circuit with n_reps gates."""
    qml = _qml
    if qml is None:
        return None

    dev = qml.device(device_name, wires=n_qubits)

    gate_map = {
        "H":     lambda: qml.Hadamard(wires=0),
        "X":     lambda: qml.PauliX(wires=0),
        "RX":    lambda: qml.RX(np.pi / 3, wires=0),
        "RZ":    lambda: qml.RZ(np.pi / 6, wires=0),
        "CNOT":  lambda: qml.CNOT(wires=[0, 1]),
        "SWAP":  lambda: qml.SWAP(wires=[0, 1]),
        "CCNOT": lambda: qml.Toffoli(wires=[0, 1, 2]),
    }
    gate_fn = gate_map[gate_name]

    @qml.qnode(dev)
    def circuit():
        for _ in range(n_reps):
            gate_fn()
        return qml.state()

    r = bench(circuit, n_warmup=2, n_runs=N_RUNS)
    return r.median / n_reps * 1e6  # microseconds per gate


def _gate_bench_qiskit(gate_name, n_qubits, n_reps=1000):
    """Time a single gate in qiskit Statevector."""
    if not _has_qiskit:
        return None

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(n_qubits)
    gate_map = {
        "H":     lambda: qc.h(0),
        "X":     lambda: qc.x(0),
        "RX":    lambda: qc.rx(np.pi / 3, 0),
        "RZ":    lambda: qc.rz(np.pi / 6, 0),
        "CNOT":  lambda: qc.cx(0, 1),
        "SWAP":  lambda: qc.swap(0, 1),
        "CCNOT": lambda: qc.ccx(0, 1, 2),
    }
    gate_map[gate_name]()
    # Build a circuit with n_reps repetitions
    big_qc = QuantumCircuit(n_qubits)
    for _ in range(n_reps):
        big_qc.compose(qc, inplace=True)

    def run():
        sv = Statevector.from_label("0" * n_qubits)
        sv = sv.evolve(big_qc)
        return sv

    r = bench(run, n_warmup=2, n_runs=N_RUNS)
    return r.median / n_reps * 1e6


def run_gates_benchmark(max_qubits):
    _section("CATEGORY 1: PRIMITIVE GATE OPERATIONS",
             f"Microseconds per gate (median of {N_RUNS} runs, {1000} reps each)")

    gates = ["H", "X", "RX", "RZ", "CNOT", "SWAP", "CCNOT"]
    qubit_list = [n for n in [4, 8, 12, 16, 20] if n <= max_qubits]

    for nq in qubit_list:
        print(f"\n  --- {nq} qubits ---")
        headers = ["Gate", "Qforge(us)", "PennyLane(us)", "Qiskit(us)", "Speedup"]
        rows = []
        for g in gates:
            if g == "CCNOT" and nq < 3:
                continue
            t_qf = _gate_bench_qforge(g, nq)
            t_pl = _gate_bench_pennylane(g, nq)
            t_qk = _gate_bench_qiskit(g, nq)

            times = [t for t in [t_qf, t_pl, t_qk] if t is not None]
            slowest = max(times) if times else 1
            speedup = f"{slowest / t_qf:.1f}x" if t_qf else "N/A"

            rows.append([
                g,
                f"{t_qf:.2f}" if t_qf else "N/A",
                f"{t_pl:.2f}" if t_pl else "N/A",
                f"{t_qk:.2f}" if t_qk else "N/A",
                speedup,
            ])
            _store("gates", f"{g}_{nq}q", {
                "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk
            })
        _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 2: Circuit Execution Patterns
# ─────────────────────────────────────────────────────────────────────────────

def _build_hcnot_qforge(nq, depth):
    """H+CNOT ladder circuit in qforge."""
    from qforge.circuit import Qubit
    from qforge.gates import H, CNOT
    def run():
        wf = Qubit(nq, backend="cpu")
        for _ in range(depth):
            for q in range(nq):
                H(wf, q)
            for q in range(nq - 1):
                CNOT(wf, q, q + 1)
        _ = wf.amplitude[0]
    return run

def _build_hcnot_pennylane(nq, depth):
    qml = _qml
    if qml is None:
        return None
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev)
    def circuit():
        for _ in range(depth):
            for q in range(nq):
                qml.Hadamard(wires=q)
            for q in range(nq - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()
    return circuit

def _build_hcnot_qiskit(nq, depth):
    if not _has_qiskit:
        return None
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(nq)
    for _ in range(depth):
        qc.h(range(nq))
        for q in range(nq - 1):
            qc.cx(q, q + 1)
    def run():
        return Statevector.from_label("0" * nq).evolve(qc)
    return run

def _build_qft_qforge(nq):
    from qforge.circuit import Qubit
    from qforge.gates import H, CPhase
    def run():
        wf = Qubit(nq, backend="cpu")
        for i in range(nq):
            H(wf, i)
            for j in range(i + 1, nq):
                CPhase(wf, j, i, np.pi / (2 ** (j - i)))
        _ = wf.amplitude[0]
    return run

def _build_qft_pennylane(nq):
    qml = _qml
    if qml is None:
        return None
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev)
    def circuit():
        for i in range(nq):
            qml.Hadamard(wires=i)
            for j in range(i + 1, nq):
                qml.ControlledPhaseShift(np.pi / (2 ** (j - i)), wires=[j, i])
        return qml.state()
    return circuit

def _build_qft_qiskit(nq):
    if not _has_qiskit:
        return None
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(nq)
    for i in range(nq):
        qc.h(i)
        for j in range(i + 1, nq):
            qc.cp(np.pi / (2 ** (j - i)), j, i)
    def run():
        return Statevector.from_label("0" * nq).evolve(qc)
    return run

def _build_random_qforge(nq, n_gates=50):
    from qforge.circuit import Qubit
    from qforge.gates import H, CNOT, RX, RY, RZ
    rng = np.random.RandomState(SEED)
    ops = []
    for _ in range(n_gates):
        g = rng.choice(["H", "CNOT", "RX", "RY", "RZ"])
        if g == "H":
            q = rng.randint(nq)
            ops.append(("H", q, None))
        elif g == "CNOT" and nq >= 2:
            c, t = rng.choice(nq, 2, replace=False)
            ops.append(("CNOT", int(c), int(t)))
        elif g in ("RX", "RY", "RZ"):
            q = rng.randint(nq)
            angle = rng.uniform(0, 2 * np.pi)
            ops.append((g, q, angle))
        else:
            q = rng.randint(nq)
            ops.append(("H", q, None))
    gate_map = {"H": H, "CNOT": CNOT, "RX": RX, "RY": RY, "RZ": RZ}
    def run():
        wf = Qubit(nq, backend="cpu")
        for op_name, a, b in ops:
            if op_name == "CNOT":
                gate_map[op_name](wf, a, b)
            elif b is not None:
                gate_map[op_name](wf, a, b)
            else:
                gate_map[op_name](wf, a)
        _ = wf.amplitude[0]
    return run

def _build_random_pennylane(nq, n_gates=50):
    qml = _qml
    if qml is None:
        return None
    rng = np.random.RandomState(SEED)
    ops = []
    for _ in range(n_gates):
        g = rng.choice(["H", "CNOT", "RX", "RY", "RZ"])
        if g == "H":
            q = rng.randint(nq)
            ops.append(("H", q, None))
        elif g == "CNOT" and nq >= 2:
            c, t = rng.choice(nq, 2, replace=False)
            ops.append(("CNOT", int(c), int(t)))
        elif g in ("RX", "RY", "RZ"):
            q = rng.randint(nq)
            angle = rng.uniform(0, 2 * np.pi)
            ops.append((g, q, angle))
        else:
            q = rng.randint(nq)
            ops.append(("H", q, None))
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev)
    def circuit():
        for op_name, a, b in ops:
            if op_name == "H":
                qml.Hadamard(wires=a)
            elif op_name == "CNOT":
                qml.CNOT(wires=[a, b])
            elif op_name == "RX":
                qml.RX(b, wires=a)
            elif op_name == "RY":
                qml.RY(b, wires=a)
            elif op_name == "RZ":
                qml.RZ(b, wires=a)
        return qml.state()
    return circuit

def _build_random_qiskit(nq, n_gates=50):
    if not _has_qiskit:
        return None
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.RandomState(SEED)
    qc = QuantumCircuit(nq)
    for _ in range(n_gates):
        g = rng.choice(["H", "CNOT", "RX", "RY", "RZ"])
        if g == "H":
            q = rng.randint(nq)
            qc.h(q)
        elif g == "CNOT" and nq >= 2:
            c, t = rng.choice(nq, 2, replace=False)
            qc.cx(int(c), int(t))
        elif g in ("RX", "RY", "RZ"):
            q = rng.randint(nq)
            angle = rng.uniform(0, 2 * np.pi)
            if g == "RX": qc.rx(angle, q)
            elif g == "RY": qc.ry(angle, q)
            else: qc.rz(angle, q)
        else:
            q = rng.randint(nq)
            qc.h(q)
    def run():
        return Statevector.from_label("0" * nq).evolve(qc)
    return run

def _build_hea_qforge(nq, n_layers=3):
    from qforge.circuit import Qubit
    from qforge.algo.ansatz import hardware_efficient_ansatz
    n_params = nq * (n_layers + 1)
    rng = np.random.RandomState(SEED)
    params = rng.uniform(0, 2 * np.pi, n_params)
    def run():
        wf = Qubit(nq, backend="cpu")
        hardware_efficient_ansatz(wf, params, n_layers)
        _ = wf.amplitude[0]
    return run

def _build_hea_pennylane(nq, n_layers=3):
    qml = _qml
    if qml is None:
        return None
    dev = qml.device("default.qubit", wires=nq)
    n_params = nq * (n_layers + 1)
    rng = np.random.RandomState(SEED)
    params = rng.uniform(0, 2 * np.pi, n_params)
    @qml.qnode(dev)
    def circuit():
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(nq):
                qml.RY(params[idx], wires=q)
                idx += 1
            if layer < n_layers:
                for q in range(nq - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.state()
    return circuit

def _build_hea_qiskit(nq, n_layers=3):
    if not _has_qiskit:
        return None
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n_params = nq * (n_layers + 1)
    rng = np.random.RandomState(SEED)
    params = rng.uniform(0, 2 * np.pi, n_params)
    qc = QuantumCircuit(nq)
    idx = 0
    for layer in range(n_layers + 1):
        for q in range(nq):
            qc.ry(params[idx], q)
            idx += 1
        if layer < n_layers:
            for q in range(nq - 1):
                qc.cx(q, q + 1)
    def run():
        return Statevector.from_label("0" * nq).evolve(qc)
    return run


def run_circuits_benchmark(max_qubits):
    _section("CATEGORY 2: CIRCUIT EXECUTION PATTERNS",
             f"Total execution time in ms (median of {N_RUNS} runs)")

    qubit_list = [n for n in [4, 8, 12, 16, 20] if n <= max_qubits]

    patterns = [
        ("H+CNOT (d=10)", lambda nq: _build_hcnot_qforge(nq, 10),
         lambda nq: _build_hcnot_pennylane(nq, 10),
         lambda nq: _build_hcnot_qiskit(nq, 10)),
        ("QFT-like", _build_qft_qforge, _build_qft_pennylane, _build_qft_qiskit),
        ("Random (50g)", _build_random_qforge, _build_random_pennylane, _build_random_qiskit),
        ("HEA (3L)", _build_hea_qforge, _build_hea_pennylane, _build_hea_qiskit),
    ]

    for pat_name, build_qf, build_pl, build_qk in patterns:
        print(f"\n  --- {pat_name} ---")
        headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "QF speedup"]
        rows = []
        for nq in qubit_list:
            t_qf = t_pl = t_qk = None
            try:
                fn_qf = build_qf(nq)
                if fn_qf:
                    r = bench(fn_qf)
                    t_qf = r.median * 1000
            except Exception:
                pass
            try:
                fn_pl = build_pl(nq)
                if fn_pl:
                    r = bench(fn_pl)
                    t_pl = r.median * 1000
            except Exception:
                pass
            try:
                fn_qk = build_qk(nq)
                if fn_qk:
                    r = bench(fn_qk)
                    t_qk = r.median * 1000
            except Exception:
                pass

            others = [t for t in [t_pl, t_qk] if t is not None]
            if t_qf and others:
                speedup = f"{max(others) / t_qf:.1f}x"
            else:
                speedup = "N/A"

            rows.append([
                nq,
                f"{t_qf:.2f}" if t_qf else "N/A",
                f"{t_pl:.2f}" if t_pl else "N/A",
                f"{t_qk:.2f}" if t_qk else "N/A",
                speedup,
            ])
            _store("circuits", f"{pat_name}_{nq}q", {
                "qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk
            })
        _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 3: VQE Algorithm
# ─────────────────────────────────────────────────────────────────────────────

# H2 Hamiltonian coefficients
VQE_COEFFS_H2 = [-1.0523, 0.3979, -0.3979, -0.0112, 0.1809]
VQE_TERMS_H2 = [[], [('Z', 0)], [('Z', 1)], [('Z', 0), ('Z', 1)], [('X', 0), ('X', 1)]]

# LiH-like 6-qubit Hamiltonian (15 Pauli terms)
def _lih_hamiltonian():
    rng = np.random.RandomState(SEED)
    coeffs = []
    terms = []
    # Identity
    coeffs.append(-7.498)
    terms.append([])
    # Single Z terms
    for q in range(6):
        coeffs.append(rng.uniform(-0.5, 0.5))
        terms.append([('Z', q)])
    # ZZ interaction terms
    for i in range(5):
        coeffs.append(rng.uniform(-0.3, 0.3))
        terms.append([('Z', i), ('Z', i + 1)])
    # XX interaction terms
    for i in range(0, 4, 2):
        coeffs.append(rng.uniform(0.05, 0.2))
        terms.append([('X', i), ('X', i + 1)])
    return coeffs, terms

LIH_COEFFS, LIH_TERMS = _lih_hamiltonian()


def _vqe_qforge(coeffs, terms, n_qubits, n_layers=1, backend="cpu"):
    from qforge.algo import Hamiltonian, VQE, GradientDescent
    H = Hamiltonian(coeffs=coeffs, terms=terms)
    n_p = VQE.n_params_hardware_efficient(n_qubits, n_layers=n_layers)
    def run():
        vqe = VQE(n_qubits=n_qubits, hamiltonian=H, n_layers=n_layers, backend=backend)
        _, history = vqe.optimize(np.zeros(n_p), optimizer=GradientDescent(lr=LR), steps=STEPS)
        return history[-1]
    return bench(run, n_warmup=1, n_runs=3)


def _vqe_pennylane(coeffs, terms, n_qubits, n_layers=1):
    qml = _qml
    if qml is None:
        return None
    pnp = qml.numpy
    dev = qml.device("default.qubit", wires=n_qubits)

    # Build PennyLane Hamiltonian
    obs_list = []
    for t in terms:
        if not t:
            obs_list.append(qml.Identity(0))
        else:
            ops = []
            for pauli, q in t:
                if pauli == 'X': ops.append(qml.PauliX(q))
                elif pauli == 'Y': ops.append(qml.PauliY(q))
                elif pauli == 'Z': ops.append(qml.PauliZ(q))
            ob = ops[0]
            for o in ops[1:]:
                ob = ob @ o
            obs_list.append(ob)
    try:
        H_pl = qml.Hamiltonian(pnp.array(coeffs, requires_grad=False), obs_list)
    except Exception:
        H_pl = qml.ops.LinearCombination(coeffs, obs_list)

    n_p = n_qubits * (n_layers + 1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            if layer < n_layers:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(H_pl)

    opt = qml.GradientDescentOptimizer(stepsize=LR)

    def run():
        params = pnp.zeros(n_p, requires_grad=True)
        cost = None
        for _ in range(STEPS):
            params, cost = opt.step_and_cost(circuit, params)
        return float(cost)

    return bench(run, n_warmup=1, n_runs=3)


def _vqe_qiskit(coeffs, terms, n_qubits, n_layers=1):
    if not _has_qiskit:
        return None

    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    # Build Qiskit Pauli strings (LSB ordering: qubit 0 is rightmost)
    pauli_list = []
    for c, t in zip(coeffs, terms):
        label = ["I"] * n_qubits
        for pauli, q in t:
            label[n_qubits - 1 - q] = pauli
        pauli_list.append(("".join(label), c))
    H_qk = SparsePauliOp.from_list(pauli_list)

    ansatz = TwoLocal(
        n_qubits, rotation_blocks=["ry"], entanglement_blocks="cx",
        reps=n_layers, entanglement="linear"
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

    return bench(run, n_warmup=1, n_runs=3)


def run_vqe_benchmark(max_qubits):
    from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
    _section("CATEGORY 3: VQE ALGORITHM",
             f"{STEPS} steps, lr={LR}, parameter-shift gradient")

    problems = [
        ("H2 (2q)", VQE_COEFFS_H2, VQE_TERMS_H2, 2, 1),
    ]
    if max_qubits >= 6:
        problems.append(("LiH-like (6q)", LIH_COEFFS, LIH_TERMS, 6, 1))

    for prob_name, coeffs, terms, nq, nl in problems:
        print(f"\n  --- {prob_name}, {nq} qubits, {nl} layer ---")
        headers = ["Framework", "ms/step", "total(s)", "energy"]
        rows = []

        # Qforge backends
        for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA and _BENCH_CUDA), ("metal", _HAS_METAL)]:
            if not available:
                continue
            try:
                r = _vqe_qforge(coeffs, terms, nq, nl, backend)
                ms_step = r.median / STEPS * 1000
                rows.append([f"Qforge ({backend})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("vqe", f"{prob_name}_{backend}", {"time": r.median, "energy": r.result})
            except Exception as e:
                rows.append([f"Qforge ({backend})", "ERR", "ERR", str(e)[:30]])

        # PennyLane
        try:
            r = _vqe_pennylane(coeffs, terms, nq, nl)
            if r:
                ms_step = r.median / STEPS * 1000
                ver = _qml.__version__ if _qml else "?"
                rows.append([f"PennyLane (v{ver})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("vqe", f"{prob_name}_pennylane", {"time": r.median, "energy": r.result})
            else:
                rows.append(["PennyLane", "N/A", "N/A", "not installed"])
        except Exception as e:
            rows.append(["PennyLane", "ERR", "ERR", str(e)[:30]])

        # Qiskit
        try:
            r = _vqe_qiskit(coeffs, terms, nq, nl)
            if r:
                import qiskit
                ver = qiskit.__version__
                ms_step = r.median / STEPS * 1000
                rows.append([f"Qiskit (v{ver})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("vqe", f"{prob_name}_qiskit", {"time": r.median, "energy": r.result})
            else:
                rows.append(["Qiskit", "N/A", "N/A", "not installed"])
        except Exception as e:
            rows.append(["Qiskit", "ERR", "ERR", str(e)[:30]])

        _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 4: QAOA Algorithm
# ─────────────────────────────────────────────────────────────────────────────

QAOA_RING_4 = [(0, 1), (1, 2), (2, 3), (3, 0)]

def _qaoa_8node_edges():
    """Generate a fixed 8-node 3-regular graph."""
    # Deterministic 3-regular graph on 8 nodes
    return [(0, 1), (0, 3), (0, 5), (1, 2), (1, 7), (2, 3),
            (2, 6), (3, 4), (4, 5), (4, 7), (5, 6), (6, 7)]

QAOA_GRAPH_8 = _qaoa_8node_edges()


def _qaoa_qforge(edges, n_qubits, p_layers=1, backend="cpu"):
    from qforge.algo import QAOA, GradientDescent
    def run():
        qaoa = QAOA(n_qubits=n_qubits, edges=edges, p_layers=p_layers, backend=backend)
        params0 = np.full(qaoa.n_params, 0.5)
        _, history = qaoa.optimize(params0, optimizer=GradientDescent(lr=LR), steps=STEPS)
        return -history[-1]
    return bench(run, n_warmup=1, n_runs=3)


def _qaoa_pennylane(edges, n_qubits, p_layers=1):
    qml = _qml
    if qml is None:
        return None
    pnp = qml.numpy
    dev = qml.device("default.qubit", wires=n_qubits)

    # Cost Hamiltonian: H_C = 0.5 * sum_{(i,j)} (I - ZiZj)
    obs_list = []
    coeffs_list = []
    for (i, j) in edges:
        obs_list.append(qml.Identity(0))
        coeffs_list.append(0.5)
        obs_list.append(qml.PauliZ(i) @ qml.PauliZ(j))
        coeffs_list.append(-0.5)
    try:
        H_cost = qml.Hamiltonian(pnp.array(coeffs_list, requires_grad=False), obs_list)
    except Exception:
        H_cost = qml.ops.LinearCombination(coeffs_list, obs_list)

    @qml.qnode(dev, diff_method="parameter-shift")
    def qaoa_circuit(params):
        gammas = params[:p_layers]
        betas = params[p_layers:]
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for l in range(p_layers):
            for (i, j) in edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(-gammas[l], wires=j)
                qml.CNOT(wires=[i, j])
            for w in range(n_qubits):
                qml.RX(2.0 * betas[l], wires=w)
        return qml.expval(H_cost)

    def neg_qaoa(params):
        return -qaoa_circuit(params)

    opt = qml.GradientDescentOptimizer(stepsize=LR)

    def run():
        params = pnp.array([0.5] * (2 * p_layers), requires_grad=True)
        neg_cost = None
        for _ in range(STEPS):
            params, neg_cost = opt.step_and_cost(neg_qaoa, params)
        return float(-neg_cost)

    return bench(run, n_warmup=1, n_runs=3)


def _qaoa_qiskit(edges, n_qubits, p_layers=1):
    if not _has_qiskit:
        return None
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    # Build cost Hamiltonian
    pauli_list = []
    for (i, j) in edges:
        label_i = "I" * n_qubits
        pauli_list.append((label_i, 0.5))
        label_zz = ["I"] * n_qubits
        label_zz[n_qubits - 1 - i] = "Z"
        label_zz[n_qubits - 1 - j] = "Z"
        pauli_list.append(("".join(label_zz), -0.5))
    H_cost_qk = SparsePauliOp.from_list(pauli_list)

    gamma = ParameterVector("g", p_layers)
    beta = ParameterVector("b", p_layers)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for l in range(p_layers):
        for (i, j) in edges:
            qc.cx(i, j)
            qc.rz(-gamma[l], j)
            qc.cx(i, j)
        for w in range(n_qubits):
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
        params = np.array([0.5] * (2 * p_layers))
        cut = None
        for _ in range(STEPS):
            cut = evaluate(params)
            params += LR * grad_param_shift(params)
        return cut

    return bench(run, n_warmup=1, n_runs=3)


def run_qaoa_benchmark(max_qubits):
    from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
    _section("CATEGORY 4: QAOA ALGORITHM (MAX-CUT)",
             f"p=1, {STEPS} steps, lr={LR}, parameter-shift gradient")

    problems = [("4-node ring", QAOA_RING_4, 4)]
    if max_qubits >= 8:
        problems.append(("8-node 3-reg", QAOA_GRAPH_8, 8))

    for prob_name, edges, nq in problems:
        print(f"\n  --- {prob_name}, {nq} qubits ---")
        headers = ["Framework", "ms/step", "total(s)", "cut value"]
        rows = []

        for backend, available in [("cpu", _HAS_CPP), ("cuda", _HAS_CUDA and _BENCH_CUDA), ("metal", _HAS_METAL)]:
            if not available:
                continue
            try:
                r = _qaoa_qforge(edges, nq, 1, backend)
                ms_step = r.median / STEPS * 1000
                rows.append([f"Qforge ({backend})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("qaoa", f"{prob_name}_{backend}", {"time": r.median, "cut": r.result})
            except Exception as e:
                rows.append([f"Qforge ({backend})", "ERR", "ERR", str(e)[:30]])

        try:
            r = _qaoa_pennylane(edges, nq, 1)
            if r:
                ver = _qml.__version__ if _qml else "?"
                ms_step = r.median / STEPS * 1000
                rows.append([f"PennyLane (v{ver})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("qaoa", f"{prob_name}_pennylane", {"time": r.median, "cut": r.result})
            else:
                rows.append(["PennyLane", "N/A", "N/A", "not installed"])
        except Exception as e:
            rows.append(["PennyLane", "ERR", "ERR", str(e)[:30]])

        try:
            r = _qaoa_qiskit(edges, nq, 1)
            if r:
                import qiskit
                ver = qiskit.__version__
                ms_step = r.median / STEPS * 1000
                rows.append([f"Qiskit (v{ver})", f"{ms_step:.2f}", f"{r.median:.3f}", f"{r.result:.4f}"])
                _store("qaoa", f"{prob_name}_qiskit", {"time": r.median, "cut": r.result})
            else:
                rows.append(["Qiskit", "N/A", "N/A", "not installed"])
        except Exception as e:
            rows.append(["Qiskit", "ERR", "ERR", str(e)[:30]])

        _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 5: Gradient Computation
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_bench_qforge(n_qubits, n_layers, backend="cpu"):
    from qforge.algo import Hamiltonian, VQE

    rng = np.random.default_rng(SEED)
    coeffs = rng.uniform(-1, 1, n_qubits).tolist()
    terms = [[('Z', q)] for q in range(n_qubits)]
    H = Hamiltonian(coeffs, terms)

    n_p = VQE.n_params_hardware_efficient(n_qubits, n_layers)
    vqe = VQE(n_qubits=n_qubits, hamiltonian=H, n_layers=n_layers, backend=backend)
    params = np.zeros(n_p)

    # Warmup
    vqe._evaluate(params)
    vqe.gradient(params)

    def run():
        return vqe.gradient(params)

    return bench(run, n_warmup=1, n_runs=N_RUNS)


def _gradient_bench_pennylane(n_qubits, n_layers):
    qml = _qml
    if qml is None:
        return None
    pnp = qml.numpy
    dev = qml.device("default.qubit", wires=n_qubits)

    rng = np.random.default_rng(SEED)
    coeffs = rng.uniform(-1, 1, n_qubits).tolist()

    obs_list = [qml.PauliZ(q) for q in range(n_qubits)]
    try:
        H_pl = qml.Hamiltonian(pnp.array(coeffs, requires_grad=False), obs_list)
    except Exception:
        H_pl = qml.ops.LinearCombination(coeffs, obs_list)

    n_p = n_qubits * (n_layers + 1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            if layer < n_layers:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(H_pl)

    grad_fn = qml.grad(circuit)

    def run():
        params = pnp.zeros(n_p, requires_grad=True)
        return grad_fn(params)

    return bench(run, n_warmup=2, n_runs=N_RUNS)


def _gradient_bench_qiskit(n_qubits, n_layers):
    if not _has_qiskit:
        return None

    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator

    rng = np.random.default_rng(SEED)
    coeffs = rng.uniform(-1, 1, n_qubits).tolist()

    pauli_list = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        pauli_list.append(("".join(label), coeffs[q]))
    H_qk = SparsePauliOp.from_list(pauli_list)

    ansatz = TwoLocal(n_qubits, rotation_blocks=["ry"], entanglement_blocks="cx",
                      reps=n_layers, entanglement="linear")
    estimator = StatevectorEstimator()

    def evaluate(params):
        job = estimator.run([(ansatz, H_qk, params)])
        return float(job.result()[0].data.evs)

    def run():
        params = np.zeros(ansatz.num_parameters)
        g = np.zeros_like(params)
        denom = 2.0 * np.sin(SHIFT)
        for i in range(len(params)):
            pp = params.copy(); pp[i] += SHIFT
            pm = params.copy(); pm[i] -= SHIFT
            g[i] = (evaluate(pp) - evaluate(pm)) / denom
        return g

    return bench(run, n_warmup=2, n_runs=N_RUNS)


def run_gradient_benchmark(max_qubits):
    from qforge import _HAS_CPP
    _section("CATEGORY 5: GRADIENT COMPUTATION",
             "Parameter-shift gradient time (ms)")

    configs = [(4, 1, 8), (8, 1, 16), (4, 2, 12)]
    configs = [(nq, nl, np_) for nq, nl, np_ in configs if nq <= max_qubits]

    headers = ["Config", "n_params", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "Speedup"]
    rows = []

    for nq, nl, n_p in configs:
        label = f"{nq}q/{nl}L"
        t_qf = t_pl = t_qk = None

        if _HAS_CPP:
            try:
                r = _gradient_bench_qforge(nq, nl)
                t_qf = r.median * 1000
            except Exception:
                pass
        try:
            r = _gradient_bench_pennylane(nq, nl)
            if r:
                t_pl = r.median * 1000
        except Exception:
            pass
        try:
            r = _gradient_bench_qiskit(nq, nl)
            if r:
                t_qk = r.median * 1000
        except Exception:
            pass

        others = [t for t in [t_pl, t_qk] if t is not None]
        speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"

        rows.append([
            label, n_p,
            f"{t_qf:.1f}" if t_qf else "N/A",
            f"{t_pl:.1f}" if t_pl else "N/A",
            f"{t_qk:.1f}" if t_qk else "N/A",
            speedup,
        ])
        _store("gradient", label, {"qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})

    _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 6: Measurement Operations
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_state_qforge(nq):
    """Prepare a non-trivial state for measurement benchmarks."""
    from qforge.circuit import Qubit
    from qforge.gates import H, CNOT
    wf = Qubit(nq, backend="cpu")
    for q in range(nq):
        H(wf, q)
    for q in range(nq - 1):
        CNOT(wf, q, q + 1)
    return wf


def run_measurement_benchmark(max_qubits):
    from qforge import _HAS_CPP
    from qforge.measurement import pauli_expectation, measure_all
    from qforge.algo.hamiltonian import Hamiltonian

    _section("CATEGORY 6: MEASUREMENT OPERATIONS",
             "Time per operation (microseconds)")

    qubit_list = [n for n in [4, 8, 12, 16] if n <= max_qubits]

    # --- Sub-benchmark 1: Single Pauli-Z expectation ---
    print("\n  --- Single Pauli-Z expectation ---")
    headers = ["Qubits", "Qforge(us)", "PennyLane(us)", "Qiskit(us)", "Speedup"]
    rows = []

    for nq in qubit_list:
        t_qf = t_pl = t_qk = None

        # Qforge
        if _HAS_CPP:
            wf = _prepare_state_qforge(nq)
            def run_qf(wf=wf):
                return pauli_expectation(wf, 0, 'Z')
            r = bench(run_qf, n_warmup=3, n_runs=N_RUNS)
            t_qf = r.median * 1e6

        # PennyLane
        if _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def pl_circuit():
                for q in range(nq):
                    qml.Hadamard(wires=q)
                for q in range(nq - 1):
                    qml.CNOT(wires=[q, q + 1])
                return qml.expval(qml.PauliZ(0))
            r = bench(pl_circuit, n_warmup=3, n_runs=N_RUNS)
            t_pl = r.median * 1e6

        # Qiskit
        if _has_qiskit:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp, Statevector
            qc = QuantumCircuit(nq)
            qc.h(range(nq))
            for q in range(nq - 1):
                qc.cx(q, q + 1)
            z_label = ["I"] * nq
            z_label[nq - 1] = "Z"
            z_op = SparsePauliOp.from_list([("".join(z_label), 1.0)])
            def run_qk(qc=qc, z_op=z_op):
                sv = Statevector.from_label("0" * nq).evolve(qc)
                return sv.expectation_value(z_op).real
            r = bench(run_qk, n_warmup=3, n_runs=N_RUNS)
            t_qk = r.median * 1e6

        others = [t for t in [t_pl, t_qk] if t is not None]
        speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"
        rows.append([
            nq,
            f"{t_qf:.1f}" if t_qf else "N/A",
            f"{t_pl:.1f}" if t_pl else "N/A",
            f"{t_qk:.1f}" if t_qk else "N/A",
            speedup,
        ])
        _store("measurement", f"pauliZ_{nq}q", {"qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})
    _table(headers, rows)

    # --- Sub-benchmark 2: Sampling 10,000 shots ---
    print(f"\n  --- Sampling (10,000 shots) ---")
    headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)", "Speedup"]
    rows = []
    n_shots = 10000

    for nq in qubit_list:
        t_qf = t_pl = t_qk = None

        if _HAS_CPP:
            wf = _prepare_state_qforge(nq)
            def run_qf_s(wf=wf):
                return measure_all(wf, n_shots)
            r = bench(run_qf_s, n_warmup=2, n_runs=N_RUNS)
            t_qf = r.median * 1000

        if _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq, shots=n_shots)
            @qml.qnode(dev)
            def pl_sample():
                for q in range(nq):
                    qml.Hadamard(wires=q)
                for q in range(nq - 1):
                    qml.CNOT(wires=[q, q + 1])
                return qml.sample()
            r = bench(pl_sample, n_warmup=2, n_runs=N_RUNS)
            t_pl = r.median * 1000

        if _has_qiskit:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit(nq)
            qc.h(range(nq))
            for q in range(nq - 1):
                qc.cx(q, q + 1)
            qc.measure_all()
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            def run_qk_s(qc=qc):
                job = sampler.run([qc], shots=n_shots)
                return job.result()
            r = bench(run_qk_s, n_warmup=2, n_runs=N_RUNS)
            t_qk = r.median * 1000

        others = [t for t in [t_pl, t_qk] if t is not None]
        speedup = f"{max(others) / t_qf:.1f}x" if t_qf and others else "N/A"
        rows.append([
            nq,
            f"{t_qf:.2f}" if t_qf else "N/A",
            f"{t_pl:.2f}" if t_pl else "N/A",
            f"{t_qk:.2f}" if t_qk else "N/A",
            speedup,
        ])
        _store("measurement", f"sampling_{nq}q", {"qforge": t_qf, "pennylane": t_pl, "qiskit": t_qk})
    _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 7: Scalability
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling_benchmark(max_qubits):
    from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
    _section("CATEGORY 7: SCALABILITY",
             "HEA 1-layer forward pass time (ms) vs qubit count")

    qubit_list = [n for n in range(2, min(max_qubits, 24) + 1, 2)]
    timeout = 60.0

    bench_cuda = _HAS_CUDA and _BENCH_CUDA

    headers = ["Qubits", "Qforge(ms)", "PennyLane(ms)", "Qiskit(ms)"]
    if bench_cuda:
        headers.insert(2, "QF-CUDA(ms)")
    rows = []

    skip_pl = False
    skip_qk = False

    for nq in qubit_list:
        n_p = nq * 2  # 1 layer
        rng = np.random.RandomState(SEED)
        params = rng.uniform(0, 2 * np.pi, n_p)

        row = [nq]
        t_qf = t_cuda = t_pl = t_qk = None

        # Qforge CPU
        if _HAS_CPP:
            from qforge.circuit import Qubit
            from qforge.algo.ansatz import hardware_efficient_ansatz
            def run_qf(nq=nq, params=params):
                wf = Qubit(nq, backend="cpu")
                hardware_efficient_ansatz(wf, params, 1)
                _ = wf.amplitude[0]
            try:
                r = bench(run_qf, n_warmup=2, n_runs=N_RUNS)
                t_qf = r.median * 1000
            except Exception:
                pass
        row.append(f"{t_qf:.3f}" if t_qf else "N/A")

        # Qforge CUDA
        if bench_cuda:
            def run_cuda(nq=nq, params=params):
                wf = Qubit(nq, backend="cuda")
                hardware_efficient_ansatz(wf, params, 1)
                _ = wf.amplitude[0]
            try:
                r = bench(run_cuda, n_warmup=2, n_runs=N_RUNS)
                t_cuda = r.median * 1000
            except Exception:
                pass
            row.append(f"{t_cuda:.3f}" if t_cuda is not None else "N/A")

        # PennyLane
        if not skip_pl and _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def pl_hea(params=params):
                idx = 0
                for layer in range(2):
                    for q in range(nq):
                        qml.RY(params[idx], wires=q)
                        idx += 1
                    if layer < 1:
                        for q in range(nq - 1):
                            qml.CNOT(wires=[q, q + 1])
                return qml.state()
            try:
                t0 = time.perf_counter()
                r = bench(pl_hea, n_warmup=1, n_runs=3)
                if time.perf_counter() - t0 > timeout:
                    skip_pl = True
                t_pl = r.median * 1000
            except Exception:
                pass
        row.append(f"{t_pl:.3f}" if t_pl else "N/A" if not skip_pl else "TIMEOUT")

        # Qiskit
        if not skip_qk and _has_qiskit:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit(nq)
            idx = 0
            for layer in range(2):
                for q in range(nq):
                    qc.ry(params[idx], q)
                    idx += 1
                if layer < 1:
                    for q in range(nq - 1):
                        qc.cx(q, q + 1)
            def run_qk(qc=qc, nq=nq):
                return Statevector.from_label("0" * nq).evolve(qc)
            try:
                t0 = time.perf_counter()
                r = bench(run_qk, n_warmup=1, n_runs=3)
                if time.perf_counter() - t0 > timeout:
                    skip_qk = True
                t_qk = r.median * 1000
            except Exception:
                pass
        row.append(f"{t_qk:.3f}" if t_qk else "N/A" if not skip_qk else "TIMEOUT")

        rows.append(row)
        _store("scaling", f"{nq}q", {
            "qforge": t_qf, "qforge_cuda": t_cuda,
            "pennylane": t_pl, "qiskit": t_qk,
        })

    _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 8: Accuracy & Correctness
# ─────────────────────────────────────────────────────────────────────────────

def _get_statevector_qforge(circuit_fn, nq):
    from qforge.circuit import Qubit
    wf = Qubit(nq, backend="cpu")
    circuit_fn(wf)
    return wf.amplitude.copy()

def _get_statevector_pennylane(circuit_fn_pl, nq):
    if _qml is None:
        return None
    return np.array(circuit_fn_pl())

def _get_statevector_qiskit(qc, nq):
    """Get statevector from qiskit, reversing qubit order to match qforge/pennylane convention."""
    if not _has_qiskit:
        return None
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_label("0" * nq).evolve(qc)
    # Qiskit uses LSB ordering (qubit 0 = rightmost bit), while qforge/pennylane
    # use MSB ordering (qubit 0 = leftmost bit). Reverse the qubit order.
    data = np.array(sv.data)
    # Reorder amplitudes: swap bit ordering by reversing the binary indices
    reordered = np.zeros_like(data)
    for i in range(len(data)):
        # Reverse bit order of index i
        rev = int(f"{i:0{nq}b}"[::-1], 2)
        reordered[rev] = data[i]
    return reordered


def run_accuracy_benchmark(max_qubits):
    from qforge import _HAS_CPP
    from qforge.gates import H, CNOT
    _section("CATEGORY 8: ACCURACY & CORRECTNESS",
             "Max |amplitude difference| between frameworks")

    headers = ["Test", "QF vs PL", "QF vs QK", "PL vs QK"]
    rows = []

    # --- Bell state ---
    if _HAS_CPP:
        sv_qf = _get_statevector_qforge(lambda wf: (H(wf, 0), CNOT(wf, 0, 1)), 2)
    else:
        sv_qf = None

    if _qml is not None:
        qml = _qml
        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def bell_pl():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()
        sv_pl = np.array(bell_pl())
    else:
        sv_pl = None

    if _has_qiskit:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1)
        sv_qk = _get_statevector_qiskit(qc, 2)
    else:
        sv_qk = None

    def _max_diff(a, b):
        if a is None or b is None:
            return "N/A"
        return f"{np.max(np.abs(a - b)):.2e}"

    rows.append(["Bell (2q)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])
    _store("accuracy", "bell", {
        "qf_pl": float(np.max(np.abs(sv_qf - sv_pl))) if sv_qf is not None and sv_pl is not None else None,
        "qf_qk": float(np.max(np.abs(sv_qf - sv_qk))) if sv_qf is not None and sv_qk is not None else None,
    })

    # --- GHZ states ---
    for nq in [4, 8, 12]:
        if nq > max_qubits:
            break

        if _HAS_CPP:
            def ghz_qf(wf, nq=nq):
                H(wf, 0)
                for q in range(nq - 1):
                    CNOT(wf, q, q + 1)
            sv_qf = _get_statevector_qforge(ghz_qf, nq)
        else:
            sv_qf = None

        if _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def ghz_pl(nq=nq):
                qml.Hadamard(wires=0)
                for q in range(nq - 1):
                    qml.CNOT(wires=[q, q + 1])
                return qml.state()
            sv_pl = np.array(ghz_pl())
        else:
            sv_pl = None

        if _has_qiskit:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(nq)
            qc.h(0)
            for q in range(nq - 1):
                qc.cx(q, q + 1)
            sv_qk = _get_statevector_qiskit(qc, nq)
        else:
            sv_qk = None

        rows.append([f"GHZ ({nq}q)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])

    # --- Random circuit cross-validation (8 qubits) ---
    nq_rand = min(8, max_qubits)
    if nq_rand >= 4:
        rng = np.random.RandomState(SEED)
        ops_spec = []
        for _ in range(50):
            g = rng.choice(["H", "CNOT", "RX", "RY", "RZ"])
            if g == "H":
                ops_spec.append(("H", rng.randint(nq_rand), None))
            elif g == "CNOT" and nq_rand >= 2:
                c, t = rng.choice(nq_rand, 2, replace=False)
                ops_spec.append(("CNOT", int(c), int(t)))
            elif g in ("RX", "RY", "RZ"):
                ops_spec.append((g, rng.randint(nq_rand), rng.uniform(0, 2 * np.pi)))
            else:
                ops_spec.append(("H", rng.randint(nq_rand), None))

        # Qforge
        if _HAS_CPP:
            from qforge.gates import RX, RY, RZ
            gmap = {"H": H, "CNOT": CNOT, "RX": RX, "RY": RY, "RZ": RZ}
            def rand_qf(wf):
                for op, a, b in ops_spec:
                    if op == "CNOT":
                        gmap[op](wf, a, b)
                    elif b is not None:
                        gmap[op](wf, a, b)
                    else:
                        gmap[op](wf, a)
            sv_qf = _get_statevector_qforge(rand_qf, nq_rand)
        else:
            sv_qf = None

        if _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq_rand)
            @qml.qnode(dev)
            def rand_pl():
                for op, a, b in ops_spec:
                    if op == "H": qml.Hadamard(wires=a)
                    elif op == "CNOT": qml.CNOT(wires=[a, b])
                    elif op == "RX": qml.RX(b, wires=a)
                    elif op == "RY": qml.RY(b, wires=a)
                    elif op == "RZ": qml.RZ(b, wires=a)
                return qml.state()
            sv_pl = np.array(rand_pl())
        else:
            sv_pl = None

        if _has_qiskit:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(nq_rand)
            for op, a, b in ops_spec:
                if op == "H": qc.h(a)
                elif op == "CNOT": qc.cx(a, b)
                elif op == "RX": qc.rx(b, a)
                elif op == "RY": qc.ry(b, a)
                elif op == "RZ": qc.rz(b, a)
            sv_qk = _get_statevector_qiskit(qc, nq_rand)
        else:
            sv_qk = None

        rows.append([f"Random ({nq_rand}q, 50g)", _max_diff(sv_qf, sv_pl), _max_diff(sv_qf, sv_qk), _max_diff(sv_pl, sv_qk)])

    _table(headers, rows)


# ─────────────────────────────────────────────────────────────────────────────
# Category 9: Memory Usage
# ─────────────────────────────────────────────────────────────────────────────

def run_memory_benchmark(max_qubits):
    from qforge import _HAS_CPP
    _section("CATEGORY 9: MEMORY USAGE",
             "Peak Python-side memory (MB) for HEA 1-layer")

    qubit_list = [n for n in [4, 8, 12, 16, 20, 24] if n <= max_qubits]

    headers = ["Qubits", "Theoretical(MB)", "Qforge(MB)", "PennyLane(MB)", "Qiskit(MB)"]
    rows = []

    for nq in qubit_list:
        theoretical = 2**nq * 16 / (1024 * 1024)  # complex128
        n_p = nq * 2
        rng = np.random.RandomState(SEED)
        params = rng.uniform(0, 2 * np.pi, n_p)

        m_qf = m_pl = m_qk = None

        if _HAS_CPP:
            from qforge.circuit import Qubit
            from qforge.algo.ansatz import hardware_efficient_ansatz
            def fn_qf(nq=nq, params=params):
                wf = Qubit(nq, backend="cpu")
                hardware_efficient_ansatz(wf, params, 1)
                _ = wf.amplitude[0]
            try:
                m_qf = measure_memory(fn_qf)
            except Exception:
                pass

        if _qml is not None:
            qml = _qml
            dev = qml.device("default.qubit", wires=nq)
            @qml.qnode(dev)
            def pl_fn(params=params):
                idx = 0
                for layer in range(2):
                    for q in range(nq):
                        qml.RY(params[idx], wires=q)
                        idx += 1
                    if layer < 1:
                        for q in range(nq - 1):
                            qml.CNOT(wires=[q, q + 1])
                return qml.state()
            try:
                m_pl = measure_memory(pl_fn)
            except Exception:
                pass

        if _has_qiskit:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit(nq)
            idx = 0
            for layer in range(2):
                for q in range(nq):
                    qc.ry(params[idx], q)
                    idx += 1
                if layer < 1:
                    for q in range(nq - 1):
                        qc.cx(q, q + 1)
            def fn_qk(qc=qc, nq=nq):
                return Statevector.from_label("0" * nq).evolve(qc)
            try:
                m_qk = measure_memory(fn_qk)
            except Exception:
                pass

        rows.append([
            nq,
            f"{theoretical:.3f}",
            f"{m_qf:.3f}" if m_qf is not None else "N/A",
            f"{m_pl:.3f}" if m_pl is not None else "N/A",
            f"{m_qk:.3f}" if m_qk is not None else "N/A",
        ])
        _store("memory", f"{nq}q", {
            "theoretical": theoretical, "qforge": m_qf,
            "pennylane": m_pl, "qiskit": m_qk,
        })

    _table(headers, rows)
    print("\n  Note: Qforge C++ allocations are not tracked by tracemalloc.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(categories):
    _section("OVERALL SUMMARY", "")
    print("  Categories benchmarked:", ", ".join(categories))

    if _results_json:
        # Find fastest framework per category
        summary = {}
        for cat, entries in _results_json.items():
            qf_wins = 0
            total = 0
            for key, data in entries.items():
                if isinstance(data, dict):
                    qf_t = data.get("qforge") or data.get("time")
                    pl_t = data.get("pennylane")
                    qk_t = data.get("qiskit")
                    times = {k: v for k, v in [("qforge", qf_t), ("pennylane", pl_t), ("qiskit", qk_t)] if v is not None}
                    if len(times) > 1:
                        total += 1
                        if times and min(times, key=times.get) == "qforge":
                            qf_wins += 1
            if total > 0:
                summary[cat] = f"Qforge fastest in {qf_wins}/{total} tests"

        if summary:
            for cat, result in summary.items():
                print(f"  {cat:20s}  {result}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark: Qforge vs PennyLane vs Qiskit"
    )
    parser.add_argument(
        "--categories", nargs="*", default=["all"],
        choices=["all"] + ALL_CATEGORIES,
        help="Which benchmark categories to run",
    )
    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json", type=str, default=None, help="Export results to JSON")
    args = parser.parse_args()

    global N_RUNS  # noqa
    N_RUNS = args.runs

    categories = ALL_CATEGORIES if "all" in args.categories else args.categories
    max_q = args.max_qubits

    # Print header
    print("=" * _WIDTH)
    print("  COMPREHENSIVE BENCHMARK: Qforge vs PennyLane vs Qiskit")
    print("=" * _WIDTH)

    from qforge import _HAS_CPP, _HAS_CUDA, _HAS_METAL
    print(f"  Qforge backends — CPU:{_HAS_CPP}  CUDA:{_HAS_CUDA}  Metal:{_HAS_METAL}")
    print(f"  Max qubits: {max_q}  |  Runs: {N_RUNS}  |  Warmup: {N_WARMUP}")
    print(f"  Categories: {', '.join(categories)}")

    print("\n  Loading external frameworks ...")
    qml = _load_pennylane()
    has_qk = _load_qiskit()
    if _qml:
        _check_lightning()
        print(f"    PennyLane v{_qml.__version__}" + (" + lightning.qubit" if _has_lightning else ""))
    if has_qk:
        import qiskit
        print(f"    Qiskit v{qiskit.__version__}")

    # Run selected categories
    dispatch = {
        "gates":       lambda: run_gates_benchmark(max_q),
        "circuits":    lambda: run_circuits_benchmark(max_q),
        "vqe":         lambda: run_vqe_benchmark(max_q),
        "qaoa":        lambda: run_qaoa_benchmark(max_q),
        "gradient":    lambda: run_gradient_benchmark(max_q),
        "measurement": lambda: run_measurement_benchmark(max_q),
        "scaling":     lambda: run_scaling_benchmark(max_q),
        "accuracy":    lambda: run_accuracy_benchmark(max_q),
        "memory":      lambda: run_memory_benchmark(max_q),
    }

    for cat in categories:
        if cat in dispatch:
            try:
                dispatch[cat]()
            except Exception as e:
                print(f"\n  ERROR in {cat}: {e}")

    print_summary(categories)

    # JSON export
    if args.json:
        with open(args.json, "w") as f:
            json.dump(_results_json, f, indent=2, default=str)
        print(f"  Results exported to {args.json}")

    print("  Done.")


if __name__ == "__main__":
    main()
