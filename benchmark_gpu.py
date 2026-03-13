"""
Benchmark: CPU vs GPU backends for qforge.

Usage:
    python benchmark_gpu.py
"""
import time
import numpy as np

from qforge.circuit import Qubit
from qforge.gates import H, CNOT, RX, SWAP, X, Y, Z, RZ, CCNOT, ISWAP, SISWAP
from qforge.measurement import measure_one
from qforge import _HAS_CPP, _HAS_METAL, _HAS_CUDA

# ============================================================
# Correctness tests
# ============================================================
def test_correctness(backend_name):
    print(f"\n--- Correctness: {backend_name} ---")
    passed = 0
    total = 0

    # Bell state
    total += 1
    wf = Qubit(2, backend=backend_name)
    H(wf, 0); CNOT(wf, 0, 1)
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    if np.allclose(wf.amplitude, expected, atol=1e-5):
        passed += 1
        print(f"  PASS: Bell state")
    else:
        print(f"  FAIL: Bell state: {wf.amplitude}")

    # GHZ 4-qubit
    total += 1
    wf = Qubit(4, backend=backend_name)
    H(wf, 0)
    for i in range(3): CNOT(wf, i, i+1)
    if np.allclose(abs(wf.amplitude[0]), 1/np.sqrt(2), atol=1e-5) and \
       np.allclose(abs(wf.amplitude[15]), 1/np.sqrt(2), atol=1e-5):
        passed += 1
        print(f"  PASS: GHZ-4")
    else:
        print(f"  FAIL: GHZ-4")

    # X gate
    total += 1
    wf = Qubit(1, backend=backend_name)
    X(wf, 0)
    if np.allclose(abs(wf.amplitude[1]), 1.0, atol=1e-5):
        passed += 1
        print(f"  PASS: X gate")
    else:
        print(f"  FAIL: X gate")

    # RX gate
    total += 1
    wf = Qubit(1, backend=backend_name)
    RX(wf, 0, np.pi/3)
    exp_amp = np.array([np.cos(np.pi/6), -1j*np.sin(np.pi/6)])
    if np.allclose(wf.amplitude, exp_amp, atol=1e-5):
        passed += 1
        print(f"  PASS: RX(pi/3)")
    else:
        print(f"  FAIL: RX(pi/3): {wf.amplitude}")

    # SWAP
    total += 1
    wf = Qubit(2, backend=backend_name)
    X(wf, 0)
    SWAP(wf, 0, 1)
    if np.allclose(abs(wf.amplitude[1]), 1.0, atol=1e-5):
        passed += 1
        print(f"  PASS: SWAP")
    else:
        print(f"  FAIL: SWAP")

    # CCNOT (Toffoli)
    total += 1
    wf = Qubit(3, backend=backend_name)
    X(wf, 0); X(wf, 1)
    CCNOT(wf, 0, 1, 2)
    if np.allclose(abs(wf.amplitude[7]), 1.0, atol=1e-5):
        passed += 1
        print(f"  PASS: CCNOT")
    else:
        print(f"  FAIL: CCNOT: {wf.amplitude}")

    # measure_one
    total += 1
    wf = Qubit(2, backend=backend_name)
    H(wf, 0); CNOT(wf, 0, 1)
    prob = measure_one(wf, 0)
    if np.allclose(prob[0], 0.5, atol=1e-4):
        passed += 1
        print(f"  PASS: measure_one")
    else:
        print(f"  FAIL: measure_one: {prob}")

    print(f"  Result: {passed}/{total} passed")
    return passed == total

# ============================================================
# Cross-backend precision comparison
# ============================================================
def compare_precision(n_qubits=10, depth=20):
    print(f"\n--- Precision: CPU vs GPU ({n_qubits} qubits, depth={depth}) ---")
    backends = ['cpu']
    if _HAS_METAL: backends.append('metal')
    if _HAS_CUDA: backends.append('cuda')

    if len(backends) < 2:
        print("  Skipped: only one backend available")
        return

    np.random.seed(42)
    gates_seq = []
    for _ in range(depth):
        gate_type = np.random.choice(['H', 'CNOT', 'RX'])
        if gate_type == 'H':
            t = np.random.randint(n_qubits)
            gates_seq.append(('H', t))
        elif gate_type == 'CNOT':
            c, t = np.random.choice(n_qubits, 2, replace=False)
            gates_seq.append(('CNOT', int(c), int(t)))
        else:
            t = np.random.randint(n_qubits)
            phi = np.random.uniform(0, 2*np.pi)
            gates_seq.append(('RX', t, phi))

    results = {}
    for b in backends:
        wf = Qubit(n_qubits, backend=b)
        for g in gates_seq:
            if g[0] == 'H': H(wf, g[1])
            elif g[0] == 'CNOT': CNOT(wf, g[1], g[2])
            elif g[0] == 'RX': RX(wf, g[1], g[2])
        results[b] = np.array(wf.amplitude, dtype=complex)

    cpu_amp = results['cpu']
    for b in backends[1:]:
        diff = np.max(np.abs(cpu_amp - results[b]))
        print(f"  max|CPU - {b}| = {diff:.2e}")

# ============================================================
# Performance benchmark
# ============================================================
def bench_circuit(n_qubits, depth, backend, n_warmup=1, n_runs=3):
    """Time a depth-layer H+CNOT circuit."""
    for _ in range(n_warmup):
        wf = Qubit(n_qubits, backend=backend)
        for d in range(depth):
            for q in range(n_qubits):
                H(wf, q)
            for q in range(0, n_qubits - 1, 2):
                CNOT(wf, q, q + 1)

    times = []
    for _ in range(n_runs):
        wf = Qubit(n_qubits, backend=backend)
        t0 = time.perf_counter()
        for d in range(depth):
            for q in range(n_qubits):
                H(wf, q)
            for q in range(0, n_qubits - 1, 2):
                CNOT(wf, q, q + 1)
        # Force sync for GPU
        _ = wf.amplitude[0]
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def run_benchmarks():
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK: H+CNOT circuit, depth=10")
    print("="*60)

    backends = []
    if _HAS_CPP: backends.append('cpu')
    if _HAS_METAL: backends.append('metal')
    if _HAS_CUDA: backends.append('cuda')

    qubit_range = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    depth = 10

    # Header
    header = f"{'Qubits':>8}"
    for b in backends:
        header += f"  {b:>10}(s)"
    if len(backends) > 1:
        header += f"  {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for nq in qubit_range:
        row = f"{nq:>8}"
        times = {}
        for b in backends:
            try:
                t = bench_circuit(nq, depth, b)
                times[b] = t
                row += f"  {t:>12.6f}"
            except Exception as e:
                row += f"  {'ERROR':>12}"
                times[b] = None

        if len(backends) > 1 and times.get('cpu') and times.get(backends[-1]):
            speedup = times['cpu'] / times[backends[-1]]
            row += f"  {speedup:>9.1f}x"

        print(row)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Qforge GPU Benchmark")
    print(f"  CPU (C++): {_HAS_CPP}")
    print(f"  CUDA:      {_HAS_CUDA}")
    print(f"  Metal:     {_HAS_METAL}")

    all_ok = True

    if _HAS_CPP:
        all_ok &= test_correctness('cpu')
    if _HAS_METAL:
        all_ok &= test_correctness('metal')
    if _HAS_CUDA:
        all_ok &= test_correctness('cuda')

    if all_ok:
        compare_precision()
        run_benchmarks()
    else:
        print("\nCorrectness tests failed — skipping benchmarks.")
