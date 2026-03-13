#!/usr/bin/env python3
"""Compare C++ engine vs pure Python performance."""
import numpy as np
import time
import sys
import itertools

# Force pure Python mode for comparison
from qforge.wavefunction import Wavefunction

def make_python_circuit(n):
    """Create circuit WITHOUT C++ backend."""
    states = ["".join(seq) for seq in itertools.product("01", repeat=n)]
    amp = np.zeros(2**n, dtype=complex)
    amp[0] = 1.0
    wf = Wavefunction(np.array(states), amp)
    wf._sv = None  # force Python path
    return wf

def make_cpp_circuit(n):
    """Create circuit WITH C++ backend."""
    from qforge import Qcircuit
    return Qcircuit.Qubit(n)

from qforge import Qgates

def run_benchmark(make_circuit_fn, n_qubits, depth=10, repeats=3):
    times = []
    for _ in range(repeats):
        c = make_circuit_fn(n_qubits)
        start = time.perf_counter()
        for _ in range(depth):
            for q in range(n_qubits):
                Qgates.H(c, q)
            for q in range(n_qubits - 1):
                Qgates.CNOT(c, q, q + 1)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)

print("="*70)
print("PERFORMANCE COMPARISON: C++ Engine vs Pure Python")
print("="*70)
print(f"\n{'Qubits':>8} {'Python (s)':>12} {'C++ (s)':>12} {'Speedup':>10}")
print("-" * 46)

for n in [2, 4, 6, 8, 10, 12, 13, 14]:
    depth = 10
    try:
        t_py = run_benchmark(make_python_circuit, n, depth)
        t_cpp = run_benchmark(make_cpp_circuit, n, depth)
        speedup = t_py / t_cpp if t_cpp > 0 else float('inf')
        print(f"{n:>8} {t_py:>12.6f} {t_cpp:>12.6f} {speedup:>9.1f}x")
    except Exception as e:
        print(f"{n:>8} ERROR: {e}")

# Extended C++ only benchmarks
print(f"\n{'Qubits':>8} {'C++ (s)':>12}  (Python too slow for these)")
print("-" * 40)
for n in [16, 18, 20, 22]:
    depth = 10
    try:
        t_cpp = run_benchmark(make_cpp_circuit, n, depth)
        print(f"{n:>8} {t_cpp:>12.6f}")
    except Exception as e:
        print(f"{n:>8} ERROR: {e}")

print("\nDone!")
