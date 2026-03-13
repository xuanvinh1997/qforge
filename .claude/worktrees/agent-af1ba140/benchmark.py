#!/usr/bin/env python3
"""Correctness validation and performance benchmark for Qsun C++ engine."""
import numpy as np
import time
import sys

# Check C++ engine availability
from Qforge import _HAS_CPP, _HAS_METAL
print(f"C++ engine available: {_HAS_CPP}")

from Qforge import Qcircuit, Qgates, Qmeas, Qdata

# Metal uses float32 internally — use relaxed tolerance
_DEFAULT_TOL = 1e-6 if _HAS_METAL else 1e-12

# ============================================================
# CORRECTNESS TESTS
# ============================================================
print("\n" + "="*60)
print("CORRECTNESS TESTS")
print("="*60)

def assert_close(a, b, tol=None, msg=""):
    if tol is None:
        tol = _DEFAULT_TOL
    diff = np.max(np.abs(np.array(a) - np.array(b)))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{status}] {msg} (max diff: {diff:.2e})")
    if diff >= tol:
        print(f"    Expected: {b}")
        print(f"    Got:      {a}")
    return diff < tol

all_pass = True

# Test 1: Bell state
print("\n1. Bell State (H + CNOT)")
c = Qcircuit.Qubit(2)
Qgates.H(c, 0)
Qgates.CNOT(c, 0, 1)
expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
all_pass &= assert_close(c.amplitude, expected, msg="Bell state amplitudes")

# Test 2: GHZ state
print("\n2. GHZ State (3 qubits)")
c = Qcircuit.Qubit(3)
Qgates.H(c, 0)
Qgates.CNOT(c, 0, 1)
Qgates.CNOT(c, 1, 2)
expected = np.zeros(8, dtype=complex)
expected[0] = 1/np.sqrt(2)
expected[7] = 1/np.sqrt(2)
all_pass &= assert_close(c.amplitude, expected, msg="GHZ amplitudes")

# Test 3: All single-qubit gates
print("\n3. Single-Qubit Gates")
for gate_name, gate_fn, args in [
    ("X", Qgates.X, []),
    ("Y", Qgates.Y, []),
    ("Z", Qgates.Z, []),
    ("H", Qgates.H, []),
    ("RX(pi/3)", Qgates.RX, [np.pi/3]),
    ("RY(pi/4)", Qgates.RY, [np.pi/4]),
    ("RZ(pi/6)", Qgates.RZ, [np.pi/6]),
    ("Phase(pi/5)", Qgates.Phase, [np.pi/5]),
    ("Xsquare", Qgates.Xsquare, []),
]:
    c = Qcircuit.Qubit(1)
    gate_fn(c, 0, *args)
    # Verify unitarity: sum of |amp|^2 = 1
    norm = np.sum(np.abs(c.amplitude)**2)
    all_pass &= assert_close(norm, 1.0, msg=f"{gate_name} unitarity")

# Test 4: Controlled gates
print("\n4. Controlled Gates")
# CNOT: |10> -> |11>
c = Qcircuit.Qubit(2)
Qgates.X(c, 0)  # |10>
Qgates.CNOT(c, 0, 1)  # -> |11>
all_pass &= assert_close(c.amplitude[3], 1.0, msg="CNOT |10>->|11>")

# CCNOT: |110> -> |111>
c = Qcircuit.Qubit(3)
Qgates.X(c, 0)
Qgates.X(c, 1)  # |110>
Qgates.CCNOT(c, 0, 1, 2)  # -> |111>
all_pass &= assert_close(c.amplitude[7], 1.0, msg="CCNOT |110>->|111>")

# Test 5: SWAP gates
print("\n5. SWAP Gates")
c = Qcircuit.Qubit(2)
Qgates.X(c, 0)  # |10>
Qgates.SWAP(c, 0, 1)  # -> |01>
all_pass &= assert_close(c.amplitude[1], 1.0, msg="SWAP |10>->|01>")

# ISWAP: |10> -> i|01>
c = Qcircuit.Qubit(2)
Qgates.X(c, 0)  # |10>
Qgates.ISWAP(c, 0, 1)
all_pass &= assert_close(c.amplitude[1], 1j, msg="ISWAP |10>->i|01>")

# SISWAP
c = Qcircuit.Qubit(2)
Qgates.X(c, 0)  # |10>
Qgates.SISWAP(c, 0, 1)
norm = np.sum(np.abs(c.amplitude)**2)
all_pass &= assert_close(norm, 1.0, msg="SISWAP unitarity")

# Test 6: Measurement
print("\n6. Measurement")
c = Qcircuit.Qubit(2)
Qgates.H(c, 0)
Qgates.CNOT(c, 0, 1)
probs = Qmeas.measure_one(c, 0)
all_pass &= assert_close(probs, [0.5, 0.5], msg="Bell state measure_one")

# Pauli expectations on |+> state
c = Qcircuit.Qubit(1)
Qgates.H(c, 0)  # |+>
exp_x = Qmeas.pauli_expectation(c, 0, 'X')
exp_z = Qmeas.pauli_expectation(c, 0, 'Z')
all_pass &= assert_close(exp_x, 1.0, msg="<+|X|+> = 1")
all_pass &= assert_close(exp_z, 0.0, msg="<+|Z|+> = 0")

# Test 7: PauliZ expectation (Qdata)
print("\n7. PauliZ Expectation")
c = Qcircuit.Qubit(2)
Qgates.H(c, 0)
Qgates.CNOT(c, 0, 1)
pz = Qdata.PauliZExpectation(c)
all_pass &= assert_close(pz.one_body(0), 0.0, msg="Bell <Z0>=0")
all_pass &= assert_close(pz.two_body(0, 1), 1.0, msg="Bell <Z0Z1>=1")

# Test 8: Entropy
print("\n8. Entanglement Entropy")
c = Qcircuit.Qubit(2)
Qgates.H(c, 0)
Qgates.CNOT(c, 0, 1)
ee = Qdata.EntanglementEntropy(c)
entropy = ee.von_neumann_entropy(keep_qubits=[0])
all_pass &= assert_close(entropy, 1.0, msg="Bell entropy = 1 bit")

# Test 9: Random circuit consistency (larger qubit count)
print("\n9. Random Circuit (8 qubits, 50 gates)")
np.random.seed(42)
n = 8
c = Qcircuit.Qubit(n)
gates_list = [Qgates.H, Qgates.X, Qgates.Y, Qgates.Z]
for _ in range(50):
    q = np.random.randint(0, n)
    g = gates_list[np.random.randint(0, len(gates_list))]
    g(c, q)
    if np.random.random() > 0.5 and n > 1:
        q1, q2 = np.random.choice(n, 2, replace=False)
        Qgates.CNOT(c, int(q1), int(q2))
norm = np.sum(np.abs(c.amplitude)**2)
all_pass &= assert_close(norm, 1.0, msg="Random circuit unitarity")

print("\n" + "="*60)
print(f"ALL TESTS: {'PASSED' if all_pass else 'SOME FAILED'}")
print("="*60)

# ============================================================
# PERFORMANCE BENCHMARKS
# ============================================================
print("\n" + "="*60)
print("PERFORMANCE BENCHMARKS")
print("="*60)

def benchmark_circuit(n_qubits, depth=10, repeats=3):
    """Benchmark: depth layers of H + CNOT chain."""
    times = []
    for _ in range(repeats):
        c = Qcircuit.Qubit(n_qubits)
        start = time.perf_counter()
        for _ in range(depth):
            for q in range(n_qubits):
                Qgates.H(c, q)
            for q in range(n_qubits - 1):
                Qgates.CNOT(c, q, q + 1)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times)

print(f"\n{'Qubits':>8} {'Depth':>6} {'Time (s)':>12} {'Gates':>8}")
print("-" * 40)

for n in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    depth = 10
    try:
        t = benchmark_circuit(n, depth)
        total_gates = depth * (n + n - 1)
        print(f"{n:>8} {depth:>6} {t:>12.6f} {total_gates:>8}")
    except Exception as e:
        print(f"{n:>8} {depth:>6} {'ERROR':>12} - {e}")

# Single gate benchmark
print(f"\nSingle H gate timing:")
print(f"{'Qubits':>8} {'Time (us)':>12}")
print("-" * 24)
for n in [8, 10, 12, 14, 16, 18, 20]:
    c = Qcircuit.Qubit(n)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        Qgates.H(c, 0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_us = np.median(times) * 1e6
    print(f"{n:>8} {avg_us:>12.1f}")

print("\nDone!")
