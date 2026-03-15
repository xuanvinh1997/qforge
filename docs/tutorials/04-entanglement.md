# Tutorial 4: Entanglement

This tutorial covers creating entangled states and analyzing entanglement using Qforge's analysis tools.

## Setup

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT, X
from Qforge.measurement import measure_all, pauli_expectation
from Qforge.data import PauliZExpectation, ConnectedCorrelator, EntanglementEntropy
import numpy as np
```

## Bell States

The four Bell states are maximally entangled 2-qubit states. Each is a specific combination of Hadamard and CNOT with optional bit/phase flips.

```python
def make_bell(state_index):
    """Create one of the four Bell states (0-3)."""
    wf = Qubit(2)
    if state_index in (2, 3):
        X(wf, 0)               # Flip qubit 0 for Psi states
    H(wf, 0)
    CNOT(wf, 0, 1)
    if state_index in (1, 3):
        from Qforge.gates import Z
        Z(wf, 0)               # Phase flip for - variants
    return wf

# |Phi+> = (|00> + |11>) / sqrt(2)
phi_plus = make_bell(0)
print("Phi+:", np.round(phi_plus.probabilities(), 4))
# [0.5 0.  0.  0.5]

# |Phi-> = (|00> - |11>) / sqrt(2)
phi_minus = make_bell(1)
print("Phi-:", np.round(phi_minus.probabilities(), 4))
# [0.5 0.  0.  0.5]

# |Psi+> = (|01> + |10>) / sqrt(2)
psi_plus = make_bell(2)
print("Psi+:", np.round(psi_plus.probabilities(), 4))
# [0.  0.5 0.5 0. ]

# |Psi-> = (|01> - |10>) / sqrt(2)
psi_minus = make_bell(3)
print("Psi-:", np.round(psi_minus.probabilities(), 4))
# [0.  0.5 0.5 0. ]
```

> **Note:** Bell states have identical marginal probabilities per qubit (50/50) but are distinguished by their correlations and phases.

## GHZ State

The GHZ (Greenberger-Horne-Zeilinger) state generalizes the Bell state to n qubits: (|00...0> + |11...1>) / sqrt(2).

```python
def make_ghz(n):
    """Create an n-qubit GHZ state."""
    wf = Qubit(n)
    H(wf, 0)
    for i in range(n - 1):
        CNOT(wf, i, i + 1)
    return wf

ghz4 = make_ghz(4)
print("GHZ-4 probabilities:")
print(np.round(ghz4.probabilities(), 4))
# [0.5 0.  0.  ...  0.  0.5]  -- only |0000> and |1111>

# Verify with sampling
states, counts = measure_all(ghz4, 10000)
print(dict(zip(states, counts)))
# {'0000': ~5000, '1111': ~5000}
```

## PauliZExpectation

`PauliZExpectation` computes n-body Pauli-Z expectation values. These measure correlations between qubits in the computational basis.

```python
wf = make_ghz(4)
pz = PauliZExpectation(wf)
```

### One-Body: `<Z_i>`

For the GHZ state, each qubit is individually maximally mixed.

```python
for i in range(4):
    print(f"<Z_{i}> = {pz.one_body(i):.4f}")
# <Z_0> = 0.0000
# <Z_1> = 0.0000
# <Z_2> = 0.0000
# <Z_3> = 0.0000
```

### Two-Body: `<Z_i Z_j>`

The GHZ state has perfect two-body ZZ correlations.

```python
print(f"<Z_0 Z_1> = {pz.two_body(0, 1):.4f}")  # 1.0
print(f"<Z_0 Z_2> = {pz.two_body(0, 2):.4f}")  # 1.0
print(f"<Z_1 Z_3> = {pz.two_body(1, 3):.4f}")  # 1.0
```

All pairs have `<Z_i Z_j> = 1` because in both GHZ components (`|0000>` and `|1111>`), every pair of qubits agrees.

### Three-Body and Four-Body

```python
print(f"<Z_0 Z_1 Z_2> = {pz.three_body(0, 1, 2):.4f}")      # 0.0
print(f"<Z_0 Z_1 Z_2 Z_3> = {pz.four_body(0, 1, 2, 3):.4f}")  # 1.0
```

The three-body correlator is zero because the parity of three qubits differs between `|0000>` (even) and `|1111>` (odd for 3 qubits, since 3 ones). The four-body correlator is +1 because 4 ones has even parity.

## ConnectedCorrelator

Connected correlators isolate genuine multi-body correlations by subtracting disconnected parts.

```python
wf = make_ghz(4)
cc = ConnectedCorrelator(wf)
```

### Two-Point Connected Correlator

`u2(i, j) = <Z_i Z_j> - <Z_i><Z_j>`

```python
print(f"u2(0,1) = {cc.u2(0, 1):.4f}")  # 1.0
# Since <Z_0> = <Z_1> = 0, u2 = <Z_0 Z_1> - 0 = 1.0
```

### Three-Point Connected Correlator

```python
print(f"u3(0,1,2) = {cc.u3(0, 1, 2):.4f}")  # 0.0
```

### Four-Point Connected Correlator

```python
print(f"u4(0,1,2,3) = {cc.u4(0, 1, 2, 3):.4f}")
```

### Comparing Product State vs Entangled State

Connected correlators are zero for product (unentangled) states:

```python
# Product state: each qubit independent
wf_prod = Qubit(4)
H(wf_prod, 0)
H(wf_prod, 1)
H(wf_prod, 2)
H(wf_prod, 3)

cc_prod = ConnectedCorrelator(wf_prod)
print(f"Product state u2(0,1) = {cc_prod.u2(0, 1):.4f}")  # 0.0

# Entangled state
cc_ghz = ConnectedCorrelator(make_ghz(4))
print(f"GHZ state u2(0,1) = {cc_ghz.u2(0, 1):.4f}")       # 1.0
```

## EntanglementEntropy

`EntanglementEntropy` computes von Neumann entropy and reduced density matrices to quantify entanglement.

```python
wf = make_ghz(4)
ee = EntanglementEntropy(wf)
```

### Reduced Density Matrix

Trace out some qubits to get the density matrix of a subsystem.

```python
# Reduced density matrix of qubit 0
rho_0 = ee.reduced_density_matrix([0])
print("rho(qubit 0):")
print(np.round(rho_0, 4))
# [[0.5 0. ]
#  [0.  0.5]]  -- maximally mixed
```

For the GHZ state, each individual qubit is maximally mixed (50/50), reflecting maximal entanglement.

```python
# Reduced density matrix of qubits 0 and 1
rho_01 = ee.reduced_density_matrix([0, 1])
print("rho(qubits 0,1):")
print(np.round(rho_01, 4))
# [[0.5 0.  0.  0.5]
#  [0.  0.  0.  0. ]
#  [0.  0.  0.  0. ]
#  [0.5 0.  0.  0.5]]
```

### Von Neumann Entropy

`von_neumann_entropy(keep_qubits, base=2)` computes `S = -Tr(rho * log(rho))`.

```python
# Entropy of single qubit in GHZ state: S = 1 bit (maximally entangled)
s0 = ee.von_neumann_entropy([0])
print(f"S(qubit 0) = {s0:.4f}")  # 1.0

# Entropy of full pure state: S = 0
s_full = ee.von_neumann_entropy()
print(f"S(full state) = {s_full:.4f}")  # 0.0
```

### Bipartite Entanglement Entropy

`entanglement_entropy(bipartition)` computes the entropy across a bipartition.

```python
# Split: {0} vs {1,2,3}
s_01 = ee.entanglement_entropy([0])
print(f"S({{0}} | {{1,2,3}}) = {s_01:.4f}")  # 1.0

# Split: {0,1} vs {2,3}
s_0012 = ee.entanglement_entropy([0, 1])
print(f"S({{0,1}} | {{2,3}}) = {s_0012:.4f}")  # 1.0

# Explicit bipartition tuple
s_explicit = ee.entanglement_entropy(([0, 1], [2, 3]))
print(f"S({{0,1}} | {{2,3}}) = {s_explicit:.4f}")  # 1.0
```

For the GHZ state, any bipartition gives exactly 1 bit of entanglement entropy.

### Entropy with Different Bases

```python
# Natural log (nats)
s_nat = ee.von_neumann_entropy([0], base=np.e)
print(f"S (nats) = {s_nat:.4f}")  # 0.6931 = ln(2)

# Base-10
s_10 = ee.von_neumann_entropy([0], base=10)
print(f"S (base-10) = {s_10:.4f}")  # 0.3010
```

## Full Example: Entanglement Profile of a GHZ State

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT
from Qforge.data import PauliZExpectation, ConnectedCorrelator, EntanglementEntropy

n = 6
wf = Qubit(n)
H(wf, 0)
for i in range(n - 1):
    CNOT(wf, i, i + 1)

# Correlators
pz = PauliZExpectation(wf)
print("Two-body ZZ correlations:")
for i in range(n):
    for j in range(i + 1, n):
        print(f"  <Z_{i} Z_{j}> = {pz.two_body(i, j):.4f}")

# Connected correlators
cc = ConnectedCorrelator(wf)
print(f"\nConnected u2(0,1) = {cc.u2(0, 1):.4f}")
print(f"Connected u2(0,{n-1}) = {cc.u2(0, n-1):.4f}")

# Entanglement entropy across all bipartitions
ee = EntanglementEntropy(wf)
print("\nEntanglement entropy (bipartition at each cut):")
for k in range(1, n):
    s = ee.entanglement_entropy(list(range(k)))
    print(f"  S({list(range(k))} | {list(range(k, n))}) = {s:.4f}")
```
