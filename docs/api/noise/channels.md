# Qforge.noise.channels

Quantum noise channels defined by their Kraus operators. Each channel satisfies
the trace-preservation condition: sum_k K_k^dagger K_k = I.

## Usage

```python
from Qforge.noise.channels import BitFlip, Depolarizing, AmplitudeDamping

bf = BitFlip(p=0.05)
print(bf.kraus_ops)              # [sqrt(0.95)*I, sqrt(0.05)*X]
print(bf.is_trace_preserving())  # True

dep = Depolarizing(p=0.01)
ad = AmplitudeDamping(gamma=0.02)
```

## Base Class

### `QuantumChannel` (ABC)

Abstract base class for all noise channels.

**Methods:**

- `kraus_ops` -- Property returning the list of Kraus operator matrices.
- `is_trace_preserving(atol=1e-10)` -- Check that sum_k K_k^dagger K_k = I.

## Channel Classes

### `BitFlip(p)`

Bit-flip channel. K0 = sqrt(1-p) * I, K1 = sqrt(p) * X.

**Parameters:** `p` -- Probability of a bit flip, in [0, 1].

---

### `PhaseFlip(p)`

Phase-flip (dephasing) channel. K0 = sqrt(1-p) * I, K1 = sqrt(p) * Z.

**Parameters:** `p` -- Probability of a phase flip, in [0, 1].

---

### `Depolarizing(p)`

Depolarizing channel. Applies I, X, Y, Z each with appropriate probability.
The state is replaced by the maximally mixed state with probability p.

**Parameters:** `p` -- Depolarizing probability, in [0, 1].

---

### `AmplitudeDamping(gamma)`

Amplitude damping channel modeling energy dissipation (T1 decay).

K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]]

**Parameters:** `gamma` -- Damping parameter, in [0, 1].

---

### `PhaseDamping(gamma)`

Phase damping channel modeling pure dephasing (T2 process without T1).

**Parameters:** `gamma` -- Damping parameter, in [0, 1].

---

### `ThermalRelaxation(t1, t2, gate_time, excited_population=0.0)`

Combined thermal relaxation channel modeling both T1 and T2 processes.

**Parameters:**

- `t1` -- T1 relaxation time.
- `t2` -- T2 dephasing time (must satisfy `t2 <= 2*t1`).
- `gate_time` -- Duration of the gate.
- `excited_population` -- Thermal equilibrium excited-state population (default 0).

---

### `ReadoutError(p0_given1, p1_given0)`

Classical readout error (not a quantum channel). Defines a confusion matrix
for measurement outcomes.

**Parameters:**

- `p0_given1` -- Probability of reading 0 when the state is |1>.
- `p1_given0` -- Probability of reading 1 when the state is |0>.

---

### `KrausChannel(kraus_ops)`

User-defined channel from an explicit list of Kraus operators.

**Parameters:** `kraus_ops` -- List of numpy arrays (Kraus matrices).

## Full API

::: Qforge.noise.channels
    options:
      show_source: false
