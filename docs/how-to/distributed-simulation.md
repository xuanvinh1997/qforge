# Distributed Simulation (MPI)

For exact statevector simulation beyond ~22 qubits, Qforge can shard the amplitude vector across multiple MPI processes — on a single machine or across a cluster of nodes.

## How it works

The $2^n$ amplitude vector is split evenly across $R$ MPI ranks, where $R$ must be a power of 2. Each rank holds $2^{n-k}$ amplitudes, with $k = \log_2 R$.

- **Local qubits** (index $\ge k$): gates operate entirely within one rank — no network traffic.
- **Global qubits** (index $< k$): gates require `MPI_Sendrecv` to exchange amplitudes with a partner rank. This is the main bottleneck.

!!! tip "Performance"
    Place your most frequently gated qubits at **high indices** (local) whenever possible. Gates on low-index (global) qubits are orders of magnitude slower due to network round-trips.

## Scale reference

| Ranks | Global qubits | Practical max qubits | Memory / rank |
|------:|:-------------:|:--------------------:|--------------:|
| 2     | 1             | ~27                  | 1 GB          |
| 4     | 2             | ~29                  | 1 GB          |
| 8     | 3             | ~31                  | 2 GB          |
| 16    | 4             | ~34                  | 8 GB          |
| 64    | 6             | ~36                  | 8 GB          |

---

## 1. Install MPI

=== "Ubuntu / Debian"

    ```bash
    sudo apt install libopenmpi-dev openmpi-bin
    ```

=== "macOS"

    ```bash
    brew install open-mpi
    ```

=== "MPICH"

    ```bash
    sudo apt install mpich
    ```

## 2. Build Qforge with MPI support

```bash
QFORGE_MPI=1 pip install -e .
```

Verify:

```bash
python -c "from qforge._qforge_distributed import DistributedStateVector; print('MPI backend OK')"
```

## 3. Single-machine usage

Run across multiple cores on one machine:

```bash
mpirun -np 4 python my_circuit.py
```

```python
# my_circuit.py
from qforge.distributed import DistributedQubit
from qforge.gates import H, CNOT
from qforge.measurement import measure_one

wf = DistributedQubit(28)   # 4 ranks -> 2 global qubits
H(wf, 0)
for i in range(27):
    CNOT(wf, i, i + 1)

print(measure_one(wf, 0))   # [0.5, 0.5]
```

---

## 4. Multi-node cluster setup

### 4a. Prerequisites

All nodes must have:

- The **same version** of Qforge built with `QFORGE_MPI=1`
- The **same Python path** (or a shared filesystem like NFS)
- **Passwordless SSH** between all nodes

### 4b. Set up passwordless SSH

Run this on the master node:

```bash
ssh-keygen -t rsa -N ""
ssh-copy-id user@node2
ssh-copy-id user@node3
ssh-copy-id user@node4
```

Test with `ssh user@node2 hostname` — it should return the hostname without a password prompt.

### 4c. Create a hostfile

List every node and how many ranks (slots) it should run:

```title="hostfile"
node1 slots=4
node2 slots=4
node3 slots=4
node4 slots=4
```

Or use IP addresses:

```title="hostfile"
192.168.1.10 slots=4
192.168.1.11 slots=4
192.168.1.12 slots=8
```

!!! warning
    Total slots must be a **power of 2** (2, 4, 8, 16, 32, ...). The constructor will raise an error otherwise.

### 4d. Run across the cluster

```bash
# 16 ranks across 4 nodes
mpirun --hostfile hostfile -np 16 python my_circuit.py
```

Or specify nodes inline:

```bash
mpirun -H node1:4,node2:4,node3:4,node4:4 python my_circuit.py
```

---

## 5. Complete example: 34-qubit GHZ state

```python
# ghz_distributed.py
from qforge.distributed import DistributedQubit
from qforge.gates import H, CNOT
from qforge.measurement import measure_one

n = 34
wf = DistributedQubit(n)

# Build GHZ: H on qubit 0, then cascade CNOT
H(wf, 0)
for i in range(n - 1):
    CNOT(wf, i, i + 1)

# Check: qubit 0 should be in equal superposition
p = measure_one(wf, 0)
print(f"P(|0>) = {p[0]:.6f}, P(|1>) = {p[1]:.6f}")
```

```bash
# Run on 16 ranks (4 nodes x 4 cores)
mpirun --hostfile hostfile -np 16 python ghz_distributed.py
```

## 6. API overview

`DistributedQubit` returns a standard `Wavefunction` object — the same gate and measurement functions work without any code changes:

```python
from qforge.distributed import DistributedQubit
from qforge.gates import H, X, RY, CNOT, CRZ, SWAP
from qforge.measurement import measure_all, measure_one, pauli_expectation

wf = DistributedQubit(30)

# All gates work transparently
H(wf, 0)
RY(wf, 5, 0.42)
CNOT(wf, 0, 1)
CRZ(wf, 2, 3, 1.57)

# Measurement works across ranks
print(measure_one(wf, 0))
print(pauli_expectation(wf, 0, 'Z'))
```

See the full API in the [qforge.distributed reference](../api/distributed.md).

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: Distributed backend not available` | Rebuild with `QFORGE_MPI=1 pip install -e .` |
| `n_ranks must be a power of 2` | Use `-np` with a power of 2 (2, 4, 8, 16, ...) |
| `Too many ranks for qubit count` | Reduce ranks or increase qubits. Need `n_qubits > log2(n_ranks)` |
| SSH connection refused | Run `ssh-copy-id user@nodeX` and verify with `ssh user@nodeX hostname` |
| Different Python paths on nodes | Use a shared filesystem (NFS) or install identically on each node |
| Slow performance | Minimize gates on global qubits (index 0 to `log2(n_ranks)-1`) |
