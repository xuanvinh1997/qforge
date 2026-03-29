# Gate Catalog

Complete reference for all gates in `qforge.gates`. All gate functions take a
`wavefunction` as the first argument and modify it in-place.

## Single-Qubit Gates

| Gate | Signature | Matrix | Category |
|------|-----------|--------|----------|
| `H` | `H(wf, n)` | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | Clifford |
| `X` | `X(wf, n)` | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Pauli |
| `Y` | `Y(wf, n)` | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ | Pauli |
| `Z` | `Z(wf, n)` | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Pauli |
| `S` | `S(wf, n)` | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ | Clifford |
| `T` | `T(wf, n)` | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ | Non-Clifford |
| `Xsquare` | `Xsquare(wf, n)` | $\frac{1}{2}\begin{pmatrix}1+i&1-i\\1-i&1+i\end{pmatrix}$ | Non-Clifford |

## Rotation Gates

| Gate | Signature | Matrix | Category |
|------|-----------|--------|----------|
| `RX` | `RX(wf, n, theta)` | $\begin{pmatrix}\cos\frac{\theta}{2}&-i\sin\frac{\theta}{2}\\-i\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{pmatrix}$ | Rotation |
| `RY` | `RY(wf, n, theta)` | $\begin{pmatrix}\cos\frac{\theta}{2}&-\sin\frac{\theta}{2}\\\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{pmatrix}$ | Rotation |
| `RZ` | `RZ(wf, n, theta)` | $\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}$ | Rotation |
| `Phase` | `Phase(wf, n, theta)` | $\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}$ | Phase |

## Controlled Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `CNOT` | `CNOT(wf, control, target)` | Controlled-NOT (CX). Flips target if control is \|1>. | Clifford |
| `CCNOT` | `CCNOT(wf, c1, c2, target)` | Toffoli gate. Flips target if both controls are \|1>. | Multi-controlled |
| `CRX` | `CRX(wf, control, target, theta)` | Controlled RX rotation. | Controlled rotation |
| `CRY` | `CRY(wf, control, target, theta)` | Controlled RY rotation. | Controlled rotation |
| `CRZ` | `CRZ(wf, control, target, theta)` | Controlled RZ rotation. | Controlled rotation |
| `CPhase` | `CPhase(wf, control, target, theta)` | Controlled phase gate. | Controlled phase |
| `CP` | `CP(wf, control, target, theta)` | Alias for `CPhase`. | Controlled phase |
| `OR` | `OR(wf, control, target)` | Controlled-OR: flips target if control is \|0>. | Controlled |

## Entangling / SWAP Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `SWAP` | `SWAP(wf, n1, n2)` | Swap the states of two qubits. | SWAP |
| `CSWAP` | `CSWAP(wf, control, n1, n2)` | Fredkin (controlled-SWAP) gate. | Controlled SWAP |
| `ISWAP` | `ISWAP(wf, n1, n2)` | iSWAP: swap with phase factor *i*. | Entangling |
| `SISWAP` | `SISWAP(wf, n1, n2)` | Square root of iSWAP. | Entangling |

## Multi-Controlled Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `mcx` | `mcx(wf, controls, target)` | Multi-controlled X. `controls` is a list of qubit indices. | Multi-controlled |
| `mcz` | `mcz(wf, controls, target)` | Multi-controlled Z. `controls` is a list of qubit indices. | Multi-controlled |
| `mcp` | `mcp(wf, controls, target, theta)` | Multi-controlled Phase. `controls` is a list of qubit indices. | Multi-controlled |

## Custom Unitary

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `QubitUnitary` | `QubitUnitary(wf, qubits, matrix)` | Apply an arbitrary unitary matrix. | Custom |

## Noise Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `E` | `E(wf, n, p)` | Single-qubit depolarizing channel with probability *p*. | Noise |
| `E_all` | `E_all(wf, p)` | Depolarizing channel on all qubits with probability *p*. | Noise |

## Qudit Gates (`qforge.qudit_gates`)

Gates for qudits with local dimension d >= 2. Import from `qforge.qudit_gates`.

### Single-Qudit Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `Hd` | `Hd(wf, target)` | Qudit Hadamard (DFT / sqrt(d)). | Fourier |
| `X01` | `X01(wf, target)` | Swap levels \|0> and \|1>. | Subspace swap |
| `X02` | `X02(wf, target)` | Swap levels \|0> and \|2>. | Subspace swap |
| `X12` | `X12(wf, target)` | Swap levels \|1> and \|2>. | Subspace swap |
| `CLOCK` | `CLOCK(wf, target)` | Cyclic shift: \|k> -> \|k+1 mod d>. | Shift |
| `ZPHASE` | `ZPHASE(wf, target)` | Ternary phase: diag(1, omega, omega^2, ...). | Phase |

### Qudit Rotation Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `R01` | `R01(wf, target, theta)` | Givens rotation in \|0>-\|1> subspace. | Rotation |
| `R02` | `R02(wf, target, theta)` | Givens rotation in \|0>-\|2> subspace. | Rotation |
| `R12` | `R12(wf, target, theta)` | Givens rotation in \|1>-\|2> subspace. | Rotation |
| `RGM` | `RGM(wf, target, gen, angle)` | Rotation by Gell-Mann generator lambda_k. | Rotation |

### Qudit Entangling Gates

| Gate | Signature | Description | Category |
|------|-----------|-------------|----------|
| `CSUM` | `CSUM(wf, control, target)` | \|c,t> -> \|c, (t+c) mod d>. Qutrit CNOT analog. | Entangling |
| `QUDIT_SWAP` | `QUDIT_SWAP(wf, t1, t2)` | Swap two qudits (any dimension). | SWAP |
| `apply_qudit_gate` | `apply_qudit_gate(wf, target, gate)` | Apply arbitrary d x d unitary. | Custom |
| `apply_controlled_qudit_gate` | `apply_controlled_qudit_gate(wf, ctrl, val, tgt, gate)` | Controlled d x d gate. | Controlled |

## Parameter Summary

| Gate | # Qubits | # Parameters | Parametric? |
|------|----------|-------------|-------------|
| H | 1 | 0 | No |
| X | 1 | 0 | No |
| Y | 1 | 0 | No |
| Z | 1 | 0 | No |
| S | 1 | 0 | No |
| T | 1 | 0 | No |
| Xsquare | 1 | 0 | No |
| RX | 1 | 1 (theta) | Yes |
| RY | 1 | 1 (theta) | Yes |
| RZ | 1 | 1 (theta) | Yes |
| Phase | 1 | 1 (theta) | Yes |
| CNOT | 2 | 0 | No |
| CCNOT | 3 | 0 | No |
| CRX | 2 | 1 (theta) | Yes |
| CRY | 2 | 1 (theta) | Yes |
| CRZ | 2 | 1 (theta) | Yes |
| CPhase / CP | 2 | 1 (theta) | Yes |
| OR | 2 | 0 | No |
| SWAP | 2 | 0 | No |
| CSWAP | 3 | 0 | No |
| ISWAP | 2 | 0 | No |
| SISWAP | 2 | 0 | No |
| mcx | N+1 | 0 | No |
| mcz | N+1 | 0 | No |
| mcp | N+1 | 1 (theta) | Yes |
| QubitUnitary | K | 0 (matrix) | No |
| E | 1 | 1 (p) | N/A (noise) |
| E_all | all | 1 (p) | N/A (noise) |
