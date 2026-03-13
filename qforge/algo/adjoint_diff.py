# -*- coding: utf-8 -*-
"""Adjoint differentiation method for variational quantum algorithms.

This provides an O(1) extra memory gradient method (vs O(2P) for parameter-shift),
by running the circuit forward then backward, computing gradient contributions
at each parameterised gate.
"""
from __future__ import annotations

import numpy as np
from qforge.circuit import Qubit
from qforge.ir import Circuit, GateOp, _dispatch_op, _adjoint_op


def adjoint_differentiation(circuit: Circuit, hamiltonian,
                             params: np.ndarray,
                             backend: str = 'python') -> np.ndarray:
    """Compute the gradient of ⟨H⟩ w.r.t. circuit parameters using adjoint method.

    The adjoint differentiation method:
    1. Forward pass: apply all gates to get |ψ⟩
    2. Compute |λ⟩ = H|ψ⟩  (bra state)
    3. Backward pass: for each gate in reverse:
       - If gate has parameters, compute ∂⟨λ|U_k...U_1|0⟩/∂θ_k
       - Unapply gate from both |ψ⟩ and |λ⟩

    This requires O(1) extra memory beyond the state vector.

    Args:
        circuit:     Parameterised Circuit object.
        hamiltonian: Hamiltonian object with `expectation(wf)` method, or
                     a callable `(wf) -> float` that also supports
                     `_apply_to_state(wf)` for H|ψ⟩.
        params:      Parameter values (flat array).
        backend:     Simulation backend.

    Returns:
        Gradient array of same length as params.
    """
    params = np.asarray(params, dtype=float)
    bound_circuit = circuit.bind_parameters(params)

    # Forward pass: build |ψ⟩
    wf = Qubit(circuit.n_qubits, backend=backend)
    for op in bound_circuit.ops:
        _dispatch_op(wf, op)

    # |λ⟩ = H|ψ⟩ — apply Hamiltonian to get the bra state
    bra = Qubit(circuit.n_qubits, backend=backend)
    bra.amplitude = _apply_hamiltonian(hamiltonian, wf)

    # Backward pass
    gradient = np.zeros(len(params))
    param_idx = len(params) - 1

    for k in range(len(bound_circuit.ops) - 1, -1, -1):
        op = bound_circuit.ops[k]
        n_params_in_op = len(op.params)

        if n_params_in_op > 0:
            # This gate has parameters — compute gradient contribution
            for p_local in range(n_params_in_op - 1, -1, -1):
                # ∂⟨H⟩/∂θ = 2 Re(⟨λ|dU/dθ|ψ'⟩)
                # where |ψ'⟩ is the state before this gate
                # dU/dθ for rotation gates: dRX(θ)/dθ = -i/2 · X · RX(θ)
                # For RX, RY, RZ with standard parameter-shift:
                # ∂⟨H⟩/∂θ = Im(⟨λ|G U|ψ'⟩) where G is the generator
                grad_val = _compute_gate_gradient(
                    bra, wf, op, p_local, circuit.n_qubits
                )
                gradient[param_idx] = grad_val
                param_idx -= 1

        # Unapply this gate from both states
        adj_op = _adjoint_op(op)
        _dispatch_op(wf, adj_op)
        _dispatch_op(bra, adj_op)

    return gradient


def _apply_hamiltonian(hamiltonian, wf) -> np.ndarray:
    """Apply H|ψ⟩ and return the resulting amplitude array.

    Works with qforge.algo.hamiltonian.Hamiltonian which stores
    Pauli string terms.
    """
    amp = wf.amplitude.copy()
    n_qubits = len(wf.state[0])
    dim = len(amp)

    # Check if hamiltonian has terms/coeffs (Hamiltonian class)
    if hasattr(hamiltonian, 'coeffs') and hasattr(hamiltonian, 'terms'):
        result = np.zeros(dim, dtype=complex)
        for coeff, term in zip(hamiltonian.coeffs, hamiltonian.terms):
            # Apply Pauli string to amplitude
            term_amp = amp.copy()
            for pauli, qubit in term:
                term_amp = _apply_pauli_to_amp(term_amp, pauli, qubit, n_qubits)
            result += coeff * term_amp
        return result
    else:
        raise TypeError("Hamiltonian must have coeffs and terms attributes")


def _apply_pauli_to_amp(amp: np.ndarray, pauli: str, qubit: int,
                         n_qubits: int) -> np.ndarray:
    """Apply a single Pauli operator to an amplitude array."""
    dim = len(amp)
    result = np.zeros(dim, dtype=complex)
    bit = n_qubits - qubit - 1

    if pauli == 'I':
        return amp.copy()
    elif pauli == 'Z':
        for i in range(dim):
            if (i >> bit) & 1:
                result[i] = -amp[i]
            else:
                result[i] = amp[i]
    elif pauli == 'X':
        for i in range(dim):
            j = i ^ (1 << bit)
            result[j] = amp[i]
    elif pauli == 'Y':
        for i in range(dim):
            j = i ^ (1 << bit)
            if (i >> bit) & 1:
                result[j] = -1j * amp[i]   # Y|1> = -i|0>
            else:
                result[j] = 1j * amp[i]    # Y|0> = i|1>
    return result


def _compute_gate_gradient(bra, ket, op: GateOp, param_idx: int,
                            n_qubits: int) -> float:
    """Compute the gradient contribution of a parameterised gate.

    For standard rotation gates RX(θ), RY(θ), RZ(θ):
    dU/dθ = -i/2 · G · U where G is the Pauli generator.

    The gradient is: ∂⟨H⟩/∂θ = 2·Re(⟨bra| · (-i/2 · G) |ket⟩)
                             = -Im(⟨bra|G|ket⟩)
    """
    amp_ket = ket.amplitude.copy()
    amp_bra = bra.amplitude.copy()

    # Apply generator to ket
    qubit = op.qubits[0]
    gen_amp = None

    if op.name == 'RX':
        gen_amp = _apply_pauli_to_amp(amp_ket, 'X', qubit, n_qubits)
    elif op.name == 'RY':
        gen_amp = _apply_pauli_to_amp(amp_ket, 'Y', qubit, n_qubits)
    elif op.name == 'RZ':
        gen_amp = _apply_pauli_to_amp(amp_ket, 'Z', qubit, n_qubits)
    elif op.name == 'Phase':
        # Phase(φ) = diag(1, e^{iφ}), generator = |1><1|
        gen_amp = np.zeros_like(amp_ket)
        bit = n_qubits - qubit - 1
        for i in range(len(amp_ket)):
            if (i >> bit) & 1:
                gen_amp[i] = amp_ket[i]
    elif op.name in ('CRX', 'CRY', 'CRZ', 'CPhase', 'CP'):
        # Controlled rotation: generator acts only in control=1 subspace
        control, target = op.qubits[0], op.qubits[1]
        ctrl_bit = n_qubits - control - 1
        pauli_map = {'CRX': 'X', 'CRY': 'Y', 'CRZ': 'Z', 'CPhase': 'Z', 'CP': 'Z'}
        pauli = pauli_map.get(op.name, 'Z')
        # Apply Pauli to target, masked by control=1
        gen_amp = np.zeros_like(amp_ket)
        masked = amp_ket.copy()
        for i in range(len(masked)):
            if not ((i >> ctrl_bit) & 1):
                masked[i] = 0
        gen_amp = _apply_pauli_to_amp(masked, pauli, target, n_qubits)
    else:
        # Fallback: use numerical differentiation
        return _numerical_gate_gradient(bra, ket, op, param_idx, n_qubits)

    if gen_amp is not None:
        # gradient = Im(⟨bra|gen_amp⟩)
        overlap = np.dot(amp_bra.conj(), gen_amp)
        return overlap.imag

    return 0.0


def _numerical_gate_gradient(bra, ket, op: GateOp, param_idx: int,
                              n_qubits: int, eps: float = 1e-7) -> float:
    """Numerical gradient as fallback for unsupported gates."""
    # This shouldn't normally be needed but provides a safety net
    amp_bra = bra.amplitude
    amp_ket = ket.amplitude

    params_plus = list(op.params)
    params_minus = list(op.params)
    params_plus[param_idx] += eps
    params_minus[param_idx] -= eps

    # Create modified ops
    op_plus = GateOp(name=op.name, qubits=op.qubits, params=tuple(params_plus),
                     matrix=op.matrix, is_adjoint=op.is_adjoint, controls=op.controls)
    op_minus = GateOp(name=op.name, qubits=op.qubits, params=tuple(params_minus),
                      matrix=op.matrix, is_adjoint=op.is_adjoint, controls=op.controls)

    wf_plus = Qubit(n_qubits, backend='python')
    wf_plus.amplitude = ket.amplitude.copy()
    _dispatch_op(wf_plus, _adjoint_op(op))  # undo current gate
    _dispatch_op(wf_plus, op_plus)  # apply with +eps

    wf_minus = Qubit(n_qubits, backend='python')
    wf_minus.amplitude = ket.amplitude.copy()
    _dispatch_op(wf_minus, _adjoint_op(op))
    _dispatch_op(wf_minus, op_minus)

    e_plus = np.dot(amp_bra.conj(), wf_plus.amplitude).real
    e_minus = np.dot(amp_bra.conj(), wf_minus.amplitude).real

    return (e_plus - e_minus) / (2 * eps)
