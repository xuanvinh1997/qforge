# -*- coding: utf-8 -*-
# author: vinhpx
"""Probabilistic Error Cancellation (PEC) for error mitigation."""
from __future__ import annotations

import numpy as np
from qforge.ir import Circuit, GateOp


class DepolarizingNoiseModel:
    """Simple depolarizing noise model for PEC.

    Each gate experiences depolarizing noise with probability ``p``,
    meaning the ideal gate channel is followed by a depolarizing channel:
        ρ → (1 - p) ρ + (p/3)(X ρ X + Y ρ Y + Z ρ Z)  (single-qubit)

    The quasi-probability decomposition inverts this channel.

    Args:
        p: Depolarizing noise strength (0 to 1).
    """

    def __init__(self, p: float = 0.01):
        if not 0 <= p <= 1:
            raise ValueError("Noise probability p must be in [0, 1]")
        self.p = p

    def quasi_probabilities(self, op: GateOp) -> list[tuple[float, GateOp | None]]:
        """Return quasi-probability decomposition for inverting noise after ``op``.

        For depolarizing noise with strength p on a single-qubit gate:
            E_inv = (1 + 3p/(4-4p)) * I - (p/(4-4p)) * (X + Y + Z)

        Returns a list of (quasi_prob, correction_op) pairs.
        correction_op is None for the identity correction.
        """
        if len(op.qubits) == 1:
            return self._single_qubit_decomposition(op)
        # For multi-qubit gates, use a simplified model
        return self._single_qubit_decomposition(op)

    def _single_qubit_decomposition(self, op: GateOp) -> list[tuple[float, GateOp | None]]:
        """Quasi-probability decomposition for single-qubit depolarizing channel."""
        p = self.p
        if p < 1e-15:
            return [(1.0, None)]

        # Inverse of depolarizing channel:
        # η_inv = (1 - p/2)/(1 - p) * I  - (p/2)/(1-p) * {X, Y, Z} correction
        # Simplified: identity weight and Pauli correction weights
        q = p / (4.0 - 4.0 * p)  # Pauli correction coefficient
        identity_weight = 1.0 + 3.0 * q
        pauli_weight = -q

        qubit = op.qubits[0] if op.qubits else 0
        decomposition = [
            (identity_weight, None),
            (pauli_weight, GateOp(name='X', qubits=(qubit,))),
            (pauli_weight, GateOp(name='Y', qubits=(qubit,))),
            (pauli_weight, GateOp(name='Z', qubits=(qubit,))),
        ]
        return decomposition

    @property
    def one_norm(self) -> float:
        """One-norm (sampling overhead) of the quasi-probability decomposition."""
        p = self.p
        if p < 1e-15:
            return 1.0
        q = p / (4.0 - 4.0 * p)
        return abs(1.0 + 3.0 * q) + 3.0 * abs(q)


class PEC:
    """Probabilistic Error Cancellation.

    Uses a noise model to decompose noisy operations into quasi-probability
    sums and sample correction circuits.

    Args:
        noise_model: A noise model that provides quasi_probabilities for each gate.
                     Defaults to DepolarizingNoiseModel(0.01).
    """

    def __init__(self, noise_model: DepolarizingNoiseModel | None = None):
        self.noise_model = noise_model or DepolarizingNoiseModel(0.01)

    def mitigate(
        self,
        circuit: Circuit,
        executor,
        n_samples: int = 1000,
    ) -> float:
        """Mitigate errors in a circuit using PEC.

        Args:
            circuit:   The ideal circuit.
            executor:  Callable ``(circuit) -> float`` that runs the circuit
                       and returns an expectation value.
            n_samples: Number of Monte Carlo samples.

        Returns:
            Mitigated expectation value.
        """
        gate_ops = [op for op in circuit.ops if isinstance(op, GateOp)]
        if not gate_ops:
            return executor(circuit)

        # Precompute quasi-probability decompositions for each gate
        decompositions = []
        for op in gate_ops:
            decomp = self.noise_model.quasi_probabilities(op)
            decompositions.append(decomp)

        # Compute the one-norm (sampling overhead)
        one_norm = 1.0
        for decomp in decompositions:
            gamma = sum(abs(w) for w, _ in decomp)
            one_norm *= gamma

        results = []
        for _ in range(n_samples):
            # Sample a correction circuit
            sign = 1.0
            sampled_circuit = Circuit(circuit.n_qubits)

            gate_idx = 0
            for op in circuit.ops:
                if not isinstance(op, GateOp):
                    sampled_circuit.ops.append(op)
                    continue

                decomp = decompositions[gate_idx]
                # Sample from the normalized distribution
                weights = np.array([abs(w) for w, _ in decomp])
                gamma = weights.sum()
                probs = weights / gamma

                idx = np.random.choice(len(decomp), p=probs)
                w, correction = decomp[idx]
                sign *= np.sign(w) * gamma

                # Add original gate
                sampled_circuit.ops.append(op)
                # Add correction if not identity
                if correction is not None:
                    sampled_circuit.ops.append(correction)

                gate_idx += 1

            exp_val = executor(sampled_circuit)
            results.append(sign * exp_val)

        return float(np.mean(results))


def probabilistic_error_cancellation(
    ideal_circuit: Circuit,
    noise_model: DepolarizingNoiseModel | None = None,
    executor=None,
    n_samples: int = 1000,
) -> float:
    """Functional interface for Probabilistic Error Cancellation.

    Args:
        ideal_circuit: The ideal (noiseless) circuit.
        noise_model:   Noise model for quasi-probability decomposition.
        executor:      Callable ``(circuit) -> float``.
        n_samples:     Number of Monte Carlo samples.

    Returns:
        Mitigated expectation value.
    """
    if executor is None:
        raise ValueError("executor must be provided")
    pec = PEC(noise_model=noise_model)
    return pec.mitigate(ideal_circuit, executor, n_samples)
