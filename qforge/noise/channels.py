# -*- coding: utf-8 -*-
# author: vinhpx
"""Quantum noise channels for qforge.

Each channel is defined by its Kraus operators K_k such that
the channel acts as  rho -> sum_k K_k rho K_k^dagger  and
sum_k K_k^dagger K_k = I  (trace preservation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


# Pauli matrices (2x2)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class QuantumChannel(ABC):
    """Abstract base class for a quantum noise channel.

    Subclasses must implement the ``kraus_ops`` property which returns a
    list of Kraus operator matrices.
    """

    @property
    @abstractmethod
    def kraus_ops(self) -> list[np.ndarray]:
        """Return the list of Kraus operators for this channel."""
        ...

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        """Check that sum_k K_k^dagger K_k = I."""
        ops = self.kraus_ops
        dim = ops[0].shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        for K in ops:
            total += K.conj().T @ K
        return np.allclose(total, np.eye(dim, dtype=complex), atol=atol)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_kraus={len(self.kraus_ops)})"


class BitFlip(QuantumChannel):
    """Bit-flip channel.

    K0 = sqrt(1-p) * I,  K1 = sqrt(p) * X

    Args:
        p: Probability of a bit flip (0 <= p <= 1).
    """

    def __init__(self, p: float) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self.p) * _I,
            np.sqrt(self.p) * _X,
        ]


class PhaseFlip(QuantumChannel):
    """Phase-flip (dephasing) channel.

    K0 = sqrt(1-p) * I,  K1 = sqrt(p) * Z

    Args:
        p: Probability of a phase flip (0 <= p <= 1).
    """

    def __init__(self, p: float) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self.p) * _I,
            np.sqrt(self.p) * _Z,
        ]


class Depolarizing(QuantumChannel):
    """Single-qubit depolarizing channel.

    K0 = sqrt(1 - 3p/4) * I
    K1 = sqrt(p/4) * X
    K2 = sqrt(p/4) * Y
    K3 = sqrt(p/4) * Z

    For p = 4/3 the channel maps any state to the maximally mixed state.

    Args:
        p: Depolarizing parameter (0 <= p <= 4/3).
    """

    def __init__(self, p: float) -> None:
        if not 0 <= p <= 4 / 3:
            raise ValueError(f"p must be in [0, 4/3], got {p}")
        self.p = p

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - 3 * self.p / 4) * _I,
            np.sqrt(self.p / 4) * _X,
            np.sqrt(self.p / 4) * _Y,
            np.sqrt(self.p / 4) * _Z,
        ]


class AmplitudeDamping(QuantumChannel):
    """Amplitude damping channel (energy relaxation / T1 decay).

    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, sqrt(gamma)], [0, 0]]

    Args:
        gamma: Damping parameter (0 <= gamma <= 1).
    """

    def __init__(self, gamma: float) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]], dtype=complex)
        return [K0, K1]


class PhaseDamping(QuantumChannel):
    """Phase damping channel (T2 dephasing without energy loss).

    K0 = [[1, 0], [0, sqrt(1-gamma)]]
    K1 = [[0, 0], [0, sqrt(gamma)]]

    Args:
        gamma: Damping parameter (0 <= gamma <= 1).
    """

    def __init__(self, gamma: float) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(self.gamma)]], dtype=complex)
        return [K0, K1]


class ThermalRelaxation(QuantumChannel):
    """Thermal relaxation channel parameterised by T1, T2, and gate time.

    Models combined amplitude damping and dephasing from physical T1/T2 times.

    If T2 <= T1:
        Composed of amplitude damping (gamma = 1 - exp(-t/T1))
        followed by phase damping to account for remaining T2 dephasing.

    Args:
        T1:        Longitudinal relaxation time.
        T2:        Transverse relaxation time (must satisfy T2 <= 2*T1).
        gate_time: Duration of the gate.
    """

    def __init__(self, T1: float, T2: float, gate_time: float) -> None:
        if T1 <= 0 or T2 <= 0 or gate_time < 0:
            raise ValueError("T1, T2 must be positive; gate_time non-negative")
        if T2 > 2 * T1:
            raise ValueError(f"T2 ({T2}) must be <= 2*T1 ({2*T1})")
        self.T1 = T1
        self.T2 = T2
        self.gate_time = gate_time

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        t = self.gate_time
        if t == 0:
            return [_I.copy()]

        p_reset = 1 - np.exp(-t / self.T1)
        # Amplitude damping component
        ad_K0 = np.array([[1, 0], [0, np.sqrt(1 - p_reset)]], dtype=complex)
        ad_K1 = np.array([[0, np.sqrt(p_reset)], [0, 0]], dtype=complex)

        # Additional dephasing beyond T1
        # Rate: 1/T2 - 1/(2*T1)
        if self.T2 < 2 * self.T1:
            rate_extra = 1.0 / self.T2 - 1.0 / (2 * self.T1)
            p_phase = 1 - np.exp(-rate_extra * t)
        else:
            p_phase = 0.0

        if p_phase < 1e-15:
            return [ad_K0, ad_K1]

        # Compose: phase damping after amplitude damping
        # PD: K0_pd = [[1,0],[0,sqrt(1-p)]], K1_pd = [[0,0],[0,sqrt(p)]]
        pd_K0 = np.array([[1, 0], [0, np.sqrt(1 - p_phase)]], dtype=complex)
        pd_K1 = np.array([[0, 0], [0, np.sqrt(p_phase)]], dtype=complex)

        # Compose: K_ij = pd_Ki @ ad_Kj
        result = []
        for Ki in [pd_K0, pd_K1]:
            for Kj in [ad_K0, ad_K1]:
                K = Ki @ Kj
                if np.linalg.norm(K) > 1e-15:
                    result.append(K)
        return result


class ReadoutError(QuantumChannel):
    """Classical readout error modelled as a confusion matrix.

    This is NOT a quantum channel in the Kraus sense. It is applied
    to measurement probabilities as a classical post-processing step:
    p_measured = confusion_matrix @ p_ideal.

    Args:
        p0_given_1: P(measure 0 | true state 1).
        p1_given_0: P(measure 1 | true state 0).
    """

    def __init__(self, p0_given_1: float, p1_given_0: float) -> None:
        if not (0 <= p0_given_1 <= 1 and 0 <= p1_given_0 <= 1):
            raise ValueError("Readout error probabilities must be in [0, 1]")
        self.p0_given_1 = p0_given_1
        self.p1_given_0 = p1_given_0
        self.confusion_matrix = np.array([
            [1 - p1_given_0, p0_given_1],
            [p1_given_0, 1 - p0_given_1],
        ], dtype=float)

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        """ReadoutError is classical; Kraus ops are not applicable.

        Returns the confusion matrix wrapped in a list for compatibility.
        """
        return [self.confusion_matrix.astype(complex)]

    def apply_to_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Apply readout error to a probability vector.

        For a single qubit, probs = [p0, p1].
        For multi-qubit, extend by tensor product of confusion matrices.
        """
        if len(probs) == 2:
            return self.confusion_matrix @ probs
        # Multi-qubit: not implemented in this base version
        return probs


class KrausChannel(QuantumChannel):
    """Arbitrary quantum channel defined by explicit Kraus operators.

    Args:
        kraus_operators: List of Kraus operator matrices.
    """

    def __init__(self, kraus_operators: Sequence[np.ndarray]) -> None:
        self._kraus_ops = [np.asarray(K, dtype=complex) for K in kraus_operators]

    @property
    def kraus_ops(self) -> list[np.ndarray]:
        return list(self._kraus_ops)
