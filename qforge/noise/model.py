# -*- coding: utf-8 -*-
# author: vinhpx
"""Noise model for qforge circuits.

A ``NoiseModel`` maps (gate_name, qubits) pairs to noise channels that are
applied after each matching gate operation during circuit execution.
"""
from __future__ import annotations

from typing import Sequence

from qforge.noise.channels import QuantumChannel, ReadoutError


class NoiseModel:
    """Noise model that associates quantum channels with gate operations.

    Errors can be registered for all qubits or specific qubit subsets.
    During simulation the noise model is queried after each gate to
    determine which channels to apply.

    Example::

        from qforge.noise import NoiseModel, Depolarizing

        model = NoiseModel()
        model.add_all_qubit_quantum_error(Depolarizing(0.01), ['H', 'X', 'Y', 'Z'])
        model.add_quantum_error(Depolarizing(0.05), ['CNOT'], [0, 1])
    """

    def __init__(self) -> None:
        # (gate_name, None)        -> all-qubit errors
        # (gate_name, tuple(qubits)) -> qubit-specific errors
        self._errors: dict[tuple[str, tuple[int, ...] | None], list[QuantumChannel]] = {}
        # Readout errors per qubit
        self._readout_errors: dict[int, ReadoutError] = {}

    def add_all_qubit_quantum_error(
        self, channel: QuantumChannel, gate_names: list[str]
    ) -> None:
        """Register a noise channel to be applied after every occurrence of
        the listed gates, regardless of which qubits they act on.

        Args:
            channel:    A ``QuantumChannel`` instance.
            gate_names: List of gate names (e.g. ``['H', 'X']``).
        """
        for name in gate_names:
            key = (name, None)
            self._errors.setdefault(key, []).append(channel)

    def add_quantum_error(
        self,
        channel: QuantumChannel,
        gate_names: list[str],
        qubits: list[int],
    ) -> None:
        """Register a noise channel for specific qubits.

        Args:
            channel:    A ``QuantumChannel`` instance.
            gate_names: List of gate names.
            qubits:     List of qubit indices this error applies to.
        """
        qubit_key = tuple(sorted(qubits))
        for name in gate_names:
            key = (name, qubit_key)
            self._errors.setdefault(key, []).append(channel)

    def add_readout_error(self, readout_channel: ReadoutError, qubits: list[int]) -> None:
        """Register a readout error for specific qubits.

        Args:
            readout_channel: A ``ReadoutError`` instance.
            qubits:          Qubit indices to apply the readout error to.
        """
        for q in qubits:
            self._readout_errors[q] = readout_channel

    def get_errors(self, gate_name: str, qubits: Sequence[int]) -> list[QuantumChannel]:
        """Look up all noise channels for a given gate application.

        Returns all-qubit errors for the gate name plus any qubit-specific
        errors that match the exact qubit set.

        Args:
            gate_name: Name of the gate (e.g. ``'H'``, ``'CNOT'``).
            qubits:    Qubit indices the gate acts on.

        Returns:
            List of ``QuantumChannel`` instances to apply after the gate.
        """
        errors: list[QuantumChannel] = []
        # All-qubit errors
        key_all = (gate_name, None)
        if key_all in self._errors:
            errors.extend(self._errors[key_all])
        # Qubit-specific errors
        qubit_key = tuple(sorted(qubits))
        key_specific = (gate_name, qubit_key)
        if key_specific in self._errors:
            errors.extend(self._errors[key_specific])
        return errors

    def get_readout_error(self, qubit: int) -> ReadoutError | None:
        """Return the readout error for a qubit, or None."""
        return self._readout_errors.get(qubit)

    @property
    def is_empty(self) -> bool:
        """True if no errors have been registered."""
        return len(self._errors) == 0 and len(self._readout_errors) == 0

    def __repr__(self) -> str:
        n_gate = sum(len(v) for v in self._errors.values())
        n_readout = len(self._readout_errors)
        return f"NoiseModel(gate_errors={n_gate}, readout_errors={n_readout})"
