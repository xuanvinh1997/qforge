# -*- coding: utf-8 -*-
"""Tests for the Clifford/Stabilizer simulator."""
import numpy as np
import pytest
from qforge.stabilizer import StabilizerState


class TestStabilizerBasic:
    def test_init(self):
        s = StabilizerState(2)
        assert s.n == 2
        assert s.tableau.shape == (4, 5)

    def test_invalid_n_qubits(self):
        with pytest.raises(ValueError):
            StabilizerState(0)

    def test_repr(self):
        s = StabilizerState(3)
        assert 'n_qubits=3' in repr(s)


class TestStabilizerGates:
    def test_x_gate(self):
        """X on |0> should give deterministic |1>."""
        s = StabilizerState(1)
        s.x(0)
        np.random.seed(42)
        result = s.measure(0)
        assert result == 1

    def test_z_gate_on_zero(self):
        """Z on |0> = |0> (eigenstate)."""
        s = StabilizerState(1)
        s.z(0)
        np.random.seed(42)
        result = s.measure(0)
        assert result == 0

    def test_h_creates_superposition(self):
        """H on |0> gives equal probability of 0 and 1."""
        np.random.seed(42)
        counts = {0: 0, 1: 0}
        for _ in range(200):
            s = StabilizerState(1)
            s.h(0)
            counts[s.measure(0)] += 1
        # Both outcomes should appear
        assert counts[0] > 30
        assert counts[1] > 30

    def test_cnot_bell_state(self):
        """H + CNOT creates Bell state: always correlated measurements."""
        np.random.seed(42)
        for _ in range(50):
            s = StabilizerState(2)
            s.h(0).cnot(0, 1)
            r0 = s.measure(0)
            r1 = s.measure(1)
            assert r0 == r1  # Bell state: always correlated

    def test_s_gate(self):
        """S gate shouldn't crash and should be valid."""
        s = StabilizerState(1)
        s.s(0)
        result = s.measure(0)
        assert result == 0  # S on |0> = |0>

    def test_y_gate(self):
        """Y on |0> should give |1>."""
        s = StabilizerState(1)
        s.y(0)
        result = s.measure(0)
        assert result == 1

    def test_multiple_operations(self):
        """Chain of operations shouldn't crash."""
        s = StabilizerState(3)
        s.h(0).cnot(0, 1).cnot(1, 2)
        # GHZ state: all same
        r0 = s.measure(0)
        r1 = s.measure(1)
        r2 = s.measure(2)
        assert r0 == r1 == r2


class TestStabilizerMeasurement:
    def test_deterministic_zero(self):
        """Measuring |0> always gives 0."""
        s = StabilizerState(1)
        for _ in range(10):
            s2 = StabilizerState(1)
            assert s2.measure(0) == 0

    def test_deterministic_one(self):
        """Measuring X|0> = |1> always gives 1."""
        for _ in range(10):
            s = StabilizerState(1)
            s.x(0)
            assert s.measure(0) == 1

    def test_post_measurement_deterministic(self):
        """After measurement, re-measuring gives same result."""
        np.random.seed(42)
        s = StabilizerState(1)
        s.h(0)
        r1 = s.measure(0)
        r2 = s.measure(0)
        assert r1 == r2
