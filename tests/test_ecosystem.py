# -*- coding: utf-8 -*-
# author: vinhpx
"""Tests for ecosystem features: optimizers, hessian, stabilizer, serialization, visualization."""
from __future__ import annotations

import numpy as np
import pytest

from qforge.ir import Circuit, GateOp
from qforge.algo.optimizers import SPSA, LBFGS
from qforge.algo.hessian import parameter_shift_hessian
from qforge.stabilizer import StabilizerState
from qforge.serialization import circuit_to_json, circuit_from_json
from qforge.visualization import draw_circuit


# ============================================================
# SPSA optimizer tests
# ============================================================

class TestSPSA:
    """Tests for SPSA optimizer."""

    def test_step_reduces_cost_simple(self):
        """SPSA step should reduce cost on a simple quadratic function."""
        def cost_fn(params):
            return np.sum(params ** 2)

        params = np.array([1.0, 2.0, -1.5])
        optimizer = SPSA(lr=0.05, perturbation=0.1)

        # Run several steps using estimate_and_step
        for _ in range(50):
            params = optimizer.estimate_and_step(params, cost_fn)

        # Cost should have decreased significantly
        assert cost_fn(params) < 1.0

    def test_step_api_compatibility(self):
        """SPSA.step should work with pre-computed gradients like other optimizers."""
        optimizer = SPSA(lr=0.1)
        params = np.array([1.0, 2.0])
        grad = np.array([0.5, -0.3])
        new_params = optimizer.step(params, grad)
        expected = params - 0.1 * grad
        np.testing.assert_allclose(new_params, expected)


# ============================================================
# L-BFGS optimizer tests
# ============================================================

class TestLBFGS:
    """Tests for L-BFGS optimizer."""

    def test_minimizes_quadratic(self):
        """L-BFGS should minimize a simple quadratic to near zero."""
        def cost_fn(params):
            return np.sum((params - np.array([1.0, -2.0])) ** 2)

        optimizer = LBFGS()
        params0 = np.array([5.0, 5.0])
        opt_params, final_cost = optimizer.minimize(cost_fn, params0, maxiter=100)

        np.testing.assert_allclose(opt_params, [1.0, -2.0], atol=1e-4)
        assert final_cost < 1e-8

    def test_returns_tuple(self):
        """minimize returns (params, cost) tuple."""
        optimizer = LBFGS()
        result = optimizer.minimize(lambda p: np.sum(p ** 2), np.array([1.0]), maxiter=10)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================
# Hessian tests
# ============================================================

class TestHessian:
    """Tests for parameter_shift_hessian."""

    def test_quadratic_matches_numerical(self):
        """Hessian of a quadratic should match analytical second derivatives."""
        # f(x, y) = 3*x^2 + 2*x*y + y^2
        # H = [[6, 2], [2, 2]]
        # But parameter-shift uses sin-based formulas, so test with trig functions

        def cost_fn(params):
            return np.cos(params[0]) + np.cos(params[1]) + 0.5 * np.cos(params[0] + params[1])

        params = np.array([0.3, 0.7])

        # Numerical Hessian
        eps = 1e-5
        n = len(params)
        numerical_hessian = np.zeros((n, n))
        f0 = cost_fn(params)
        for i in range(n):
            for j in range(n):
                pp = params.copy(); pp[i] += eps; pp[j] += eps
                pm = params.copy(); pm[i] += eps; pm[j] -= eps
                mp = params.copy(); mp[i] -= eps; mp[j] += eps
                mm = params.copy(); mm[i] -= eps; mm[j] -= eps
                numerical_hessian[i, j] = (cost_fn(pp) - cost_fn(pm) - cost_fn(mp) + cost_fn(mm)) / (4 * eps * eps)

        # Parameter-shift Hessian
        ps_hessian = parameter_shift_hessian(cost_fn, params)

        np.testing.assert_allclose(ps_hessian, numerical_hessian, atol=0.1)

    def test_symmetric(self):
        """Hessian should be symmetric."""
        def cost_fn(params):
            return np.sin(params[0]) * np.cos(params[1])

        params = np.array([0.5, 1.0])
        hessian = parameter_shift_hessian(cost_fn, params)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)


# ============================================================
# Stabilizer tests
# ============================================================

class TestStabilizerState:
    """Tests for StabilizerState."""

    def test_bell_state_correlations(self):
        """H on qubit 0 then CNOT(0,1) should produce Bell-like correlations."""
        # Measure many times and check correlations
        n_trials = 200
        same_count = 0

        for _ in range(n_trials):
            state = StabilizerState(2)
            state.h(0)
            state.cnot(0, 1)
            m0 = state.measure(0)
            m1 = state.measure(1)
            if m0 == m1:
                same_count += 1

        # Bell state: measurements should be perfectly correlated
        assert same_count > 0.9 * n_trials

    def test_x_gate_flips(self):
        """X gate should flip |0> to |1>."""
        n_trials = 50
        for _ in range(n_trials):
            state = StabilizerState(1)
            state.x(0)
            result = state.measure(0)
            assert result == 1

    def test_h_gate_creates_superposition(self):
        """H gate should create equal superposition."""
        n_trials = 200
        ones = 0
        for _ in range(n_trials):
            state = StabilizerState(1)
            state.h(0)
            ones += state.measure(0)

        # Should be roughly 50/50
        ratio = ones / n_trials
        assert 0.3 < ratio < 0.7

    def test_z_gate_on_zero(self):
        """Z gate on |0> should leave it as |0>."""
        for _ in range(20):
            state = StabilizerState(1)
            state.z(0)
            result = state.measure(0)
            assert result == 0

    def test_s_gate(self):
        """S gate on |0> should leave it as |0>."""
        for _ in range(20):
            state = StabilizerState(1)
            state.s(0)
            result = state.measure(0)
            assert result == 0

    def test_y_gate_flips(self):
        """Y gate should flip |0> to |1>."""
        for _ in range(20):
            state = StabilizerState(1)
            state.y(0)
            result = state.measure(0)
            assert result == 1

    def test_cnot_leaves_zero_state(self):
        """CNOT on |00> should leave it as |00>."""
        for _ in range(20):
            state = StabilizerState(2)
            state.cnot(0, 1)
            m0 = state.measure(0)
            m1 = state.measure(1)
            assert m0 == 0 and m1 == 0

    def test_probabilities_returns_dict(self):
        """probabilities() returns a dict with string keys and float values."""
        state = StabilizerState(1)
        state.h(0)
        probs = state.probabilities()
        assert isinstance(probs, dict)
        assert len(probs) > 0
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.1


# ============================================================
# Serialization tests
# ============================================================

class TestSerialization:
    """Tests for circuit JSON serialization."""

    def test_round_trip_simple(self):
        """Circuit -> JSON -> Circuit should preserve structure."""
        qc = Circuit(3)
        qc.h(0).cnot(0, 1).rx(2, 0.5)

        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)

        assert qc2.n_qubits == qc.n_qubits
        assert len(qc2.ops) == len(qc.ops)
        for op1, op2 in zip(qc.ops, qc2.ops):
            assert op1.name == op2.name
            assert op1.qubits == op2.qubits
            assert op1.params == pytest.approx(op2.params)

    def test_round_trip_with_matrix(self):
        """Circuit with unitary matrix gate survives round-trip."""
        qc = Circuit(1)
        mat = np.array([[0, 1], [1, 0]], dtype=complex)
        qc.unitary(mat, [0])

        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)

        assert len(qc2.ops) == 1
        np.testing.assert_allclose(qc2.ops[0].matrix, mat)

    def test_round_trip_empty_circuit(self):
        """Empty circuit round-trips correctly."""
        qc = Circuit(5)
        json_str = circuit_to_json(qc)
        qc2 = circuit_from_json(json_str)
        assert qc2.n_qubits == 5
        assert len(qc2.ops) == 0

    def test_json_is_valid_string(self):
        """circuit_to_json returns a valid JSON string."""
        import json
        qc = Circuit(2)
        qc.h(0).cnot(0, 1)
        json_str = circuit_to_json(qc)
        data = json.loads(json_str)
        assert 'n_qubits' in data
        assert 'ops' in data


# ============================================================
# Visualization tests
# ============================================================

class TestVisualization:
    """Tests for circuit drawing."""

    def test_text_output_nonempty(self):
        """draw_circuit returns a non-empty string."""
        qc = Circuit(3)
        qc.h(0).cnot(0, 1).rx(2, 0.5)
        result = draw_circuit(qc, output='text')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_output_has_qubit_labels(self):
        """Text output includes qubit labels."""
        qc = Circuit(2)
        qc.h(0).x(1)
        result = draw_circuit(qc, output='text')
        assert 'q0' in result
        assert 'q1' in result

    def test_text_output_has_gate_names(self):
        """Text output includes gate names."""
        qc = Circuit(1)
        qc.h(0)
        result = draw_circuit(qc, output='text')
        assert 'H' in result

    def test_empty_circuit(self):
        """Drawing an empty circuit should not crash."""
        qc = Circuit(2)
        result = draw_circuit(qc, output='text')
        assert isinstance(result, str)
        assert 'q0' in result

    def test_unknown_output_raises(self):
        """Unknown output format raises ValueError."""
        qc = Circuit(1)
        with pytest.raises(ValueError, match="Unknown output format"):
            draw_circuit(qc, output='unknown')

    def test_cnot_rendering(self):
        """CNOT gate rendering includes control and target symbols."""
        qc = Circuit(2)
        qc.cnot(0, 1)
        result = draw_circuit(qc, output='text')
        assert '*' in result
        assert '(+)' in result

    def test_swap_rendering(self):
        """SWAP gate rendering includes x symbols."""
        qc = Circuit(2)
        qc.swap(0, 1)
        result = draw_circuit(qc, output='text')
        assert 'x' in result
