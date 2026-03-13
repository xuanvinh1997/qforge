# -*- coding: utf-8 -*-
"""Tests for adjoint differentiation."""
import numpy as np
import pytest
from qforge.ir import Circuit
from qforge.algo.hamiltonian import Hamiltonian
from qforge.algo.adjoint_diff import adjoint_differentiation
from qforge.algo.gradient import parameter_shift


def _cost_fn_from_circuit(circuit, hamiltonian, backend='python'):
    """Create a cost function from a circuit and hamiltonian."""
    def cost_fn(params):
        wf = circuit.run(backend=backend, params=params)
        return hamiltonian.expectation(wf)
    return cost_fn


class TestAdjointDiff:
    def test_single_rx_gradient(self):
        """Gradient of <Z> for RX(theta)|0> should be -sin(theta)."""
        qc = Circuit(1)
        qc.rx(0, 0.0)
        H = Hamiltonian([1.0], [[('Z', 0)]])
        params = np.array([0.7])
        grad_adj = adjoint_differentiation(qc, H, params)
        # Compare with parameter-shift
        cost_fn = _cost_fn_from_circuit(qc, H)
        grad_ps = parameter_shift(cost_fn, params)
        np.testing.assert_allclose(grad_adj, grad_ps, atol=1e-5)

    def test_single_ry_gradient(self):
        """Gradient of <Z> for RY(theta)|0>."""
        qc = Circuit(1)
        qc.ry(0, 0.0)
        H = Hamiltonian([1.0], [[('Z', 0)]])
        params = np.array([1.2])
        grad_adj = adjoint_differentiation(qc, H, params)
        cost_fn = _cost_fn_from_circuit(qc, H)
        grad_ps = parameter_shift(cost_fn, params)
        np.testing.assert_allclose(grad_adj, grad_ps, atol=1e-5)

    def test_two_param_gradient(self):
        """Gradient of a 2-param circuit matches parameter-shift."""
        qc = Circuit(2)
        qc.rx(0, 0.0).ry(1, 0.0).cnot(0, 1)
        H = Hamiltonian([1.0], [[('Z', 0)]])
        params = np.array([0.5, 0.8])
        grad_adj = adjoint_differentiation(qc, H, params)
        cost_fn = _cost_fn_from_circuit(qc, H)
        grad_ps = parameter_shift(cost_fn, params)
        np.testing.assert_allclose(grad_adj, grad_ps, atol=1e-5)

    def test_rz_gradient(self):
        """RZ gradient matches parameter-shift."""
        qc = Circuit(1)
        qc.h(0).rz(0, 0.0)
        H = Hamiltonian([1.0], [[('X', 0)]])
        params = np.array([0.9])
        grad_adj = adjoint_differentiation(qc, H, params)
        cost_fn = _cost_fn_from_circuit(qc, H)
        grad_ps = parameter_shift(cost_fn, params)
        np.testing.assert_allclose(grad_adj, grad_ps, atol=1e-5)

    def test_zero_gradient_at_minimum(self):
        """At the minimum of <Z> for RY, gradient should be ~0."""
        qc = Circuit(1)
        qc.ry(0, 0.0)
        H = Hamiltonian([1.0], [[('Z', 0)]])
        # <Z> = cos(theta), minimum at theta=pi, gradient=0
        params = np.array([np.pi])
        grad_adj = adjoint_differentiation(qc, H, params)
        np.testing.assert_allclose(grad_adj, [0.0], atol=1e-5)

    def test_gradient_shape(self):
        """Output shape matches number of parameters."""
        qc = Circuit(2)
        qc.rx(0, 0.0).ry(1, 0.0).rz(0, 0.0)
        H = Hamiltonian([1.0], [[('Z', 0)]])
        params = np.array([0.1, 0.2, 0.3])
        grad = adjoint_differentiation(qc, H, params)
        assert grad.shape == (3,)
