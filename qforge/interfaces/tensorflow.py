# -*- coding: utf-8 -*-
# author: vinhpx
"""TensorFlow custom-gradient quantum function interface."""
from __future__ import annotations

import numpy as np


def qnode_tf(circuit, hamiltonian, backend='auto'):
    """Create a TensorFlow-differentiable quantum function.

    Returns a function ``params -> tf.Tensor`` that supports
    ``tf.GradientTape``.

    Args:
        circuit:     A qforge Circuit (template with parameters).
        hamiltonian: A qforge Hamiltonian.
        backend:     Simulation backend (default ``'auto'``).

    Returns:
        A TensorFlow-compatible callable.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for qnode_tf. "
            "Install it with: pip install tensorflow"
        )

    from qforge.algo.gradient import parameter_shift

    @tf.custom_gradient
    def quantum_fn(params):
        params_np = np.asarray(params, dtype=float)
        wf = circuit.run(backend=backend, params=params_np)
        val = float(hamiltonian.expectation(wf))

        def grad_fn(upstream):
            def cost_fn(p):
                wf2 = circuit.run(backend=backend, params=p)
                return hamiltonian.expectation(wf2)

            grad = parameter_shift(cost_fn, params_np)
            return tf.constant(grad, dtype=params.dtype) * upstream

        return tf.constant(val, dtype=params.dtype), grad_fn

    return quantum_fn
