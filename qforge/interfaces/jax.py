# -*- coding: utf-8 -*-
# author: vinhpx
"""JAX-differentiable quantum function interface."""
from __future__ import annotations

import numpy as np


def qnode_jax(circuit, hamiltonian, backend='python'):
    """Create a JAX-differentiable quantum function.

    Returns a function ``params -> float`` that can be differentiated
    with ``jax.grad``.

    Args:
        circuit:     A qforge Circuit (template with parameters).
        hamiltonian: A qforge Hamiltonian.
        backend:     Simulation backend (default 'python').

    Returns:
        A JAX-compatible callable.

    Raises:
        ImportError: If JAX is not installed.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for qnode_jax. "
            "Install it with: pip install jax jaxlib"
        )

    from qforge.algo.gradient import parameter_shift

    @jax.custom_vjp
    def quantum_fn(params):
        params_np = np.asarray(params, dtype=float)
        wf = circuit.run(backend=backend, params=params_np)
        return float(hamiltonian.expectation(wf))

    def quantum_fn_fwd(params):
        val = quantum_fn(params)
        return val, params

    def quantum_fn_bwd(params, g):
        params_np = np.asarray(params, dtype=float)

        def cost_fn(p):
            wf = circuit.run(backend=backend, params=p)
            return hamiltonian.expectation(wf)

        grad = parameter_shift(cost_fn, params_np)
        return (jnp.array(grad) * g,)

    quantum_fn.defvjp(quantum_fn_fwd, quantum_fn_bwd)
    return quantum_fn
