# -*- coding: utf-8 -*-
# author: vinhpx
"""PyTorch autograd-compatible quantum function interface."""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class QNodeFunction:
    """PyTorch autograd-compatible quantum function.

    Wraps a quantum circuit + Hamiltonian into a differentiable function
    that can be used with PyTorch's autograd.

    Args:
        circuit:     A qforge Circuit (template with parameters).
        hamiltonian: A qforge Hamiltonian.
        backend:     Simulation backend (default ``'auto'``).
    """

    def __init__(self, circuit, hamiltonian, backend='auto'):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for QNodeFunction. "
                "Install it with: pip install torch"
            )
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.backend = backend

    def __call__(self, params):
        """Evaluate the quantum function with PyTorch autograd support.

        Args:
            params: torch.Tensor of parameters.

        Returns:
            torch.Tensor scalar with the expectation value.
        """
        return _QNodeAutograd.apply(
            params, self.circuit, self.hamiltonian, self.backend
        )


if _HAS_TORCH:
    class _QNodeAutograd(torch.autograd.Function):
        """Internal PyTorch autograd Function for quantum circuits."""

        @staticmethod
        def forward(ctx, params, circuit, hamiltonian, backend):
            params_np = params.detach().cpu().numpy().astype(float)
            ctx.save_for_backward(params)
            ctx.circuit = circuit
            ctx.hamiltonian = hamiltonian
            ctx.backend = backend

            wf = circuit.run(backend=backend, params=params_np)
            val = hamiltonian.expectation(wf)
            return torch.tensor(val, dtype=params.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (params,) = ctx.saved_tensors
            params_np = params.detach().cpu().numpy().astype(float)

            from qforge.algo.gradient import parameter_shift

            def cost_fn(p):
                wf = ctx.circuit.run(backend=ctx.backend, params=p)
                return ctx.hamiltonian.expectation(wf)

            grad = parameter_shift(cost_fn, params_np)
            grad_tensor = torch.tensor(grad, dtype=params.dtype)
            return grad_output * grad_tensor, None, None, None
