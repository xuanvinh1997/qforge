# -*- coding: utf-8 -*-
# author: vinhpx
"""Interfaces to external ML frameworks (JAX, PyTorch, TensorFlow)."""
from __future__ import annotations

__all__: list[str] = []

try:
    from qforge.interfaces.jax import qnode_jax
    __all__.append("qnode_jax")
except ImportError:
    pass

try:
    from qforge.interfaces.torch import QNodeFunction
    __all__.append("QNodeFunction")
except ImportError:
    pass

try:
    from qforge.interfaces.tensorflow import qnode_tf
    __all__.append("qnode_tf")
except ImportError:
    pass
