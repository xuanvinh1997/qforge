# -*- coding: utf-8 -*-
# author: vinhpx
"""Symbolic parameters for parameterised quantum circuits.

Provides :class:`Parameter` and :class:`ParameterVector` so that circuits
can be built with named, unbound parameters and later resolved via
:meth:`Circuit.bind_parameters`.
"""
from __future__ import annotations

import numpy as np


class Parameter:
    """A named symbolic parameter.

    Args:
        name:  Human-readable identifier.
        value: Optional default value. ``None`` means unbound.
    """

    def __init__(self, name: str, value: float | None = None):
        self.name = name
        self.value = value

    # ---- arithmetic (returns float when evaluated) ----

    def __float__(self):
        if self.value is None:
            raise ValueError(f"Parameter {self.name!r} is unbound — "
                             f"call bind_parameters() first")
        return float(self.value)

    def __repr__(self):
        if self.value is None:
            return f"Parameter({self.name!r})"
        return f"Parameter({self.name!r}, value={self.value})"

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name and self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(('Parameter', self.name))

    def is_bound(self) -> bool:
        return self.value is not None

    def bind(self, value: float) -> 'Parameter':
        """Return a new Parameter with the given value."""
        return Parameter(self.name, float(value))


class ParameterVector:
    """A named collection of :class:`Parameter` objects.

    Usage::

        pv = ParameterVector('theta', 4)
        # pv[0] is Parameter('theta_0'), pv[1] is Parameter('theta_1'), ...
    """

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self._params = [Parameter(f"{name}_{i}") for i in range(size)]

    def __getitem__(self, idx: int) -> Parameter:
        return self._params[idx]

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self._params)

    def __repr__(self):
        return f"ParameterVector({self.name!r}, size={self.size})"

    def bind(self, values) -> list[Parameter]:
        """Return a list of bound Parameters."""
        values = np.asarray(values, dtype=float)
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values, got {len(values)}")
        return [Parameter(p.name, float(v)) for p, v in zip(self._params, values)]

    @property
    def params(self) -> list[Parameter]:
        return list(self._params)
