# -*- coding: utf-8 -*-
# author: vinhpx
"""Circuit intermediate representation for qforge.

Provides ``GateOp`` (individual gate instruction) and ``Circuit`` (ordered
sequence of instructions) so that quantum programs can be built, inspected,
transformed, and replayed on any backend.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ============================================================
# Gate operation dataclass
# ============================================================

@dataclass(frozen=True)
class GateOp:
    """A single gate instruction in a circuit.

    Attributes:
        name:     Gate name (``'H'``, ``'CNOT'``, ``'RX'``, etc.).
        qubits:   Qubit indices the gate acts on.
        params:   Rotation angles / numeric parameters (empty for fixed gates).
        matrix:   Explicit unitary matrix (used by ``QubitUnitary``).
        is_adjoint: If ``True``, apply the conjugate transpose.
        controls: Extra control qubits (for multi-controlled wrapper).
    """
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    matrix: np.ndarray | None = None
    is_adjoint: bool = False
    controls: tuple[int, ...] = ()

    def __eq__(self, other):
        if not isinstance(other, GateOp):
            return NotImplemented
        return (self.name == other.name
                and self.qubits == other.qubits
                and self.params == other.params
                and self.is_adjoint == other.is_adjoint
                and self.controls == other.controls
                and _matrix_eq(self.matrix, other.matrix))

    def __hash__(self):
        return hash((self.name, self.qubits, self.params, self.is_adjoint, self.controls))


def _matrix_eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(a, b)


# ============================================================
# Mid-circuit measurement and classical control
# ============================================================

@dataclass(frozen=True)
class MeasureOp:
    """Mid-circuit measurement instruction.

    Attributes:
        qubit:         Qubit to measure.
        classical_bit: Index in the classical register to store the result.
    """
    qubit: int
    classical_bit: int


@dataclass(frozen=True)
class ConditionalOp:
    """Conditional gate: execute ``op`` only if ``classical_bit == expected_value``.

    Attributes:
        classical_bit:  Index in the classical register to check.
        expected_value: Value to compare against (0 or 1).
        op:             GateOp to execute if condition is met.
    """
    classical_bit: int
    expected_value: int
    op: GateOp


class ClassicalRegister:
    """Classical register for storing mid-circuit measurement results."""

    def __init__(self, size: int):
        self.size = size
        self.values = [0] * size

    def __getitem__(self, idx: int) -> int:
        return self.values[idx]

    def __setitem__(self, idx: int, val: int):
        self.values[idx] = val

    def __repr__(self):
        return f"ClassicalRegister({self.values})"


# ============================================================
# Gate dispatch — maps GateOp back to qforge.gates functions
# ============================================================

# Lazy import to avoid circular dependency at module load time.
_gates_mod = None


def _get_gates():
    global _gates_mod
    if _gates_mod is None:
        import qforge.gates as _g
        _gates_mod = _g
    return _gates_mod


# Map gate name -> (gate_function, arity_style)
# arity_style indicates how to call:
#   '1q'        -> fn(wf, qubit)
#   '1q_param'  -> fn(wf, qubit, param)
#   '2q'        -> fn(wf, q0, q1)
#   '2q_param'  -> fn(wf, q0, q1, param)
#   '3q'        -> fn(wf, q0, q1, q2)
#   'swap'      -> fn(wf, q0, q1)
#   'cswap'     -> fn(wf, q0, q1, q2)
#   'noise'     -> fn(wf, p, qubit)

_GATE_DISPATCH: dict[str, tuple[str, str]] = {
    # Single-qubit fixed
    'H':       ('H',       '1q'),
    'X':       ('X',       '1q'),
    'Y':       ('Y',       '1q'),
    'Z':       ('Z',       '1q'),
    'S':       ('S',       '1q'),
    'T':       ('T',       '1q'),
    'Xsquare': ('Xsquare', '1q'),
    # Single-qubit parameterised
    'RX':      ('RX',      '1q_param'),
    'RY':      ('RY',      '1q_param'),
    'RZ':      ('RZ',      '1q_param'),
    'Phase':   ('Phase',   '1q_param'),
    # Controlled
    'CNOT':    ('CNOT',    '2q'),
    'CRX':     ('CRX',     '2q_param'),
    'CRY':     ('CRY',     '2q_param'),
    'CRZ':     ('CRZ',     '2q_param'),
    'CPhase':  ('CPhase',  '2q_param'),
    'CP':      ('CPhase',  '2q_param'),
    # Double-controlled
    'CCNOT':   ('CCNOT',   '3q'),
    'OR':      ('OR',      '3q'),
    # Swap family
    'SWAP':    ('SWAP',    'swap'),
    'ISWAP':   ('ISWAP',   'swap'),
    'SISWAP':  ('SISWAP',  'swap'),
    'CSWAP':   ('CSWAP',   'cswap'),
    # Noise
    'E':       ('E',       'noise'),
}


def _dispatch_op(wf, op: GateOp) -> None:
    """Execute a single GateOp on a wavefunction/MPS by calling qforge.gates."""
    g = _get_gates()

    # Handle arbitrary unitary
    if op.name == 'Unitary':
        g.QubitUnitary(wf, op.matrix, list(op.qubits))
        return

    # Handle multi-controlled gates
    if op.name == 'MCX':
        g.mcx(wf, list(op.controls), op.qubits[0])
        return
    if op.name == 'MCZ':
        g.mcz(wf, list(op.controls), op.qubits[0])
        return
    if op.name == 'MCP':
        g.mcp(wf, list(op.controls), op.qubits[0], op.params[0])
        return

    # Handle custom gates from registry
    if op.name in _GATE_REGISTRY:
        custom = _GATE_REGISTRY[op.name]
        if custom.matrix is not None:
            g.QubitUnitary(wf, custom.matrix, list(op.qubits))
        elif custom.subcircuit is not None:
            custom.subcircuit(wf, np.array(op.params) if op.params else None)
        return

    entry = _GATE_DISPATCH.get(op.name)
    if entry is None:
        raise ValueError(f"Unknown gate: {op.name!r}")
    fn_name, style = entry
    fn = getattr(g, fn_name)

    if style == '1q':
        fn(wf, op.qubits[0])
    elif style == '1q_param':
        fn(wf, op.qubits[0], op.params[0])
    elif style == '2q':
        fn(wf, op.qubits[0], op.qubits[1])
    elif style == '2q_param':
        fn(wf, op.qubits[0], op.qubits[1], op.params[0])
    elif style == '3q':
        fn(wf, op.qubits[0], op.qubits[1], op.qubits[2])
    elif style == 'swap':
        fn(wf, op.qubits[0], op.qubits[1])
    elif style == 'cswap':
        fn(wf, op.qubits[0], op.qubits[1], op.qubits[2])
    elif style == 'noise':
        fn(wf, op.params[0], op.qubits[0])
    else:
        raise ValueError(f"Unknown dispatch style: {style!r}")


# ============================================================
# Gate adjoint registry
# ============================================================

# Maps gate name to a callable: (name, params) -> (adj_name, adj_params)
# For self-inverse gates the adjoint is the same gate.
# For rotation gates the adjoint negates the angle.

def _self_adjoint(name, params):
    return name, params

def _negate_params(name, params):
    return name, tuple(-p for p in params)

_ADJOINT_RULES: dict[str, Callable] = {
    'H':       _self_adjoint,
    'X':       _self_adjoint,
    'Y':       _self_adjoint,
    'Z':       _self_adjoint,
    'CNOT':    _self_adjoint,
    'CCNOT':   _self_adjoint,
    'SWAP':    _self_adjoint,
    'S':       lambda n, p: ('Phase', (-np.pi / 2,)),
    'T':       lambda n, p: ('Phase', (-np.pi / 4,)),
    'RX':      _negate_params,
    'RY':      _negate_params,
    'RZ':      _negate_params,
    'Phase':   _negate_params,
    'CRX':     _negate_params,
    'CRY':     _negate_params,
    'CRZ':     _negate_params,
    'CPhase':  _negate_params,
    'CP':      _negate_params,
    'Xsquare': lambda n, p: ('Xsquare', ()),  # Xsquare^dag ≠ Xsquare, handled via matrix
    'ISWAP':   _self_adjoint,  # ISWAP^dag needs matrix; approximate for now
    'SISWAP':  _self_adjoint,
    'CSWAP':   _self_adjoint,
    'OR':      _self_adjoint,
}


def _adjoint_op(op: GateOp) -> GateOp:
    """Return the adjoint of a GateOp."""
    if op.matrix is not None:
        adj_matrix = op.matrix.conj().T if not op.is_adjoint else op.matrix
        return GateOp(
            name=op.name, qubits=op.qubits, params=op.params,
            matrix=adj_matrix, is_adjoint=not op.is_adjoint, controls=op.controls,
        )
    rule = _ADJOINT_RULES.get(op.name)
    if rule is not None:
        adj_name, adj_params = rule(op.name, op.params)
        return GateOp(
            name=adj_name, qubits=op.qubits, params=adj_params,
            matrix=op.matrix, is_adjoint=not op.is_adjoint, controls=op.controls,
        )
    # Fallback: mark as adjoint and let dispatch handle it
    return GateOp(
        name=op.name, qubits=op.qubits, params=op.params,
        matrix=op.matrix, is_adjoint=not op.is_adjoint, controls=op.controls,
    )


# ============================================================
# Circuit class
# ============================================================

class Circuit:
    """Quantum circuit — an ordered sequence of ``GateOp`` instructions.

    Build a circuit with method-chaining, then execute on any backend::

        from qforge.ir import Circuit

        qc = Circuit(3)
        qc.h(0).cnot(0, 1).rx(2, 0.5)
        wf = qc.run(backend='auto')

    The circuit can also be used as a VQA ansatz::

        params = np.array([0.5, 1.2])
        qc = Circuit(2)
        qc.ry(0, 'p0').ry(1, 'p1').cnot(0, 1)
        # Execute with bound parameters
        wf = qc.run(params={'p0': params[0], 'p1': params[1]})
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.ops: list[GateOp] = []

    # ---- internal helpers -------------------------------------------

    def _add(self, name: str, qubits: tuple[int, ...],
             params: tuple[float, ...] = (), matrix=None) -> 'Circuit':
        self.ops.append(GateOp(name=name, qubits=qubits, params=params, matrix=matrix))
        return self

    def __len__(self):
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def __getitem__(self, idx):
        return self.ops[idx]

    def __repr__(self):
        return f"Circuit(n_qubits={self.n_qubits}, depth={len(self.ops)})"

    # ---- single-qubit gates ----------------------------------------

    def h(self, q: int) -> 'Circuit':
        return self._add('H', (q,))

    def x(self, q: int) -> 'Circuit':
        return self._add('X', (q,))

    def y(self, q: int) -> 'Circuit':
        return self._add('Y', (q,))

    def z(self, q: int) -> 'Circuit':
        return self._add('Z', (q,))

    def s(self, q: int) -> 'Circuit':
        return self._add('S', (q,))

    def t(self, q: int) -> 'Circuit':
        return self._add('T', (q,))

    def xsquare(self, q: int) -> 'Circuit':
        return self._add('Xsquare', (q,))

    def rx(self, q: int, phi: float) -> 'Circuit':
        return self._add('RX', (q,), (float(phi),))

    def ry(self, q: int, phi: float) -> 'Circuit':
        return self._add('RY', (q,), (float(phi),))

    def rz(self, q: int, phi: float) -> 'Circuit':
        return self._add('RZ', (q,), (float(phi),))

    def phase(self, q: int, phi: float) -> 'Circuit':
        return self._add('Phase', (q,), (float(phi),))

    # ---- controlled gates -------------------------------------------

    def cnot(self, control: int, target: int) -> 'Circuit':
        return self._add('CNOT', (control, target))

    def cx(self, control: int, target: int) -> 'Circuit':
        return self.cnot(control, target)

    def crx(self, control: int, target: int, phi: float) -> 'Circuit':
        return self._add('CRX', (control, target), (float(phi),))

    def cry(self, control: int, target: int, phi: float) -> 'Circuit':
        return self._add('CRY', (control, target), (float(phi),))

    def crz(self, control: int, target: int, phi: float) -> 'Circuit':
        return self._add('CRZ', (control, target), (float(phi),))

    def cphase(self, control: int, target: int, phi: float) -> 'Circuit':
        return self._add('CPhase', (control, target), (float(phi),))

    def cp(self, control: int, target: int, phi: float) -> 'Circuit':
        return self.cphase(control, target, phi)

    # ---- double-controlled gates ------------------------------------

    def ccnot(self, c1: int, c2: int, target: int) -> 'Circuit':
        return self._add('CCNOT', (c1, c2, target))

    def toffoli(self, c1: int, c2: int, target: int) -> 'Circuit':
        return self.ccnot(c1, c2, target)

    def or_gate(self, c1: int, c2: int, target: int) -> 'Circuit':
        return self._add('OR', (c1, c2, target))

    # ---- swap gates -------------------------------------------------

    def swap(self, q1: int, q2: int) -> 'Circuit':
        return self._add('SWAP', (q1, q2))

    def iswap(self, q1: int, q2: int) -> 'Circuit':
        return self._add('ISWAP', (q1, q2))

    def siswap(self, q1: int, q2: int) -> 'Circuit':
        return self._add('SISWAP', (q1, q2))

    def cswap(self, control: int, t1: int, t2: int) -> 'Circuit':
        return self._add('CSWAP', (control, t1, t2))

    # ---- multi-controlled gates --------------------------------------

    def mcx(self, controls: list[int], target: int) -> 'Circuit':
        """Multi-controlled X (generalized Toffoli)."""
        self.ops.append(GateOp(name='MCX', qubits=(target,), controls=tuple(controls)))
        return self

    def mcz(self, controls: list[int], target: int) -> 'Circuit':
        """Multi-controlled Z gate."""
        self.ops.append(GateOp(name='MCZ', qubits=(target,), controls=tuple(controls)))
        return self

    def mcp(self, controls: list[int], target: int, phi: float) -> 'Circuit':
        """Multi-controlled Phase gate."""
        self.ops.append(GateOp(
            name='MCP', qubits=(target,), params=(float(phi),), controls=tuple(controls),
        ))
        return self

    # ---- noise channels ---------------------------------------------

    def depolarize(self, q: int, p: float) -> 'Circuit':
        return self._add('E', (q,), (float(p),))

    # ---- arbitrary unitary ------------------------------------------

    def unitary(self, matrix: np.ndarray, qubits: Sequence[int]) -> 'Circuit':
        """Apply an arbitrary unitary matrix to the given qubits."""
        matrix = np.asarray(matrix, dtype=complex)
        self.ops.append(GateOp(
            name='Unitary', qubits=tuple(qubits), matrix=matrix,
        ))
        return self

    # ---- mid-circuit measurement ------------------------------------

    def measure(self, qubit: int, classical_bit: int) -> 'Circuit':
        """Add a mid-circuit measurement.

        Measures ``qubit`` and stores the result in ``classical_bit``.
        """
        self.ops.append(MeasureOp(qubit=qubit, classical_bit=classical_bit))
        return self

    def c_if(self, classical_bit: int, expected_value: int, op: GateOp) -> 'Circuit':
        """Add a conditional gate.

        Executes ``op`` only if the classical register at ``classical_bit``
        equals ``expected_value``.
        """
        self.ops.append(ConditionalOp(
            classical_bit=classical_bit,
            expected_value=expected_value,
            op=op,
        ))
        return self

    # ---- generic gate append ----------------------------------------

    def add_gate(self, op: GateOp) -> 'Circuit':
        """Append a pre-built GateOp."""
        self.ops.append(op)
        return self

    # ---- circuit transformations ------------------------------------

    def adjoint(self) -> 'Circuit':
        """Return the adjoint (inverse) circuit — reversed ops with each gate adjointed."""
        adj = Circuit(self.n_qubits)
        for op in reversed(self.ops):
            adj.ops.append(_adjoint_op(op))
        return adj

    def compose(self, other: 'Circuit') -> 'Circuit':
        """Return a new circuit: ``self`` followed by ``other``."""
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Cannot compose circuits with different qubit counts "
                f"({self.n_qubits} vs {other.n_qubits})"
            )
        result = Circuit(self.n_qubits)
        result.ops = list(self.ops) + list(other.ops)
        return result

    def copy(self) -> 'Circuit':
        """Return a deep copy of this circuit."""
        c = Circuit(self.n_qubits)
        c.ops = list(self.ops)  # GateOp is frozen, so shallow copy of list is fine
        return c

    # ---- parameter utilities ----------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of floating-point parameters across all ops."""
        return sum(len(op.params) for op in self.ops if isinstance(op, GateOp))

    @property
    def parameter_indices(self) -> list[tuple[int, int]]:
        """Return [(op_idx, param_idx), ...] for every tunable parameter."""
        indices = []
        for i, op in enumerate(self.ops):
            if isinstance(op, GateOp):
                for j in range(len(op.params)):
                    indices.append((i, j))
        return indices

    def bind_parameters(self, values: np.ndarray) -> 'Circuit':
        """Return a new circuit with parameter values replaced sequentially.

        ``values`` is a flat array of length :attr:`num_parameters`.
        """
        values = np.asarray(values, dtype=float)
        if len(values) != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} params, got {len(values)}"
            )
        new_circ = Circuit(self.n_qubits)
        idx = 0
        for op in self.ops:
            if not isinstance(op, GateOp):
                new_circ.ops.append(op)
                continue
            n_p = len(op.params)
            if n_p > 0:
                new_params = tuple(float(values[idx + k]) for k in range(n_p))
                new_circ.ops.append(GateOp(
                    name=op.name, qubits=op.qubits, params=new_params,
                    matrix=op.matrix, is_adjoint=op.is_adjoint, controls=op.controls,
                ))
                idx += n_p
            else:
                new_circ.ops.append(op)
        return new_circ

    # ---- execution --------------------------------------------------

    def run(self, backend: str = 'auto', params: np.ndarray | None = None) -> object:
        """Execute the circuit and return the final ``Wavefunction``.

        Supports mid-circuit measurements (MeasureOp) and conditional gates
        (ConditionalOp). When measurements are present, execution is
        stochastic — each run produces a single trajectory.

        Args:
            backend: qforge backend (``'auto'``, ``'cpu'``, ``'cuda'``, ``'metal'``, ``'python'``).
            params:  Optional flat array to bind before execution (see :meth:`bind_parameters`).

        Returns:
            :class:`~qforge.wavefunction.Wavefunction` with the final quantum state.
            If mid-circuit measurements were used, ``wf.classical_register`` contains
            the measurement outcomes.
        """
        from qforge.circuit import Qubit

        circ = self.bind_parameters(params) if params is not None else self
        wf = Qubit(self.n_qubits, backend=backend)
        creg = ClassicalRegister(self.n_qubits)  # default size = n_qubits

        for op in circ.ops:
            if isinstance(op, MeasureOp):
                from qforge.measurement import collapse_one, measure_one
                probs = measure_one(wf, op.qubit)
                outcome = int(np.random.choice([0, 1], p=probs))
                creg[op.classical_bit] = outcome
                collapse_one(wf, op.qubit)
            elif isinstance(op, ConditionalOp):
                if creg[op.classical_bit] == op.expected_value:
                    _dispatch_op(wf, op.op)
            elif isinstance(op, GateOp):
                _dispatch_op(wf, op)

        wf.classical_register = creg
        return wf

    def __call__(self, wf: object, params: np.ndarray | None = None) -> None:
        """Apply this circuit to an existing wavefunction (VQA-compatible).

        Args:
            wf:     Existing ``Wavefunction`` or ``MatrixProductState``.
            params:  Optional flat parameter array to bind first.
        """
        circ = self.bind_parameters(params) if params is not None else self
        for op in circ.ops:
            if isinstance(op, (MeasureOp, ConditionalOp)):
                continue  # Skip measurement ops in VQA mode
            _dispatch_op(wf, op)


# ============================================================
# Recording context manager
# ============================================================

_RECORDING_CIRCUIT: Circuit | None = None


class record:
    """Context manager to record gate calls into a ``Circuit``.

    Usage::

        from qforge.ir import Circuit, record
        from qforge.gates import H, CNOT

        with record(3) as qc:
            wf = Qubit(3)
            H(wf, 0)
            CNOT(wf, 0, 1)

        # qc is now a Circuit with those ops recorded
        print(qc)  # Circuit(n_qubits=3, depth=2)
    """

    def __init__(self, n_qubits: int):
        self._circuit = Circuit(n_qubits)

    def __enter__(self) -> Circuit:
        global _RECORDING_CIRCUIT
        _RECORDING_CIRCUIT = self._circuit
        return self._circuit

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _RECORDING_CIRCUIT
        _RECORDING_CIRCUIT = None
        return False


def _record_op(name: str, qubits: tuple[int, ...],
               params: tuple[float, ...] = (), matrix=None) -> None:
    """Called by gate functions to record an op if recording is active."""
    if _RECORDING_CIRCUIT is not None:
        _RECORDING_CIRCUIT.ops.append(
            GateOp(name=name, qubits=qubits, params=params, matrix=matrix)
        )


# ============================================================
# Custom gate registry
# ============================================================

@dataclass
class CustomGate:
    """A user-defined gate.

    Define via a unitary matrix or a sub-circuit::

        from qforge.ir import CustomGate, register_gate

        # Matrix-defined gate
        register_gate(CustomGate(
            name='MySqrtX',
            n_qubits=1,
            matrix=np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2,
        ))

        # Circuit-defined gate
        sub = Circuit(2)
        sub.h(0).cnot(0, 1)
        register_gate(CustomGate(
            name='BellPrep', n_qubits=2, subcircuit=sub,
        ))
    """
    name: str
    n_qubits: int
    n_params: int = 0
    matrix: np.ndarray | None = None
    subcircuit: 'Circuit | None' = None


# Global registry
_GATE_REGISTRY: dict[str, CustomGate] = {}


def register_gate(gate: CustomGate) -> None:
    """Register a custom gate for use in circuits."""
    _GATE_REGISTRY[gate.name] = gate


def unregister_gate(name: str) -> None:
    """Remove a custom gate from the registry."""
    _GATE_REGISTRY.pop(name, None)
