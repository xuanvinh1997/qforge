"""qforge — Large-Scale Quantum Simulation Framework.

Wavefunction-based quantum circuit simulator with C++/CUDA/Metal acceleration.
Includes MPS (Matrix Product States) and DMRG simulation backends.
"""

__version__ = "3.0.0"
__all__ = [
    "circuit", "gates", "measurement", "data", "encodings",
    "kernels", "wavefunction", "gate_walk", "algo",
    "mps", "dmrg", "ir", "qudit_gates",
    "Circuit", "GateOp", "MeasureOp", "ConditionalOp",
    "Parameter", "ParameterVector",
    "StabilizerState",
    "DensityMatrix",
    "draw_circuit",
    "circuit_to_json", "circuit_from_json",
    "circuit_to_binary", "circuit_from_binary",
    "transpiler", "noise", "mitigation", "chem", "qasm", "interfaces",
    "_HAS_CPP", "_HAS_CUDA", "_HAS_METAL", "_HAS_MPS", "_HAS_DISTRIBUTED",
    "set_backend", "get_backend", "backend_info",
]

from qforge.ir import Circuit, GateOp, MeasureOp, ConditionalOp
from qforge.parameters import Parameter, ParameterVector
from qforge.stabilizer import StabilizerState
from qforge.visualization import draw_circuit
from qforge.serialization import (
    circuit_to_json, circuit_from_json,
    circuit_to_binary, circuit_from_binary,
)

try:
    from qforge.density_matrix import DensityMatrix
except ImportError:
    pass

try:
    from qforge._qforge_core import StateVector as _StateVector
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

try:
    from qforge._qforge_cuda import CudaStateVector as _CudaStateVector
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False

try:
    from qforge._qforge_metal import MetalStateVector as _MetalStateVector
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False

try:
    from qforge._qforge_mps import MPS as _MPS_core
    _HAS_MPS = True
except ImportError:
    _HAS_MPS = False

try:
    from qforge._qforge_distributed import DistributedStateVector as _DistSV
    _HAS_DISTRIBUTED = True
except ImportError:
    _HAS_DISTRIBUTED = False


# ================================================================
# Centralized backend selection
# ================================================================

_VALID_BACKENDS = ('auto', 'metal', 'cuda', 'cpu', 'distributed', 'python')
_default_backend = 'auto'


def _resolve_backend(backend: str) -> str:
    """Resolve 'auto' to the best available backend. Returns a concrete name."""
    if backend != 'auto':
        return backend
    if _HAS_CUDA:
        return 'cuda'
    if _HAS_METAL:
        return 'metal'
    if _HAS_CPP:
        return 'cpu'
    return 'python'


def set_backend(backend: str) -> None:
    """Set the global default backend for all new circuits.

    Args:
        backend: One of ``'auto'``, ``'metal'``, ``'cuda'``, ``'cpu'``, ``'python'``.

    Raises:
        ValueError: If the backend name is invalid or not available.

    Example::

        import qforge
        qforge.set_backend('cpu')       # force C++ CPU
        qforge.set_backend('metal')     # force Metal GPU
        qforge.set_backend('auto')      # best available
    """
    global _default_backend
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Choose from: {', '.join(_VALID_BACKENDS)}")
    if backend == 'cuda' and not _HAS_CUDA:
        raise ValueError("CUDA backend requested but _qforge_cuda is not available")
    if backend == 'metal' and not _HAS_METAL:
        raise ValueError("Metal backend requested but _qforge_metal is not available")
    if backend == 'cpu' and not _HAS_CPP:
        raise ValueError("CPU C++ backend requested but _qforge_core is not available")
    if backend == 'distributed' and not _HAS_DISTRIBUTED:
        raise ValueError("Distributed backend requested but _qforge_distributed is not available")
    _default_backend = backend


def get_backend() -> str:
    """Return the current global default backend name.

    Returns:
        str: The current default (e.g. ``'auto'``, ``'metal'``, ``'cpu'``).
    """
    return _default_backend


def backend_info() -> dict:
    """Return a dict describing available backends and the current default.

    Example::

        >>> import qforge
        >>> qforge.backend_info()
        {'default': 'auto', 'resolved': 'metal', 'available': {...}}
    """
    available = {
        'metal':       _HAS_METAL,
        'cuda':        _HAS_CUDA,
        'cpu':         _HAS_CPP,
        'mps_cpp':     _HAS_MPS,
        'distributed': _HAS_DISTRIBUTED,
        'python':      True,
    }
    return {
        'default':   _default_backend,
        'resolved':  _resolve_backend(_default_backend),
        'available': available,
    }
