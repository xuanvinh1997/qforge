"""qforge — Large-Scale Quantum Simulation Framework.

Wavefunction-based quantum circuit simulator with C++/CUDA/Metal acceleration.
Includes MPS (Matrix Product States) and DMRG simulation backends.
"""

__version__ = "3.0.0"
__all__ = [
    "circuit", "gates", "measurement", "data", "encodings",
    "kernels", "wavefunction", "gate_walk", "algo",
    "mps", "dmrg", "ir",
    "Circuit", "GateOp", "MeasureOp", "ConditionalOp",
    "Parameter", "ParameterVector",
    "StabilizerState",
    "DensityMatrix",
    "draw_circuit",
    "circuit_to_json", "circuit_from_json",
    "circuit_to_binary", "circuit_from_binary",
    "transpiler", "noise", "mitigation", "chem", "qasm", "interfaces",
    "_HAS_CPP", "_HAS_CUDA", "_HAS_METAL", "_HAS_MPS",
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
