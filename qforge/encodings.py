# -*- coding: utf-8 -*-
# author: vinhpx
"""Quantum encoding strategies."""
from qforge.wavefunction import Wavefunction
from qforge.circuit import Qubit
from qforge.gates import *
import numpy as np
import math

def amplitude_encode(sample):
    """Encode a classical vector directly into quantum amplitudes.

    The sample is normalized and mapped to the amplitude vector of a
    ``ceil(log2(len(sample)))``-qubit state.

    Args:
        sample: Real-valued array of length up to 2^n.

    Returns:
        Wavefunction: Initialized quantum state.
    """
    qubit_num = int(math.ceil(np.log2(len(sample))))
    circuit_initial = Qubit(qubit_num)
    circuit_initial.amplitude[0:len(sample)] = sample/np.sqrt(np.sum(sample**2))
    return circuit_initial

def qubit_encode(sample):
    """Encode each feature into one qubit via rotation angles.

    Each feature ``x_i`` maps to ``cos(x_i/2)|0> - sin(x_i/2)|1>``
    on qubit *i*, combined as a tensor product.

    Args:
        sample: Array of rotation angles (radians), one per qubit.

    Returns:
        Wavefunction: Encoded quantum state with ``len(sample)`` qubits.
    """
    circuit_initial = Qubit(len(sample))
    ampli_vec = np.array([np.cos(sample[0]/2), -np.sin(sample[0]/2)])
    for i in range(1, len(sample)):
        ampli_vec = np.kron(ampli_vec, np.array([np.cos(sample[i]/2), -np.sin(sample[i]/2)]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def dense_encode(sample):
    """Dense angle encoding using pairs of features per qubit.

    Uses ``len(sample)/2`` qubits, encoding two features per qubit
    via combined RY and RZ rotations in the amplitude.

    Args:
        sample: Array of length ``2*n_qubits``.

    Returns:
        Wavefunction: Encoded quantum state.
    """
    qubit_num = int(len(sample)/2)
    circuit_initial = Qubit(qubit_num)
    ampli_vec = np.array([np.cos(sample[0+qubit_num]/2)*np.cos(sample[0]/2) - 1j*np.sin(sample[0+qubit_num]/2)*np.sin(sample[0]/2),
                          -np.sin(sample[0+qubit_num]/2)*np.cos(sample[0]/2) - 1j*np.cos(sample[0+qubit_num]/2)*np.sin(sample[0]/2)])
    for i in range(1, qubit_num):
        ampli_vec = np.kron(ampli_vec, np.array([np.cos(sample[i+qubit_num]/2)*np.cos(sample[i]/2) - 1j*np.sin(sample[i+qubit_num]/2)*np.sin(sample[i]/2),
                                      -np.sin(sample[i+qubit_num]/2)*np.cos(sample[i]/2) - 1j*np.cos(sample[i+qubit_num]/2)*np.sin(sample[i]/2)]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def unit_encode(sample):
    '''Encode each feature into one qubit's amplitude by using the square root function'''
    circuit_initial = Qubit(len(sample))
    ampli_vec = np.array([np.sqrt(sample[0]), np.sqrt(1-sample[0])])
    for i in range(1, len(sample)):
        ampli_vec = np.kron(ampli_vec, np.array([np.sqrt(sample[i]), np.sqrt(1-sample[i])]))
    circuit_initial.amplitude = ampli_vec
    return circuit_initial

def entangle(circuit, entanglement):
    """Apply CNOT entanglement layer to a circuit.

    Args:
        circuit: Wavefunction to entangle.
        entanglement: Connectivity pattern — ``"linear"``, ``"circular"``, or ``"full"``.

    Returns:
        Wavefunction: The same circuit object (modified in-place).
    """
    circuit_layer = circuit
    qubit_num = int(math.ceil(np.log2(len(circuit_layer.state))))
    if entanglement == "linear":
        for i in range(qubit_num - 1):
            CNOT(circuit_layer, i, i + 1)
    elif entanglement == "circular":
        for i in range(qubit_num - 1):
            CNOT(circuit_layer, i, i + 1)
        CNOT(circuit_layer, qubit_num - 1, 0)
    elif entanglement == "full":
        for i in range(qubit_num):
            for j in range(i + 1, qubit_num):
                CNOT(circuit_layer, i, j)
    return circuit_layer


def yz_cx_encode(sample, params=None, n_layers=2):
    """Variational encoding with RY-RZ rotations and CNOT entanglement.

    Each layer applies ``RY(param + 2*x)`` and ``RZ(param + 2*x)`` per qubit,
    followed by a staggered CNOT ladder.

    Args:
        sample: Feature vector (one feature per qubit).
        params: Trainable parameters. If ``None``, initialized randomly.
            Expected length: ``n_qubits * 2 * n_layers``.
        n_layers: Number of variational layers (default 2).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    expected_params = n_qubit * 2 * n_layers    
    if params is None:
        params = np.random.uniform(0, 2*np.pi, size=expected_params)
    else:
        params = np.array(params)
        if len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters "
                f"({n_qubit} qubits × 2 gates × {n_layers} layers), "
                f"got {len(params)}"
            )
    circuit_initial = Qubit(n_qubit)
    param_idx = 0
    for layer in range(n_layers):
        for q in range(n_qubit):
            RY(circuit_initial, q, phi=params[param_idx] + 2.0 * sample[q])
            param_idx += 1
            RZ(circuit_initial, q, phi=params[param_idx] + 2.0 * sample[q])
            param_idx += 1
        offset = layer % 2
        for i in range(offset, n_qubit - 1, 2):
            CNOT(circuit_initial, i, i + 1)
        if offset == 1 and n_qubit > 2:
            CNOT(circuit_initial, n_qubit - 1, 0)

    return circuit_initial

def high_dim_encode(sample):
    """High-dimensional feature map with H-RZ-RY-RZ layers and SISWAP entanglement.

    Args:
        sample: Feature vector (one feature per qubit).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for q in range(n_qubit):
        H(circuit_initial, q)
        i0 = (n_qubit - q + 0) % n_qubit  
        i1 = (n_qubit - q + 1) % n_qubit  
        i2 = (n_qubit - q + 2) % n_qubit  
        RZ(circuit_initial, q, phi=sample[i0])
        RY(circuit_initial, q, phi=sample[i1])
        RZ(circuit_initial, q, phi=sample[i2])
    for q in range(0, n_qubit - 1, 2):
        SISWAP(circuit_initial, q, q + 1)
    for q in range(1, n_qubit - 1, 2):
        SISWAP(circuit_initial, q, q + 1)
    for q in range(n_qubit):
        i0 = (n_qubit - q + 0) % n_qubit
        i1 = (n_qubit - q + 1) % n_qubit
        i2 = (n_qubit - q + 2) % n_qubit
        RZ(circuit_initial, q, phi=sample[i0])
        RY(circuit_initial, q, phi=sample[i1])
        RZ(circuit_initial, q, phi=sample[i2])

    return circuit_initial

def hzy_cz_encode(sample, params=None, n_layers=2, closed=True):
    """Variational encoding with H, RZ(feature), RY(param), CRZ(param) layers.

    Args:
        sample: Feature vector (one feature per qubit).
        params: Trainable parameters. If ``None``, initialized randomly.
        n_layers: Number of variational layers (default 2).
        closed: If ``True``, CRZ entanglement wraps around (circular topology).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    feature_vector = np.array([sample[i % n_qubit] for i in range(n_qubit)])
    params_per_layer = n_qubit
    if n_qubit > 2:
        params_per_layer += n_qubit if closed else (n_qubit - 1)
    total_params = params_per_layer * n_layers
    if params is None:
        params = np.random.rand(total_params)
    else:
        params = np.asarray(params, dtype=float)
        if len(params) < total_params:
            raise ValueError(f"Need {total_params} params, got {len(params)}")
    circuit_initial = Qubit(n_qubit)
    param_idx = 0
    for q in range(n_qubit):
        H(circuit_initial, q)
    for layer in range(n_layers):
        for q in range(n_qubit):
            RZ(circuit_initial, q, feature_vector[q])
        for q in range(n_qubit):
            RY(circuit_initial, q, params[param_idx])
            param_idx += 1    
        if n_qubit > 2:
            istop = n_qubit if closed else (n_qubit - 1)
            for i in range(istop):
                CRZ(circuit_initial, i, (i + 1) % n_qubit, phi=params[param_idx])
                param_idx += 1
    return circuit_initial

def chebyshev_encode(sample, params=None, n_layers=2):
    """Chebyshev polynomial encoding with trainable frequency scaling.

    Encodes features via ``RX(param * arccos(x))`` rotations with CRZ entanglement.

    Args:
        sample: Feature vector with values in ``[-1, 1]``.
        params: Trainable parameters. If ``None``, initialized randomly.
        n_layers: Number of encoding layers (default 2).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    crz_params_per_layer = n_qubit  
    total_num_params = (
        n_qubit + n_layers * (n_qubit + crz_params_per_layer) + n_qubit)
    if params is None:
        params = np.random.uniform(0, np.pi, size=total_num_params)
    else: 
        params = np.asarray(params, dtype=float)
    idx = 0
    for q in range(n_qubit):
        RY(circuit_initial, q, params[idx])
        idx += 1
    for layer in range(n_layers):
        for q in range(n_qubit):
            angle = params[idx] * np.arccos(np.clip(sample[q], -1, 1))
            RX(circuit_initial, q, angle)
            idx += 1
        for q in range(0, n_qubit - 1, 2):
            CRZ(circuit_initial, q, q + 1, phi=params[idx])
            idx += 1
        for q in range(1, n_qubit - 1, 2):
            CRZ(circuit_initial, q, q + 1, phi=params[idx])
            idx += 1
        CRZ(circuit_initial, n_qubit - 1, 0, phi=params[idx])
        idx += 1

    for q in range(n_qubit):
        RY(circuit_initial, q, params[idx])
        idx += 1
    
    return circuit_initial

def param_z_feature_map_encode(sample, params=None, n_layers=2):
    """Parameterized Z-feature map with trainable scaling.

    Each layer applies ``H`` then ``Phase(param * x)`` per qubit, followed
    by a linear CNOT chain.

    Args:
        sample: Feature vector (one feature per qubit).
        params: Trainable parameters. If ``None``, initialized randomly.
        n_layers: Number of layers (default 2).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    num_params = n_layers * n_qubit
    if params is None:
        params = np.random.uniform(0, np.pi, size=num_params)
    else: 
        params = np.asarray(params, dtype=float)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubit):
            H(circuit_initial, q)
            Phase(circuit_initial, q, phi=params[idx] * sample[q])
            idx += 1
        for q in range(0, n_qubit - 1, 1):
            CNOT(circuit_initial, q, q + 1)
    
    return circuit_initial

def separable_rx_encode(sample):
    """Separable encoding using two RX rotations per qubit (no entanglement).

    Args:
        sample: Feature vector (one feature per qubit).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for q in range(n_qubit):
        RX(circuit_initial, q, sample[q])
        RX(circuit_initial, q, sample[q])

    return circuit_initial

def hardware_efficient_embed_encode(sample, n_layers=2):
    """Hardware-efficient embedding with RX(feature) rotations and linear CNOT layers.

    Args:
        sample: Feature vector (one feature per qubit).
        n_layers: Number of RX + CNOT layers (default 2).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for q in range(n_qubit):
            RX(circuit_initial, q, phi=sample[q])
        for q in range(0, n_qubit - 1, 1):
            CNOT(circuit_initial, q, q + 1)
    
    return circuit_initial

def z_feature_map_encode(sample, n_layers=2):
    """First-order Z-feature map encoding.

    Each layer applies ``H`` then ``Phase(2*x)`` per qubit (no entanglement).

    Args:
        sample: Feature vector (one feature per qubit).
        n_layers: Number of layers (default 2).

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for q in range(n_qubit):
            H(circuit_initial, q)
            Phase(circuit_initial, q, phi=2 * sample[q])
    
    return circuit_initial

def zz_feature_map_encode(sample, n_layers=2, entanglement="linear"):
    """Second-order ZZ-feature map encoding with pairwise entanglement.

    Each layer applies ``H``, ``Phase(2*x_i)``, then ZZ interaction terms
    ``Phase(2*(pi - x_i)*(pi - x_j))`` via CNOT-Phase-CNOT blocks.

    Args:
        sample: Feature vector (one feature per qubit).
        n_layers: Number of layers (default 2).
        entanglement: Connectivity — ``"linear"``, ``"circular"``, or ``"full"``.

    Returns:
        Wavefunction: Encoded quantum state.
    """
    sample = np.array(sample)
    n_qubit = len(sample)
    circuit_initial = Qubit(n_qubit)
    for layer in range(n_layers):
        for i in range(n_qubit):
            H(circuit_initial, i)
        for i in range(n_qubit):
            Phase(circuit_initial, i, 2.0 * sample[i])
        if entanglement == "linear":
            for i in range(n_qubit - 1):
                angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[i+1])
                CNOT(circuit_initial, i, i+1)
                Phase(circuit_initial, i+1, angle)
                CNOT(circuit_initial, i, i+1)
        elif entanglement == "circular":
            for i in range(n_qubit - 1):
                angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[i+1])
                CNOT(circuit_initial, i, i+1)
                Phase(circuit_initial, i+1, angle)
                CNOT(circuit_initial, i, i+1)
            angle = 2.0 * (np.pi - sample[-1]) * (np.pi - sample[0])
            CNOT(circuit_initial, n_qubit-1, 0)
            Phase(circuit_initial, 0, angle)
            CNOT(circuit_initial, n_qubit-1, 0)
        elif entanglement == "full":
            for i in range(n_qubit):
                for j in range(i+1, n_qubit):
                    angle = 2.0 * (np.pi - sample[i]) * (np.pi - sample[j])
                    CNOT(circuit_initial, i, j)
                    Phase(circuit_initial, j, angle)
                    CNOT(circuit_initial, i, j)
    
    return circuit_initial
