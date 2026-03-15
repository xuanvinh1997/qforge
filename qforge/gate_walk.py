    # -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:26:23 2021

@author: ASUS
"""
import numpy as np
import math
import cmath

def shift_walk(wavefunction, dim):
    """Apply the conditional shift operator for a discrete-time quantum walk.

    Args:
        wavefunction: Quantum walk state (coin + position registers).
        dim: Spatial dimension of the walk (``1`` or ``2``).

    Raises:
        TypeError: If ``dim`` is not 1 or 2.
    """
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    if dim == 1:
        qubit_num = int((len(states) + 2)/4)
        for i in range(0, 2*qubit_num-2):
            new_amplitude[i+1] += amplitude[i]
        for i in range(2*(qubit_num), 4*(qubit_num)-2):
            new_amplitude[i-1] += amplitude[i]
    else:
        cut = int(len(wavefunction.state)/4)
        qubit_num = int(np.sqrt(len(wavefunction.state)/4))
        for i in range(0, qubit_num):
            for j in range(i*qubit_num, (i+1)*qubit_num-1):
                new_amplitude[j+1] += amplitude[j]
        for i in range(0, qubit_num-1):
            for j in range(i*qubit_num, (i+1)*qubit_num):
                new_amplitude[j+cut+qubit_num] += amplitude[j+cut]
        for i in range(1, qubit_num):
            for j in range(i*qubit_num, (i+1)*qubit_num):
                new_amplitude[j+2*cut-qubit_num] += amplitude[j+2*cut]
        for i in range(0, qubit_num):
            for j in range(i*qubit_num+1, (i+1)*qubit_num):
                new_amplitude[j+3*cut-1] += amplitude[j+3*cut]
    wavefunction.amplitude = new_amplitude
    
def h_coin(wavefunction, dim):
    """Apply the Hadamard coin operator for a quantum walk.

    Uses a 2x2 Hadamard for 1D walks or a 4x4 Grover-like matrix for 2D walks.

    Args:
        wavefunction: Quantum walk state.
        dim: Spatial dimension (``1`` or ``2``).

    Raises:
        TypeError: If ``dim`` is not 1 or 2.
    """
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    if dim == 1:
        qubit_num = int((len(states) + 2)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] += amplitude[i]/2**0.5
                new_amplitude[i+2*qubit_num-1] += amplitude[i]/2**0.5
            else:
                new_amplitude[i] -= amplitude[i]/2**0.5
                new_amplitude[i-2*qubit_num+1] += amplitude[i]/2**0.5
    else:
        cut = int(len(wavefunction.state)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] += amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
                new_amplitude[i+3*cut] += amplitude[i]/2
            elif states[i][0] == '1':
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] -= amplitude[i]/2
            elif states[i][0] == '2':
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] -= amplitude[i]/2
            else:
                new_amplitude[i-3*cut] += amplitude[i]/2
                new_amplitude[i-2*cut] -= amplitude[i]/2
                new_amplitude[i-cut] -= amplitude[i]/2
                new_amplitude[i] += amplitude[i]/2
    wavefunction.amplitude = new_amplitude

def grover_coin(wavefunction, dim):
    """Apply the Grover diffusion coin operator for a 2D quantum walk.

    Args:
        wavefunction: Quantum walk state.
        dim: Must be ``2`` (Grover coin is only defined for 2D walks).

    Raises:
        TypeError: If ``dim`` is not 2.
    """
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    new_amplitude = np.zeros(len(amplitude), dtype = complex)
    if dim != 2:
        raise TypeError('The dimension of the quantum walk must be 2')
    else:
        cut = int(len(wavefunction.state)/4)
        for i in np.nonzero(amplitude)[0]:
            if states[i][0] == '0':
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
                new_amplitude[i+3*cut] += amplitude[i]/2
            elif states[i][0] == '1':
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
                new_amplitude[i+2*cut] += amplitude[i]/2
            elif states[i][0] == '2':
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
                new_amplitude[i+cut] += amplitude[i]/2
            else:
                new_amplitude[i-3*cut] += amplitude[i]/2
                new_amplitude[i-2*cut] += amplitude[i]/2
                new_amplitude[i-cut] += amplitude[i]/2
                new_amplitude[i] -= amplitude[i]/2
    wavefunction.amplitude = new_amplitude
    
def quantum_walk_hadamard(wavefunction, dim, iteration):
    """Run a discrete-time quantum walk with a Hadamard coin.

    Applies ``iteration`` steps of (Hadamard coin + shift).

    Args:
        wavefunction: Initial quantum walk state (created via ``Walk_Qubit``).
        dim: Spatial dimension (``1`` or ``2``).
        iteration: Number of walk steps.
    """
    for i in range(iteration):
        h_coin(wavefunction, dim)
        shift_walk(wavefunction, dim)
        
def quantum_walk_grover(wavefunction, dim, iteration):
    """Run a discrete-time quantum walk with a Grover coin (2D only).

    Applies ``iteration`` steps of (Grover coin + shift).

    Args:
        wavefunction: Initial quantum walk state (created via ``Walk_Qubit``).
        dim: Must be ``2``.
        iteration: Number of walk steps.
    """
    for i in range(iteration):
        grover_coin(wavefunction, dim)
        shift_walk(wavefunction, dim)