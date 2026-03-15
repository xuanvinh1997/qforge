# -*- coding: utf-8 -*-
# author: vinhpx
"""Wavefunction class for representing quantum states."""
import numpy as np
from qforge._utils import (
    _VIS_SINGLE, _VIS_ROTATION, _VIS_WIRE, _VIS_CTRL_ROT,
    _VIS_CTRL, _VIS_CTRL_PHASE, _VIS_DOUBLE_CTRL, _VIS_CTRL_SWAP,
)


class Wavefunction(object):
    """Quantum state represented as a complex amplitude vector over computational basis states.

    A Wavefunction holds a 2^n-dimensional complex vector (amplitudes) and the
    corresponding n-qubit basis-state labels. It transparently wraps either a
    pure-Python numpy backend or a C++ ``StateVector`` for accelerated simulation.

    Args:
        states: List of basis-state strings, e.g. ``['00', '01', '10', '11']``.
        amplitude_vector: Complex amplitude array of length 2^n.
        _sv: Optional C++ ``StateVector`` backend (set automatically by ``Qubit()``).

    Attributes:
        state: List of basis-state label strings.
        amplitude: Complex amplitude vector (numpy array). When using the C++
            backend this is a zero-copy view into the ``StateVector``.
        visual: List of gate operations applied, used by ``visual_circuit()``.

    Example:
        >>> from qforge.circuit import Qubit
        >>> from qforge.gates import H, CNOT
        >>> wf = Qubit(2)          # |00>
        >>> H(wf, 0)               # (|00> + |10>) / sqrt(2)
        >>> CNOT(wf, 0, 1)         # Bell state (|00> + |11>) / sqrt(2)
        >>> wf.probabilities()     # array([0.5, 0. , 0. , 0.5])
    """

    def __init__(self, states, amplitude_vector, _sv=None):
        self.state = states
        self._sv = _sv
        if _sv is not None:
            # C++ backend: amplitude is a zero-copy numpy view
            self._sv.amplitude[:] = amplitude_vector
        else:
            self._amplitude = amplitude_vector
        self.visual = []

    @property
    def amplitude(self):
        if self._sv is not None:
            return self._sv.amplitude
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if self._sv is not None:
            self._sv.amplitude = np.asarray(value, dtype=complex)
        else:
            self._amplitude = value

    def probabilities(self):
        """Return the probability of each basis state as a numpy array.

        Returns:
            numpy.ndarray: Array of ``|amplitude|^2`` values, indexed by basis state.
        """
        return np.abs(self.amplitude) ** 2

    def print_state(self):
        """Return the quantum state as a human-readable bra-ket string.

        Returns:
            str: State in the form ``(0.707+0j)|00> + (0.707+0j)|11>``.
        """
        states = self.state
        amp = self.amplitude
        string = str(amp[0]) + '|' + states[0] + '>'
        for i in range(1, len(states)):
            string += ' + ' + str(amp[i]) + '|' + states[i] + '>'
        return string

    def visual_circuit(self):
        """Print an ASCII circuit diagram of all gates applied to this wavefunction."""
        n = len((self.state)[0])
        a = self.visual
        b = [[]]*(2*n)
        for i in range(2*n):
            b[i] = [0]*len(a)

        for i in range(n):
            for j in range(len(a)):
                if i in a[j]:
                    if ('RX' in a[j]) or ('RY' in a[j]) or ('RZ' in a[j]):
                        b[2*i][j] = _VIS_ROTATION
                    elif ('CRX' in a[j]) or ('CRY' in a[j]) or ('CRZ' in a[j]):
                        b[2*i][j] = _VIS_CTRL_ROT
                    elif ('CX' in a[j]) or ('SWAP' in a[j]):
                        b[2*i][j] = _VIS_CTRL
                    elif ('CP' in a[j]):
                        b[2*i][j] = _VIS_CTRL_PHASE
                    elif ('CCX' in a[j]):
                        b[2*i][j] = _VIS_DOUBLE_CTRL
                    elif ('CSWAP' in a[j]):
                        b[2*i][j] = _VIS_CTRL_SWAP
                    else:
                        b[2*i][j] = _VIS_SINGLE

        for j in range(len(a)):
            if ('CX' in a[j]) or ('CCX' in a[j]) or ('SWAP' in a[j]) or ('CSWAP' in a[j]):
                for i in range(2*min(a[j][:-1])+1, 2*max(a[j][:-1]), 2):
                    b[i][j] = _VIS_WIRE
            if ('CP' in a[j]) or ('CRX' in a[j]):
                for i in range(2*min(a[j][:-2])+1, 2*max(a[j][:-2]), 2):
                    b[i][j] = _VIS_WIRE

        string_out = [[]]*(2*n)
        for i in range(2*n):
            string_out[i] = []

        for i in range(n):
            out = ''
            if i < 10:
                out += '|Q_'+str(i)+'> : '
            else:
                out += '|Q_'+str(i)+'>: '
            space = ' '*len(out)
            string_out[2*i].append(out)
            string_out[2*i+1].append(space)

            out = ''
            space = ''
            for j in range(len(a)):

                if b[2*i][j] == 0:
                    out += '---'

                if b[2*i][j] == _VIS_SINGLE:
                    out += a[j][-1] + '--'

                if b[2*i][j] == _VIS_ROTATION:
                    out += a[j][-2] + '-'

                if b[2*i][j] == _VIS_CTRL_ROT:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1:] + '-'

                if b[2*i][j] == _VIS_CTRL:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += 'x--'

                if b[2*i][j] == _VIS_CTRL_PHASE:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1] + '--'

                if b[2*i][j] == _VIS_DOUBLE_CTRL:
                    if i == a[j][0] or i == a[j][1]:
                        out += 'o--'
                    elif i == a[j][2]:
                        out += 'x--'

                if b[2*i][j] == _VIS_CTRL_SWAP:
                    if i == a[j][0] or i == a[j][1]:
                        out += 'x--'
                    elif i == a[j][2]:
                        out += 'o--'


                if b[2*i+1][j] == _VIS_WIRE:
                    space += '|  '
                if b[2*i+1][j] == 0:
                    space += '   '

            string_out[2*i].append(out+'-M')
            string_out[2*i+1].append(space+'  ')

        for i in string_out:
            print(i[0]+i[1])
