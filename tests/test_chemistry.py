# -*- coding: utf-8 -*-
"""Tests for qforge.chem — quantum chemistry module."""
import numpy as np
import pytest
from qforge.chem.molecule import Molecule
from qforge.chem.fermion import FermionicOperator, jordan_wigner
from qforge.chem.hartree_fock import hartree_fock_state
from qforge.chem.ucc import uccsd_n_params, _get_excitations
from qforge.circuit import Qubit


class TestMolecule:
    def test_h2_creation(self):
        h2 = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
        assert h2.n_electrons == 2
        assert h2.n_orbitals == 2
        assert h2.n_qubits == 4

    def test_h2_hamiltonian(self):
        h2 = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
        H = h2.hamiltonian(mapping='jordan_wigner')
        assert len(H.coeffs) > 0
        assert len(H.terms) == len(H.coeffs)

    def test_nuclear_repulsion(self):
        h2 = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
        E_nuc = h2.nuclear_repulsion
        # H2 at 0.74 A: nuclear repulsion ~ 0.72 Hartree
        assert 0.5 < E_nuc < 1.0

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Unknown element"):
            Molecule([('Xx', (0, 0, 0))])

    def test_repr(self):
        h2 = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
        assert 'HH' in repr(h2)


class TestFermionicOperator:
    def test_creation(self):
        fop = FermionicOperator({((0, 'c'), (1, 'a')): 0.5})
        assert len(fop.terms) == 1
        assert fop.n_modes == 2

    def test_add(self):
        f1 = FermionicOperator({((0, 'c'), (0, 'a')): 1.0})
        f2 = FermionicOperator({((1, 'c'), (1, 'a')): 2.0})
        result = f1 + f2
        assert len(result.terms) == 2

    def test_scalar_mul(self):
        f = FermionicOperator({((0, 'c'), (0, 'a')): 1.0})
        result = f * 3.0
        key = list(result.terms.keys())[0]
        assert abs(result.terms[key] - 3.0) < 1e-10

    def test_jordan_wigner_number_op(self):
        """JW transform of n_0 = a†_0 a_0 should give (I - Z_0)/2."""
        fop = FermionicOperator({((0, 'c'), (0, 'a')): 1.0})
        H = jordan_wigner(fop)
        assert len(H.coeffs) > 0


class TestHartreeFock:
    def test_hf_state_2e_4q(self):
        """HF state for 2 electrons in 4 qubits: |1100>."""
        wf = Qubit(4, backend='python')
        hartree_fock_state(wf, 2)
        # |1100> = index 12 (binary 1100)
        assert abs(wf.amplitude[12]) > 0.99
        assert abs(wf.amplitude[0]) < 0.01

    def test_hf_state_0e(self):
        """HF with 0 electrons stays |0000>."""
        wf = Qubit(4, backend='python')
        hartree_fock_state(wf, 0)
        assert abs(wf.amplitude[0]) > 0.99


class TestUCCSD:
    def test_n_params(self):
        n = uccsd_n_params(2, 2)
        assert n > 0

    def test_excitations(self):
        singles, doubles = _get_excitations(2, 2)
        assert len(singles) > 0
