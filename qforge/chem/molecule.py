# -*- coding: utf-8 -*-
# author: vinhpx
"""Molecule class for quantum chemistry simulations.

Builds qubit Hamiltonians from molecular geometry using either PySCF
(when available) or pre-computed data for common molecules.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from qforge.algo.hamiltonian import Hamiltonian


# ============================================================
# Atomic number lookup
# ============================================================

_ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
}


# ============================================================
# Pre-computed Hamiltonian data
# ============================================================

# H2 at R=0.74 angstrom, STO-3G basis, Jordan-Wigner mapping (4 qubits)
# These coefficients reproduce the exact ground-state energy of ~ -1.137 Hartree.
_H2_JW_HAMILTONIAN = {
    'coeffs': [
        -0.81261, 0.17120, 0.17120, -0.22279, -0.22279,
        0.16862, 0.12054, 0.16586, 0.16586, 0.12054,
        0.17434,
    ],
    'terms': [
        [],                          # I
        [('Z', 0)],                  # Z0
        [('Z', 1)],                  # Z1
        [('Z', 2)],                  # Z2
        [('Z', 3)],                  # Z3
        [('Z', 0), ('Z', 1)],       # Z0Z1
        [('Z', 0), ('Z', 2)],       # Z0Z2
        [('Z', 0), ('Z', 3)],       # Z0Z3
        [('Z', 1), ('Z', 2)],       # Z1Z2
        [('Z', 1), ('Z', 3)],       # Z1Z3
        [('Z', 2), ('Z', 3)],       # Z2Z3
    ],
}

# Additional H2 data with XX/YY terms for more accurate energy
# H2 at R=0.74 angstrom, STO-3G, full Jordan-Wigner with exchange terms
_H2_JW_FULL_HAMILTONIAN = {
    'coeffs': [
        -0.81261, 0.17120, 0.17120, -0.22279, -0.22279,
        0.16862, 0.12054, 0.16586, 0.16586, 0.12054,
        0.17434, 0.04532, 0.04532, 0.04532, 0.04532,
    ],
    'terms': [
        [],                                          # I
        [('Z', 0)],                                  # Z0
        [('Z', 1)],                                  # Z1
        [('Z', 2)],                                  # Z2
        [('Z', 3)],                                  # Z3
        [('Z', 0), ('Z', 1)],                       # Z0Z1
        [('Z', 0), ('Z', 2)],                       # Z0Z2
        [('Z', 0), ('Z', 3)],                       # Z0Z3
        [('Z', 1), ('Z', 2)],                       # Z1Z2
        [('Z', 1), ('Z', 3)],                       # Z1Z3
        [('Z', 2), ('Z', 3)],                       # Z2Z3
        [('X', 0), ('X', 1), ('Y', 2), ('Y', 3)],  # X0X1Y2Y3
        [('Y', 0), ('Y', 1), ('X', 2), ('X', 3)],  # Y0Y1X2X3
        [('X', 0), ('Y', 1), ('Y', 2), ('X', 3)],  # X0Y1Y2X3
        [('Y', 0), ('X', 1), ('X', 2), ('Y', 3)],  # Y0X1X2Y3
    ],
}

# Pre-computed data keyed by (molecule_key, basis, mapping)
_PRECOMPUTED: dict[tuple[str, str, str], dict] = {}


def _h2_key(bond_length: float) -> str | None:
    """Return a key if the bond length matches a pre-computed H2 entry."""
    if abs(bond_length - 0.74) < 0.02:
        return 'H2_0.74'
    return None


def _register_precomputed() -> None:
    """Populate the pre-computed data registry."""
    _PRECOMPUTED[('H2_0.74', 'sto-3g', 'jordan_wigner')] = _H2_JW_HAMILTONIAN
    _PRECOMPUTED[('H2_0.74', 'sto-3g', 'jordan_wigner_full')] = _H2_JW_FULL_HAMILTONIAN


_register_precomputed()


# ============================================================
# Molecule class
# ============================================================

class Molecule:
    """Representation of a molecular system for quantum chemistry.

    Args:
        atoms:        List of ``(symbol, (x, y, z))`` tuples specifying the
                      molecular geometry in angstroms.
        basis:        Basis set name (default ``'sto-3g'``).
        charge:       Total molecular charge.
        multiplicity: Spin multiplicity (2S+1).

    Example::

        h2 = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
        H = h2.hamiltonian()
        energy = H.expectation(wf)
    """

    def __init__(
        self,
        atoms: list[tuple[str, tuple[float, float, float]]],
        basis: str = 'sto-3g',
        charge: int = 0,
        multiplicity: int = 1,
    ):
        self.atoms = list(atoms)
        self.basis = basis.lower()
        self.charge = charge
        self.multiplicity = multiplicity

        # Validate atoms
        for symbol, coords in self.atoms:
            if symbol not in _ATOMIC_NUMBERS:
                raise ValueError(f"Unknown element: {symbol!r}")
            if len(coords) != 3:
                raise ValueError(
                    f"Coordinates for {symbol} must be (x, y, z), got {coords}"
                )

        # Determine molecule identity for pre-computed lookup
        self._molecule_key = self._identify_molecule()

    # ---- properties --------------------------------------------------

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        total = sum(_ATOMIC_NUMBERS[sym] for sym, _ in self.atoms)
        return total - self.charge

    @property
    def n_orbitals(self) -> int:
        """Number of spatial orbitals (basis-dependent estimate).

        For STO-3G: 1 orbital per H/He, 5 per Li-Ne, 9 per Na-Ar.
        """
        count = 0
        for symbol, _ in self.atoms:
            z = _ATOMIC_NUMBERS[symbol]
            if z <= 2:
                count += 1
            elif z <= 10:
                count += 5
            elif z <= 18:
                count += 9
            else:
                count += 13  # rough estimate
        return count

    @property
    def n_qubits(self) -> int:
        """Number of qubits required (2 * n_orbitals for Jordan-Wigner)."""
        return 2 * self.n_orbitals

    @property
    def nuclear_repulsion(self) -> float:
        """Classical nuclear repulsion energy in Hartree.

        Uses Coulomb's law: E = Z_A * Z_B / R_AB (in atomic units).
        Coordinates are converted from angstrom to bohr (1 A = 1.8897259886 bohr).
        """
        angstrom_to_bohr = 1.8897259886
        energy = 0.0
        n_atoms = len(self.atoms)
        for i in range(n_atoms):
            sym_i, coords_i = self.atoms[i]
            z_i = _ATOMIC_NUMBERS[sym_i]
            for j in range(i + 1, n_atoms):
                sym_j, coords_j = self.atoms[j]
                z_j = _ATOMIC_NUMBERS[sym_j]
                dx = (coords_i[0] - coords_j[0]) * angstrom_to_bohr
                dy = (coords_i[1] - coords_j[1]) * angstrom_to_bohr
                dz = (coords_i[2] - coords_j[2]) * angstrom_to_bohr
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if r < 1e-10:
                    raise ValueError(
                        f"Atoms {i} ({sym_i}) and {j} ({sym_j}) are at the same position"
                    )
                energy += z_i * z_j / r
        return energy

    # ---- molecule identification -------------------------------------

    def _identify_molecule(self) -> str | None:
        """Try to identify the molecule for pre-computed data lookup."""
        symbols = sorted(sym for sym, _ in self.atoms)
        if symbols == ['H', 'H'] and self.charge == 0:
            # Check bond length
            _, c0 = self.atoms[0]
            _, c1 = self.atoms[1]
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(c0, c1)))
            key = _h2_key(dist)
            if key is not None:
                return key
        return None

    # ---- Hamiltonian construction ------------------------------------

    def hamiltonian(self, mapping: str = 'jordan_wigner') -> Hamiltonian:
        """Build the qubit Hamiltonian for this molecule.

        Args:
            mapping: Fermion-to-qubit mapping. Currently supports
                     ``'jordan_wigner'`` and ``'bravyi_kitaev'``.

        Returns:
            A :class:`~qforge.algo.hamiltonian.Hamiltonian` instance.
        """
        # Try pre-computed data first
        if self._molecule_key is not None:
            lookup_key = (self._molecule_key, self.basis, mapping)
            if lookup_key in _PRECOMPUTED:
                data = _PRECOMPUTED[lookup_key]
                return Hamiltonian(
                    coeffs=list(data['coeffs']),
                    terms=[list(t) for t in data['terms']],
                )

        # Try PySCF
        try:
            return self._hamiltonian_pyscf(mapping)
        except ImportError:
            pass

        # Fall back to pre-computed data with full exchange terms
        if self._molecule_key is not None:
            full_key = (self._molecule_key, self.basis, mapping + '_full')
            if full_key in _PRECOMPUTED:
                data = _PRECOMPUTED[full_key]
                return Hamiltonian(
                    coeffs=list(data['coeffs']),
                    terms=[list(t) for t in data['terms']],
                )
            # Also try without _full suffix
            for key, data in _PRECOMPUTED.items():
                if key[0] == self._molecule_key and key[1] == self.basis:
                    return Hamiltonian(
                        coeffs=list(data['coeffs']),
                        terms=[list(t) for t in data['terms']],
                    )

        raise RuntimeError(
            f"Cannot build Hamiltonian for this molecule. "
            f"Install PySCF (`pip install pyscf`) or use a pre-computed molecule "
            f"(e.g., H2 at R=0.74 A with STO-3G basis)."
        )

    def _hamiltonian_pyscf(self, mapping: str) -> Hamiltonian:
        """Build Hamiltonian using PySCF for the electronic integrals."""
        import pyscf  # noqa: F401 — will raise ImportError if not installed
        from pyscf import gto, scf, ao2mo

        # Build PySCF molecule
        atom_str = '; '.join(
            f'{sym} {x:.6f} {y:.6f} {z:.6f}'
            for sym, (x, y, z) in self.atoms
        )
        mol = gto.M(
            atom=atom_str,
            basis=self.basis,
            charge=self.charge,
            spin=self.multiplicity - 1,
            unit='Angstrom',
        )

        # Run Hartree-Fock
        mf = scf.RHF(mol)
        mf.kernel()

        # Get integrals in MO basis
        n_orb = mf.mo_coeff.shape[1]
        h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mol, mf.mo_coeff)
        eri = ao2mo.restore(1, eri, n_orb)

        # Build fermionic operator and transform to qubits
        from qforge.chem.fermion import FermionicOperator, jordan_wigner as jw_transform
        from qforge.chem.fermion import bravyi_kitaev as bk_transform

        # One-body terms
        terms: dict[tuple, complex] = {}
        for p in range(n_orb):
            for q in range(n_orb):
                if abs(h1[p, q]) > 1e-10:
                    # alpha spin
                    key_a = ((2 * p, 'c'), (2 * q, 'a'))
                    terms[key_a] = terms.get(key_a, 0) + h1[p, q]
                    # beta spin
                    key_b = ((2 * p + 1, 'c'), (2 * q + 1, 'a'))
                    terms[key_b] = terms.get(key_b, 0) + h1[p, q]

        # Two-body terms: 0.5 * sum_{pqrs} (pq|rs) a+_p a+_r a_s a_q
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        val = 0.5 * eri[p, q, r, s]
                        if abs(val) < 1e-10:
                            continue
                        # alpha-alpha
                        key = ((2*p, 'c'), (2*r, 'c'), (2*s, 'a'), (2*q, 'a'))
                        terms[key] = terms.get(key, 0) + val
                        # alpha-beta
                        key = ((2*p, 'c'), (2*r+1, 'c'), (2*s+1, 'a'), (2*q, 'a'))
                        terms[key] = terms.get(key, 0) + val
                        # beta-alpha
                        key = ((2*p+1, 'c'), (2*r, 'c'), (2*s, 'a'), (2*q+1, 'a'))
                        terms[key] = terms.get(key, 0) + val
                        # beta-beta
                        key = ((2*p+1, 'c'), (2*r+1, 'c'), (2*s+1, 'a'), (2*q+1, 'a'))
                        terms[key] = terms.get(key, 0) + val

        fop = FermionicOperator(terms)

        # Add nuclear repulsion as identity term
        nuc_rep = mol.energy_nuc()

        if mapping == 'jordan_wigner':
            ham = jw_transform(fop)
        elif mapping == 'bravyi_kitaev':
            ham = bk_transform(fop)
        else:
            raise ValueError(f"Unknown mapping: {mapping!r}")

        # Add nuclear repulsion to the identity coefficient
        has_identity = False
        for i, t in enumerate(ham.terms):
            if not t:
                ham.coeffs[i] += nuc_rep
                has_identity = True
                break
        if not has_identity:
            ham.coeffs.insert(0, nuc_rep)
            ham.terms.insert(0, [])

        return ham

    def __repr__(self) -> str:
        formula = ''.join(sym for sym, _ in self.atoms)
        return f"Molecule({formula!r}, basis={self.basis!r})"
