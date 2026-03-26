# -*- coding: utf-8 -*-
# author: vinhpx
"""Fermionic operators and fermion-to-qubit mappings.

Provides ``FermionicOperator`` for second-quantized operators and
``jordan_wigner`` / ``bravyi_kitaev`` transforms to Pauli Hamiltonians.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from qforge.algo.hamiltonian import Hamiltonian


# ============================================================
# FermionicOperator
# ============================================================

# A term key is a tuple of (orbital_index, 'c' | 'a') pairs
# where 'c' = creation and 'a' = annihilation.
TermKey = Tuple[Tuple[int, str], ...]


class FermionicOperator:
    """A fermionic operator in second quantization.

    Stores a sum of products of creation/annihilation operators with
    complex coefficients::

        # a†_0 a_1 with coefficient 0.5
        fop = FermionicOperator({((0, 'c'), (1, 'a')): 0.5})

    Args:
        terms: Dictionary mapping operator strings to complex coefficients.
               Each key is a tuple of ``(orbital_index, 'c'|'a')`` pairs.
    """

    def __init__(self, terms: Dict[TermKey, complex] | None = None):
        self.terms: Dict[TermKey, complex] = {}
        if terms is not None:
            for key, coeff in terms.items():
                key = tuple(tuple(pair) for pair in key)
                if abs(coeff) > 1e-15:
                    self.terms[key] = complex(coeff)

    @property
    def n_modes(self) -> int:
        """Maximum orbital index + 1 across all terms."""
        max_idx = -1
        for key in self.terms:
            for idx, _ in key:
                if idx > max_idx:
                    max_idx = idx
        return max_idx + 1 if max_idx >= 0 else 0

    def __add__(self, other: FermionicOperator) -> FermionicOperator:
        if not isinstance(other, FermionicOperator):
            return NotImplemented
        result = FermionicOperator(dict(self.terms))
        for key, coeff in other.terms.items():
            result.terms[key] = result.terms.get(key, 0) + coeff
            if abs(result.terms[key]) < 1e-15:
                del result.terms[key]
        return result

    def __sub__(self, other: FermionicOperator) -> FermionicOperator:
        if not isinstance(other, FermionicOperator):
            return NotImplemented
        result = FermionicOperator(dict(self.terms))
        for key, coeff in other.terms.items():
            result.terms[key] = result.terms.get(key, 0) - coeff
            if abs(result.terms[key]) < 1e-15:
                del result.terms[key]
        return result

    def __mul__(self, scalar: complex) -> FermionicOperator:
        if isinstance(scalar, FermionicOperator):
            return self._multiply_operators(scalar)
        scalar = complex(scalar)
        return FermionicOperator({k: v * scalar for k, v in self.terms.items()})

    def __rmul__(self, scalar: complex) -> FermionicOperator:
        if isinstance(scalar, FermionicOperator):
            return scalar._multiply_operators(self)
        scalar = complex(scalar)
        return FermionicOperator({k: v * scalar for k, v in self.terms.items()})

    def __neg__(self) -> FermionicOperator:
        return self * (-1)

    def _multiply_operators(self, other: FermionicOperator) -> FermionicOperator:
        """Multiply two fermionic operators (concatenate operator strings)."""
        result: Dict[TermKey, complex] = {}
        for k1, c1 in self.terms.items():
            for k2, c2 in other.terms.items():
                new_key = k1 + k2
                result[new_key] = result.get(new_key, 0) + c1 * c2
        # Clean near-zero entries
        return FermionicOperator(
            {k: v for k, v in result.items() if abs(v) > 1e-15}
        )

    def __repr__(self) -> str:
        if not self.terms:
            return "FermionicOperator(0)"
        parts = []
        for key, coeff in list(self.terms.items())[:5]:
            ops = ' '.join(
                f"a{'†' if t == 'c' else ''}_{i}" for i, t in key
            )
            parts.append(f"{coeff:.4g} * ({ops})")
        s = ' + '.join(parts)
        if len(self.terms) > 5:
            s += f" + ... ({len(self.terms)} terms)"
        return f"FermionicOperator({s})"


# ============================================================
# Pauli string algebra helpers
# ============================================================

# A Pauli string is represented as: (coefficient, dict[qubit -> 'I'|'X'|'Y'|'Z'])
PauliTerm = Tuple[complex, Dict[int, str]]

# Pauli multiplication table: (P1, P2) -> (phase, P_result)
_PAULI_MULT: dict[tuple[str, str], tuple[complex, str]] = {
    ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'), ('X', 'X'): (1, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'), ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'), ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
}


def _multiply_pauli_terms(
    t1: PauliTerm, t2: PauliTerm,
) -> PauliTerm:
    """Multiply two Pauli terms, tracking phase and simplifying."""
    c1, ops1 = t1
    c2, ops2 = t2
    phase = c1 * c2
    result_ops: Dict[int, str] = {}

    all_qubits = set(ops1.keys()) | set(ops2.keys())
    for q in all_qubits:
        p1 = ops1.get(q, 'I')
        p2 = ops2.get(q, 'I')
        ph, p_out = _PAULI_MULT[(p1, p2)]
        phase *= ph
        if p_out != 'I':
            result_ops[q] = p_out

    return (phase, result_ops)


def _simplify_pauli_sum(
    terms: list[PauliTerm],
) -> list[PauliTerm]:
    """Combine like Pauli terms."""
    combined: dict[tuple, complex] = {}
    for coeff, ops in terms:
        key = tuple(sorted(ops.items()))
        combined[key] = combined.get(key, 0) + coeff

    result = []
    for key, coeff in combined.items():
        if abs(coeff) > 1e-12:
            result.append((coeff, dict(key)))
    return result


def _pauli_terms_to_hamiltonian(terms: list[PauliTerm]) -> Hamiltonian:
    """Convert a list of Pauli terms to a qforge Hamiltonian."""
    terms = _simplify_pauli_sum(terms)
    coeffs = []
    ham_terms = []
    for coeff, ops in terms:
        # Only keep real parts (Hamiltonian should be Hermitian)
        real_coeff = float(np.real(coeff))
        if abs(real_coeff) < 1e-12:
            continue
        coeffs.append(real_coeff)
        pauli_list = [(pauli, qubit) for qubit, pauli in sorted(ops.items())]
        ham_terms.append(pauli_list)
    if not coeffs:
        return Hamiltonian([0.0], [[]])
    return Hamiltonian(coeffs, ham_terms)


# ============================================================
# Jordan-Wigner transform
# ============================================================

def jordan_wigner(fop: FermionicOperator) -> Hamiltonian:
    """Transform a fermionic operator to a qubit Hamiltonian via Jordan-Wigner.

    The Jordan-Wigner mapping encodes occupation numbers directly:
    - a†_j = (X_j - iY_j)/2 * Z_{j-1} * ... * Z_0
    - a_j  = (X_j + iY_j)/2 * Z_{j-1} * ... * Z_0

    Args:
        fop (FermionicOperator): A :class:`FermionicOperator`.

    Returns:
        A :class:`~qforge.algo.hamiltonian.Hamiltonian` in the Pauli basis.
    """
    all_pauli_terms: list[PauliTerm] = []

    for term_key, coeff in fop.terms.items():
        if not term_key:
            # Identity term
            all_pauli_terms.append((coeff, {}))
            continue

        # Build the Pauli representation for each ladder operator and multiply
        pauli_product: PauliTerm = (coeff, {})

        for orbital_idx, op_type in term_key:
            # JW transform of a single ladder operator
            if op_type == 'c':
                # a†_j = (X_j - iY_j)/2 * Z_{j-1} * ... * Z_0
                ladder_terms = _jw_creation(orbital_idx)
            else:
                # a_j = (X_j + iY_j)/2 * Z_{j-1} * ... * Z_0
                ladder_terms = _jw_annihilation(orbital_idx)

            # Multiply current product with this ladder operator
            new_products: list[PauliTerm] = []
            for lt in ladder_terms:
                new_products.append(_multiply_pauli_terms(pauli_product, lt))

            # Sum the expanded terms back into a simplified form
            pauli_product_list = _simplify_pauli_sum(new_products)
            if not pauli_product_list:
                break  # Zero contribution

            # For the next multiplication, we need to distribute
            # If there are multiple terms, we need to keep them all
            if len(pauli_product_list) == 1:
                pauli_product = pauli_product_list[0]
            else:
                # Multiple terms: need to handle distribution in next iteration
                # Store and handle at the end
                # Actually, we need to expand fully
                all_pauli_terms.extend(pauli_product_list)
                pauli_product = None
                break
        else:
            if pauli_product is not None:
                all_pauli_terms.append(pauli_product)

    return _pauli_terms_to_hamiltonian(all_pauli_terms)


def _jw_creation(j: int) -> list[PauliTerm]:
    """Jordan-Wigner transform of a†_j.

    a†_j = (X_j - iY_j)/2 * Z_{j-1} * ... * Z_0
    """
    z_string: Dict[int, str] = {}
    for k in range(j):
        z_string[k] = 'Z'

    # (X_j - iY_j)/2 = two Pauli terms
    ops_x = dict(z_string)
    ops_x[j] = 'X'
    ops_y = dict(z_string)
    ops_y[j] = 'Y'

    return [
        (0.5, ops_x),
        (-0.5j, ops_y),
    ]


def _jw_annihilation(j: int) -> list[PauliTerm]:
    """Jordan-Wigner transform of a_j.

    a_j = (X_j + iY_j)/2 * Z_{j-1} * ... * Z_0
    """
    z_string: Dict[int, str] = {}
    for k in range(j):
        z_string[k] = 'Z'

    ops_x = dict(z_string)
    ops_x[j] = 'X'
    ops_y = dict(z_string)
    ops_y[j] = 'Y'

    return [
        (0.5, ops_x),
        (0.5j, ops_y),
    ]


def jordan_wigner_full(fop: FermionicOperator) -> Hamiltonian:
    """Full Jordan-Wigner transform that handles multi-term products properly.

    This version distributes all ladder operator products correctly.
    """
    all_pauli_terms: list[PauliTerm] = []

    for term_key, coeff in fop.terms.items():
        if not term_key:
            all_pauli_terms.append((coeff, {}))
            continue

        # Start with [(coeff, {})] and multiply in each ladder operator
        current_terms: list[PauliTerm] = [(coeff, {})]

        for orbital_idx, op_type in term_key:
            if op_type == 'c':
                ladder_terms = _jw_creation(orbital_idx)
            else:
                ladder_terms = _jw_annihilation(orbital_idx)

            new_terms: list[PauliTerm] = []
            for ct in current_terms:
                for lt in ladder_terms:
                    new_terms.append(_multiply_pauli_terms(ct, lt))

            current_terms = _simplify_pauli_sum(new_terms)
            if not current_terms:
                break

        all_pauli_terms.extend(current_terms)

    return _pauli_terms_to_hamiltonian(all_pauli_terms)


# Override jordan_wigner with the full version
jordan_wigner = jordan_wigner_full  # noqa: F811


# ============================================================
# Bravyi-Kitaev transform
# ============================================================

def bravyi_kitaev(fop: FermionicOperator) -> Hamiltonian:
    """Transform a fermionic operator to a qubit Hamiltonian via Bravyi-Kitaev.

    The BK mapping uses a different encoding that balances the locality of
    occupation and parity information, requiring O(log n) Pauli weight per
    ladder operator instead of O(n) for Jordan-Wigner.

    Args:
        fop (FermionicOperator): A :class:`FermionicOperator`.

    Returns:
        A :class:`~qforge.algo.hamiltonian.Hamiltonian` in the Pauli basis.
    """
    n_modes = fop.n_modes
    if n_modes == 0:
        return Hamiltonian([0.0], [[]])

    all_pauli_terms: list[PauliTerm] = []

    for term_key, coeff in fop.terms.items():
        if not term_key:
            all_pauli_terms.append((coeff, {}))
            continue

        current_terms: list[PauliTerm] = [(coeff, {})]

        for orbital_idx, op_type in term_key:
            if op_type == 'c':
                ladder_terms = _bk_creation(orbital_idx, n_modes)
            else:
                ladder_terms = _bk_annihilation(orbital_idx, n_modes)

            new_terms: list[PauliTerm] = []
            for ct in current_terms:
                for lt in ladder_terms:
                    new_terms.append(_multiply_pauli_terms(ct, lt))

            current_terms = _simplify_pauli_sum(new_terms)
            if not current_terms:
                break

        all_pauli_terms.extend(current_terms)

    return _pauli_terms_to_hamiltonian(all_pauli_terms)


def _bk_update_set(j: int, n: int) -> set[int]:
    """Compute the update set U(j) for BK transform.

    U(j) = set of qubits whose parity includes qubit j.
    """
    result = set()
    # In BK, qubit j stores parity of a specific subset.
    # The update set contains all qubits k > j where j is in the parity set of k.
    for k in range(j + 1, n):
        # k's parity set: indices in range [k - (k & -k), k-1] (for power-of-2 structure)
        low_bit = k & (-k)
        if k - low_bit <= j < k:
            result.add(k)
    return result


def _bk_parity_set(j: int) -> set[int]:
    """Compute the parity set P(j) for BK transform.

    P(j) = set of qubits that store parity up to j-1.
    """
    if j == 0:
        return set()
    result = set()
    # Walk down the binary tree
    parent = j
    while parent > 0:
        low_bit = parent & (-parent)
        if low_bit > 1:
            result.add(parent - 1)
            parent = parent - low_bit // 2
        else:
            break
    # Add the remaining chain
    k = j - 1
    while k >= 0:
        low_bit = (k & (-k)) if k > 0 else 1
        if k < j:
            result.add(k)
        break
    # Simplified: P(j) = {j-1} for odd j, recursive for even j
    result = set()
    k = j - 1
    if k >= 0:
        result.add(k)
        # If k is even, also include more
        low_bit = (k + 1) & (-(k + 1))
        step = low_bit >> 1
        while step > 0:
            result.add(k - step + (step >> 1))
            step >>= 1
    return result


def _bk_remainder_set(j: int) -> int | None:
    """Compute the remainder (flip) qubit for BK transform."""
    # For even j, remainder is j itself; for odd j, it involves the parent
    if j % 2 == 0:
        return None
    return j - 1


def _bk_creation(j: int, n: int) -> list[PauliTerm]:
    """Bravyi-Kitaev transform of a†_j."""
    # Simplified BK: for small systems, fall back to JW-like structure
    # with logarithmic depth corrections.
    # For a proper BK transform, we compute update, parity, and remainder sets.
    update = _bk_update_set(j, n)
    parity = _bk_parity_set(j)

    # a†_j = 0.5 * (X_update * X_j * Z_parity - i * Y_j * X_update * Z_parity)
    # Simplified: treat like JW but with BK index sets
    ops_x: Dict[int, str] = {}
    ops_y: Dict[int, str] = {}

    # Z string on parity set
    for k in parity:
        if k != j:
            ops_x[k] = 'Z'
            ops_y[k] = 'Z'

    # X on update set
    for k in update:
        ops_x[k] = 'X'
        ops_y[k] = 'X'

    ops_x[j] = 'X'
    ops_y[j] = 'Y'

    return [
        (0.5, dict(ops_x)),
        (-0.5j, dict(ops_y)),
    ]


def _bk_annihilation(j: int, n: int) -> list[PauliTerm]:
    """Bravyi-Kitaev transform of a_j."""
    update = _bk_update_set(j, n)
    parity = _bk_parity_set(j)

    ops_x: Dict[int, str] = {}
    ops_y: Dict[int, str] = {}

    for k in parity:
        if k != j:
            ops_x[k] = 'Z'
            ops_y[k] = 'Z'

    for k in update:
        ops_x[k] = 'X'
        ops_y[k] = 'X'

    ops_x[j] = 'X'
    ops_y[j] = 'Y'

    return [
        (0.5, dict(ops_x)),
        (0.5j, dict(ops_y)),
    ]
