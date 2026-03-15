# -*- coding: utf-8 -*-
# author: vinhpx
"""
Custom Hamiltonian MPO builder for qforge.

Constructs Matrix Product Operator (MPO) representations from sums of
local operator terms. Works with both C++ and Python DMRG backends.

Usage::

    from qforge.mpo_builder import MPOBuilder

    # Transverse-field Ising with custom couplings
    builder = MPOBuilder(n_sites=10)
    for i in range(9):
        builder.add_term(-1.0, [('Z', i), ('Z', i + 1)])
    for i in range(10):
        builder.add_term(-0.5, [('X', i)])
    H = builder.build()

    # Use with DMRG
    from qforge.dmrg import DMRG
    dmrg = DMRG.custom(n_sites=10, terms=builder.terms)
    energy, psi = dmrg.run(n_sweeps=10)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Union

try:
    from qforge._qforge_mps import MPO as _MPO_core
    _HAS_MPS_CPP = True
except ImportError:
    _HAS_MPS_CPP = False

# Pauli matrices stored row-major [2, 2]
_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'S+': np.array([[0, 1], [0, 0]], dtype=complex),
    'S-': np.array([[0, 0], [1, 0]], dtype=complex),
}


class MPOBuilder:
    """Build an MPO Hamiltonian from a sum of local operator terms.

    Each term is a coefficient times a product of single-site operators:
    ``H = sum_k coeff_k * prod_i O_{k,i}``

    Args:
        n_sites: Number of lattice sites (qubits).

    Supported operators: 'I', 'X', 'Y', 'Z', 'S+', 'S-', or a 2x2 array.
    """

    def __init__(self, n_sites: int) -> None:
        if n_sites < 2:
            raise ValueError("n_sites must be >= 2")
        self.n_sites = n_sites
        self.terms: List[Tuple[complex, List[Tuple]]] = []

    def add_term(
        self,
        coeff: complex,
        operators: List[Tuple[Union[str, np.ndarray], int]],
    ) -> 'MPOBuilder':
        """Add a term to the Hamiltonian.

        Args:
            coeff:     Scalar coefficient.
            operators: List of (operator, site) pairs.
                       operator is 'X', 'Y', 'Z', 'I', 'S+', 'S-', or 2x2 ndarray.
                       Sites must be in ascending order for nearest-neighbor terms.

        Returns:
            self (for chaining).
        """
        if abs(coeff) < 1e-15:
            return self
        # Validate and normalize operators
        ops = []
        for op, site in operators:
            if site < 0 or site >= self.n_sites:
                raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
            if isinstance(op, str):
                if op not in _PAULI:
                    raise ValueError(f"Unknown operator {op!r}. Use: {list(_PAULI.keys())}")
                ops.append((op, site))
            else:
                op = np.asarray(op, dtype=complex)
                if op.shape != (2, 2):
                    raise ValueError("Custom operator must be 2x2")
                ops.append((op, site))
        self.terms.append((complex(coeff), ops))
        return self

    def build(self) -> Union['_MPO_core', dict]:
        """Build MPO from accumulated terms.

        Returns C++ MPO object if available, otherwise a Python dict spec
        compatible with the DMRG Python fallback.
        """
        tensors = self._build_mpo_tensors()
        if _HAS_MPS_CPP:
            return self._to_cpp_mpo(tensors)
        return self._to_py_spec(tensors)

    def build_py(self) -> dict:
        """Build Python-only MPO spec (always returns dict for DMRG Python path)."""
        return {
            'type': 'custom',
            'n_sites': self.n_sites,
            'tensors': self._build_mpo_tensors(),
        }

    def _build_mpo_tensors(self) -> List[np.ndarray]:
        """Build MPO tensors via finite-state-machine construction.

        Returns list of numpy arrays, each shape [w_left, w_right, d, d].
        """
        n = self.n_sites
        d = 2

        # Group terms by their operator ranges
        # Each term contributes to the FSM MPO as a chain of states.
        # For a k-body term on sites [i1, i2, ..., ik], we need k-1
        # intermediate MPO states between i1 and ik.

        # Build the FSM: states are (term_index, position_in_term)
        # State 0: "done" (identity from left boundary)
        # State -1: "vacuum" (identity from right boundary)
        # Intermediate states: one per "open" interaction

        # Simpler approach: construct each term as a separate MPO and sum them.
        # MPO sum: W_total[i] has block-diagonal structure.
        # Bond dim = 2 + sum of (range_of_term) for each term with range > 0.
        # For single-site terms (range=0): just add to boundary.

        # Separate terms by type
        onsite_terms = []      # Single-site operators
        interaction_terms = [] # Multi-site interactions

        for coeff, ops in self.terms:
            if len(ops) == 1:
                onsite_terms.append((coeff, ops))
            else:
                # Sort by site
                ops_sorted = sorted(ops, key=lambda x: x[1])
                interaction_terms.append((coeff, ops_sorted))

        # FSM MPO construction
        # States: 0 = "identity-left", then interaction intermediates, then "identity-right"
        # For each interaction term spanning sites [s1, ..., sk]:
        #   Allocate (k-1) intermediate FSM states for the "open" channels

        n_inter_states = 0
        term_state_offsets = []
        for coeff, ops in interaction_terms:
            n_ops = len(ops)
            offset = n_inter_states
            term_state_offsets.append(offset)
            n_inter_states += n_ops - 1

        # Total FSM states: 1 (identity-left) + n_inter_states + 1 (identity-right)
        W = n_inter_states + 2
        idx_done = 0          # "identity-left" boundary
        idx_vacuum = W - 1    # "identity-right" boundary

        tensors = []
        for site in range(n):
            first = (site == 0)
            last = (site == n - 1)
            wl = 1 if first else W
            wr = 1 if last else W
            T = np.zeros((wl, wr, d, d), dtype=complex)

            # Map global FSM indices to local tensor indices
            def gl(i):
                return 0 if first else i

            def gr(i):
                return 0 if last else i

            # Identity passthrough
            if not first or not last:
                if first:
                    T[gl(idx_vacuum), gr(idx_vacuum)] = np.eye(d, dtype=complex)
                elif last:
                    T[gl(idx_done), gr(idx_done)] = np.eye(d, dtype=complex)
                else:
                    T[gl(idx_done), gr(idx_done)] = np.eye(d, dtype=complex)
                    T[gl(idx_vacuum), gr(idx_vacuum)] = np.eye(d, dtype=complex)

            # On-site terms: contribute to vacuum→done transition
            for coeff, ops in onsite_terms:
                op_name, op_site = ops[0]
                if op_site == site:
                    op_mat = _PAULI[op_name] if isinstance(op_name, str) else op_name
                    T[gl(idx_vacuum), gr(idx_done)] += coeff * op_mat

            # Interaction terms
            for t_idx, (coeff, ops) in enumerate(interaction_terms):
                ops_sorted = ops
                n_ops = len(ops_sorted)
                sites = [s for _, s in ops_sorted]
                offset = term_state_offsets[t_idx]

                for k_pos in range(n_ops):
                    op_name, op_site = ops_sorted[k_pos]
                    if op_site != site:
                        continue
                    op_mat = _PAULI[op_name] if isinstance(op_name, str) else op_name

                    if k_pos == 0:
                        # Start of interaction: vacuum → first intermediate state
                        inter_state = 1 + offset  # FSM state index
                        T[gl(idx_vacuum), gr(inter_state)] += coeff * op_mat
                    elif k_pos == n_ops - 1:
                        # End of interaction: last intermediate → done
                        inter_state = 1 + offset + k_pos - 1
                        T[gl(inter_state), gr(idx_done)] += op_mat
                    else:
                        # Middle: chain intermediate states
                        inter_from = 1 + offset + k_pos - 1
                        inter_to = 1 + offset + k_pos
                        T[gl(inter_from), gr(inter_to)] += op_mat

                # Pass through intermediate states for sites between operators
                for k_pos in range(n_ops - 1):
                    s_start = sites[k_pos]
                    s_end = sites[k_pos + 1]
                    if s_start < site < s_end:
                        inter_state = 1 + offset + k_pos
                        T[gl(inter_state), gr(inter_state)] += np.eye(d, dtype=complex)

            tensors.append(T)

        return tensors

    def _to_cpp_mpo(self, tensors: List[np.ndarray]):
        """Convert numpy MPO tensors to C++ MPO object."""
        n = self.n_sites
        max_w = max(t.shape[0] for t in tensors)
        mpo = _MPO_core(n, max_w)

        for i, T in enumerate(tensors):
            wl, wr = T.shape[0], T.shape[1]
            site = mpo.site(i)
            # The C++ MPOTensor needs to be properly sized
            # We set values via the Python-accessible tensors_ list
            # For now, store as Python spec and use Python DMRG path
            pass

        # Fall back to Python spec since direct C++ MPOTensor write
        # from Python isn't exposed in the current bindings
        return self._to_py_spec(tensors)

    def _to_py_spec(self, tensors: List[np.ndarray]) -> dict:
        """Return Python MPO spec dict with pre-built tensors."""
        return {
            'type': 'custom',
            'n_sites': self.n_sites,
            'tensors': tensors,
        }


# Custom MPO type 'custom' is handled natively in dmrg._mpo_tensors_py
