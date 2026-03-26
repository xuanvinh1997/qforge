# -*- coding: utf-8 -*-
"""Tests for the centralized backend selection system."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import qforge
from qforge.circuit import Qubit, Walk_Qubit
from qforge.mps import MatrixProductState


@pytest.fixture(autouse=True)
def reset_backend():
    """Reset global backend to 'auto' after each test."""
    yield
    qforge.set_backend('auto')


# ================================================================
# backend_info
# ================================================================

class TestBackendInfo:
    def test_returns_dict(self):
        info = qforge.backend_info()
        assert isinstance(info, dict)
        assert 'default' in info
        assert 'resolved' in info
        assert 'available' in info

    def test_available_has_all_keys(self):
        avail = qforge.backend_info()['available']
        for key in ('metal', 'cuda', 'cpu', 'mps_cpp', 'distributed', 'python'):
            assert key in avail

    def test_python_always_available(self):
        assert qforge.backend_info()['available']['python'] is True

    def test_resolved_is_concrete(self):
        resolved = qforge.backend_info()['resolved']
        assert resolved != 'auto'
        assert resolved in ('metal', 'cuda', 'cpu', 'python')


# ================================================================
# set_backend / get_backend
# ================================================================

class TestSetGetBackend:
    def test_default_is_auto(self):
        qforge.set_backend('auto')
        assert qforge.get_backend() == 'auto'

    def test_set_cpu(self):
        if not qforge._HAS_CPP:
            pytest.skip("C++ backend not available")
        qforge.set_backend('cpu')
        assert qforge.get_backend() == 'cpu'

    def test_set_metal(self):
        if not qforge._HAS_METAL:
            pytest.skip("Metal backend not available")
        qforge.set_backend('metal')
        assert qforge.get_backend() == 'metal'

    def test_set_python(self):
        qforge.set_backend('python')
        assert qforge.get_backend() == 'python'

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            qforge.set_backend('gpu')

    def test_unavailable_cuda_raises(self):
        if qforge._HAS_CUDA:
            pytest.skip("CUDA is available on this machine")
        with pytest.raises(ValueError, match="CUDA"):
            qforge.set_backend('cuda')

    def test_unavailable_distributed_raises(self):
        if qforge._HAS_DISTRIBUTED:
            pytest.skip("Distributed is available on this machine")
        with pytest.raises(ValueError, match="Distributed"):
            qforge.set_backend('distributed')


# ================================================================
# Global default affects Qubit()
# ================================================================

class TestGlobalDefault:
    def test_auto_resolves(self):
        qforge.set_backend('auto')
        wf = Qubit(2)
        assert wf.backend in ('metal', 'cuda', 'cpu', 'python')

    def test_force_cpu(self):
        if not qforge._HAS_CPP:
            pytest.skip("C++ backend not available")
        qforge.set_backend('cpu')
        wf = Qubit(2)
        assert wf.backend == 'cpu'
        assert wf._sv is not None

    def test_force_python(self):
        qforge.set_backend('python')
        wf = Qubit(2)
        assert wf.backend == 'python'
        assert wf._sv is None

    def test_explicit_overrides_global(self):
        qforge.set_backend('python')
        if not qforge._HAS_CPP:
            pytest.skip("C++ backend not available")
        wf = Qubit(2, backend='cpu')
        assert wf.backend == 'cpu'
        assert wf._sv is not None


# ================================================================
# Wavefunction.backend property
# ================================================================

class TestWavefunctionBackend:
    def test_python_backend(self):
        wf = Qubit(2, backend='python')
        assert wf.backend == 'python'

    def test_cpu_backend(self):
        if not qforge._HAS_CPP:
            pytest.skip("C++ backend not available")
        wf = Qubit(2, backend='cpu')
        assert wf.backend == 'cpu'

    def test_metal_backend(self):
        if not qforge._HAS_METAL:
            pytest.skip("Metal backend not available")
        wf = Qubit(2, backend='metal')
        assert wf.backend == 'metal'


# ================================================================
# MatrixProductState.backend property
# ================================================================

class TestMPSBackend:
    def test_auto_resolves(self):
        psi = MatrixProductState(4, max_bond_dim=4)
        assert psi.backend in ('cpu', 'python')

    def test_force_python(self):
        psi = MatrixProductState(4, max_bond_dim=4, backend='python')
        assert psi.backend == 'python'
        assert psi._mps is None
        assert psi._tensors is not None

    def test_force_cpu(self):
        from qforge.mps import _HAS_MPS_CPP
        if not _HAS_MPS_CPP:
            pytest.skip("C++ MPS not available")
        psi = MatrixProductState(4, max_bond_dim=4, backend='cpu')
        assert psi.backend == 'cpu'
        assert psi._mps is not None


# ================================================================
# Walk_Qubit backend tag
# ================================================================

class TestWalkQubit:
    def test_walk_qubit_backend_is_python(self):
        wf = Walk_Qubit(qubit_num=3, dim=1)
        assert wf.backend == 'python'

    def test_walk_qubit_2d_backend_is_python(self):
        wf = Walk_Qubit(qubit_num=2, dim=2)
        assert wf.backend == 'python'


# ================================================================
# Cross-backend correctness
# ================================================================

class TestCrossBackend:
    def test_bell_state_cpu_vs_python(self):
        if not qforge._HAS_CPP:
            pytest.skip("C++ backend not available")
        from qforge import gates
        wf_cpu = Qubit(2, backend='cpu')
        wf_py = Qubit(2, backend='python')
        gates.H(wf_cpu, 0); gates.H(wf_py, 0)
        gates.CNOT(wf_cpu, 0, 1); gates.CNOT(wf_py, 0, 1)
        assert np.allclose(wf_cpu.amplitude, wf_py.amplitude, atol=1e-10)

    def test_bell_state_metal_vs_cpu(self):
        if not qforge._HAS_METAL or not qforge._HAS_CPP:
            pytest.skip("Need both Metal and CPU backends")
        from qforge import gates
        wf_m = Qubit(2, backend='metal')
        wf_c = Qubit(2, backend='cpu')
        gates.H(wf_m, 0); gates.H(wf_c, 0)
        gates.CNOT(wf_m, 0, 1); gates.CNOT(wf_c, 0, 1)
        assert np.allclose(wf_m.amplitude, wf_c.amplitude, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
