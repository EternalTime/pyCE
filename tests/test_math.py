"""Tests pinning pyCE.math.radialFT to the dense formula it replaced."""

import numpy as np
import pytest
import scipy.special as sp

import pyCE.math
from pyCE.math import radialFT, radial_integrate


def dense_radialFT(d, f, r):
    #the original one-shot formula, building the whole len(k) x len(r) kernel
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    if d > 1:
        a = float(d)/2.0
        ft = np.zeros(np.shape(k))
        ft[1:] = np.sqrt(np.pi)*(2.0**(a-2))*(k[1:]**(1-a))*np.trapezoid(f*sp.jv(a-1, np.outer(k[1:], r))*(r**a), r)
        ft[0] = sum(np.array([287/48.0, -(61/4.0), 1033/48.0, -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])*ft[1:8])
    else:
        ft = np.trapezoid(np.cos(np.outer(k, r))*f, r)
    ft = ft*np.sqrt(radial_integrate(r, np.abs(f)**2, d)/radial_integrate(k, np.abs(ft)**2, d))
    return ft, k

#300 points give 1500 k values: two blocks at the default size, the second partial
@pytest.mark.parametrize('block', [pyCE.math._FT_BLOCK, 7])
@pytest.mark.parametrize('d', [1, 2, 3])
def test_blocked_matches_dense(d, block, monkeypatch):
    monkeypatch.setattr(pyCE.math, '_FT_BLOCK', block)
    r = np.linspace(0, 10, 300)
    f = np.exp(-r**2)*(1 + r)
    ft, k = radialFT(d, f, r)
    ft_ref, k_ref = dense_radialFT(d, f, r)
    assert np.array_equal(k, k_ref)
    assert np.allclose(ft, ft_ref, rtol=1e-12, atol=1e-12*np.abs(ft_ref).max())
