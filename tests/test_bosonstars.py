"""Tests pinning pyCE.bosonstars to known physics and past bugs."""

import numpy as np
import pytest

from pyCE.bosonstars import BosonStar


@pytest.fixture(scope='module')
def star01():
    return BosonStar(0.1)

def test_mini_boson_star_mass(star01):
    #reference value from the original BosonStars solver, phi0 = 0.1
    assert abs(star01.M - 0.5326) < 1e-3

def test_bound_state_frequency(star01):
    #a gravitationally bound star oscillates below the mass threshold
    assert 0 < star01.omega < 1

def test_physical_sanity(star01):
    assert (star01.rho >= 0).all()
    assert star01.R > 0
    assert np.isfinite(star01.Sc) and star01.Sc > 0
    #mass function rises monotonically to the ADM mass at the edge
    assert star01.M == star01.mass[-1] and star01.M > 0

#coarse, fast parameters: the bug regressions need repeatability, not precision
FAST = dict(dr=.02, r_max=25)

def test_bracket_not_corrupted_between_stars():
    #regression: the original solver mutated its default alpha_range,
    #so every star after the first inherited a stale bracket
    fresh = BosonStar(0.2, **FAST).M
    BosonStar(0.1, **FAST)
    again = BosonStar(0.2, **FAST).M
    assert abs(again - fresh) < 1e-10

def test_callers_list_untouched():
    bracket = [0, 1]
    BosonStar(0.1, alpha_range=bracket, **FAST)
    assert bracket == [0, 1]
