"""CMB data access and information-theoretic analysis.

Submodules
----------
data : load WMAP (bundled) and Planck (downloaded) angular power spectra
analysis : entropy and divergence measures for angular and matter power
    spectra, and nonparametric spectrum fitting
"""
from . import data
from . import analysis
