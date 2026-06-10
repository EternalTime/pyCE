"""CMB data access and information-theoretic analysis.

Submodules
----------
data
    Load WMAP (bundled) and Planck (downloaded) angular power spectra.
analysis
    Entropy and divergence measures for power spectra, and nonparametric
    spectrum fitting.
"""
from . import data
from . import analysis
