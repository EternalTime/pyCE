"""pyCE: a configurational entropy library.

Configurational entropy (CE) is an information-theoretic measure of the
spatial complexity of a field configuration, built from the Shannon entropy
of the normalized power spectrum (the "modal fraction") of the
configuration.

Modules
-------
math : radial Fourier transforms and d-dimensional radial integration
cosmology : analysis of the Cosmic Microwave Background
instantons : generating and analyzing instantons
bosonstars : generating and analyzing boson stars
oscillons : generating and analyzing oscillons
polytropes : generating and analyzing polytropic models of stars
"""
def docs():
    """Open the online pyCE documentation in a web browser."""
    import webbrowser
    webbrowser.open('https://damiansowinski.com/pyCE/')

from . import cosmology
from . import oscillons
from . import instantons
from . import bosonstars
from . import polytropes
from . import math
