Getting Started
===============

Here you will install pyCE and compute your first configurational entropy. The
library leans on the scientific Python stack - numpy, scipy, matplotlib, and
astropy - so a clean environment now saves you headaches later.

Installation
^^^^^^^^^^^^

pyCE requires Python 3.10 or newer. Clone the repository and install it into a
virtual environment::

    git clone https://github.com/EternalTime/pyCE.git
    cd pyCE
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

The ``-e`` flag installs in editable mode: changes you make to the source are
picked up immediately, with no reinstall. Check that the install works::

    >>> import pyCE

If the import goes through quietly, you're ready.

Your first configurational entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's compute the configurational entropy of a Gaussian in three spatial
dimensions, the most familiar localized profile there is::

    import numpy as np
    from pyCE.math import radialFT, radial_integrate

    r = np.linspace(1e-6, 10, 500)
    f = np.exp(-r**2)

    ft, k = radialFT(3, f, r)        # radial Fourier transform in d = 3
    mf    = np.abs(ft)**2
    mf    = mf/mf.max()              # the modal fraction

    Sc = -radial_integrate(k, mf*np.log(np.finfo(float).eps + mf), 3)
    print(Sc)

Transform, normalize, integrate. Every guide in this documentation is a
variation on that theme, with the field profile coming from a shooting
method, a lattice simulation, a stellar model, or a satellite.

Convince yourself that the transform can be trusted. Plancherel's theorem holds
on the discrete grids by construction::

    print(radial_integrate(k, np.abs(ft)**2, 3)
          / radial_integrate(r, f**2, 3))

You should get 1 to machine precision.
