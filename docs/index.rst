pyCE
====

In 2012 Marcelo Gleiser and Nikitas Stamatopoulos asked a deceptively simple
question — how much information does it take to describe a localized field
configuration? — and answered it with a new measure, the configurational
entropy, built from the Shannon entropy of the configuration's power
spectrum. Where the energy of a configuration tells you what it costs to
assemble, its configurational entropy tells you how that cost is organized
across scales. The measure has since been put to work on solitons, compact
stars, phase transitions, and the cosmic microwave background.

pyCE is a Python library for doing this kind of work. It grew out of research
in `Marcelo Gleiser`_'s group in the `Department of Physics and Astronomy`_
at `Dartmouth College`_, and it gathers in one place the machinery those
projects share: radial Fourier transforms in arbitrary spatial dimension,
modal fractions, and entropy and divergence measures for both discrete and
continuous spectra.

The library is organized around four physical arenas. The ``instantons``
module solves for the bounce profiles that mediate false-vacuum decay and
computes their configurational entropy. The ``oscillons`` module simulates
long-lived, localized oscillations of a real scalar field. The
``polytropes`` module solves the Lane-Emden equation for self-gravitating
spheres of gas. The ``cosmology`` module brings the same
information-theoretic lens to the angular power spectrum of the cosmic
microwave background — WMAP data is bundled, Planck data is a download away.
Underneath all four sits ``pyCE.math``, where the radial Fourier transform
lives.

If you're new here, start with :doc:`getting_started`, then work through
whichever guide matches your problem.

Guide
^^^^^

.. toctree::
   :maxdepth: 1

   getting_started
   guide_instantons
   guide_oscillons
   guide_polytropes
   guide_cosmology

Reference
^^^^^^^^^

.. toctree::
   :maxdepth: 2

   source/modules
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Department of Physics and Astronomy: https://physics.dartmouth.edu/
.. _Dartmouth College: https://home.dartmouth.edu/
.. _Marcelo Gleiser: http://marcelogleiser.com/
