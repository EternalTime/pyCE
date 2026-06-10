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
computes their configurational entropy. The ``bosonstars`` module solves the
Einstein-Klein-Gordon system for self-gravitating complex scalar fields.
The ``oscillons`` module simulates
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
   guide_bosonstars
   guide_oscillons
   guide_polytropes
   guide_cosmology

Reference
^^^^^^^^^

.. toctree::
   :maxdepth: 2

   source/modules
   license

References
^^^^^^^^^^

The configurational entropy program this library serves is developed in the
following papers; the PDFs are hosted on the author's website.

- D. R. Sowinski, *Complexity and Stability for Epistemic Agents: The
  Foundations and Phenomenology of Configurational Entropy*, Ph.D. thesis,
  Dartmouth College (2016).
- M. Gleiser & D. Sowinski, *Information-entropic stability bound for
  compact objects: Application to Q-balls and the Chandrasekhar limit of
  polytropes*,
  `Phys. Lett. B 727, 272 (2013)
  <https://damiansowinski.com/assets/docs/papers/Polytropes_Gleiser_2013.pdf>`_.
- M. Gleiser & D. Sowinski, *Information-entropic signature of the critical
  point*,
  `Phys. Lett. B 747, 125 (2015)
  <https://damiansowinski.com/assets/docs/papers/Critical_Gleiser_2017.pdf>`_.
- D. Sowinski & M. Gleiser, *Information dynamics at a phase transition*,
  `J. Stat. Phys. 167, 1221 (2017)
  <https://damiansowinski.com/assets/docs/papers/PhaseDynamics_Sowinski_2017.pdf>`_.
- D. Sowinski & M. Gleiser, *Configurational information approach to
  instantons and false vacuum decay in D-dimensional spacetime*,
  `Phys. Rev. D 98, 056026 (2018)
  <https://damiansowinski.com/assets/docs/papers/Instantons_Sowinski_2018.pdf>`_.
- M. Gleiser, M. Stephens & D. Sowinski, *Configurational entropy as a
  lifetime predictor and pattern discriminator for oscillons*,
  `Phys. Rev. D 97, 096007 (2018)
  <https://damiansowinski.com/assets/docs/papers/Oscillons_Gleiser_2018.pdf>`_.
- D. R. Sowinski, S. Kelty & G. Ghoshal, *Configurational information
  measures, phase transitions, and an upper bound on complexity*,
  `arXiv:2503.02980 (2025)
  <https://damiansowinski.com/assets/docs/papers/Ising_Sowinski_2025.pdf>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Department of Physics and Astronomy: https://physics.dartmouth.edu/
.. _Dartmouth College: https://home.dartmouth.edu/
.. _Marcelo Gleiser: http://marcelogleiser.com/
