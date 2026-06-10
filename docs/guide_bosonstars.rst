Boson Stars
===========

In 1968 David Kaup asked what happens when a complex scalar field is left
alone with its own gravity, and found the answer is a star: a localized,
stationary, horizonless solution of the Einstein-Klein-Gordon system. Remo
Ruffini and Silvano Bonazzola reached the same configurations a year later
from the quantum side, as the ground state of a cold cloud of gravitating
bosons. Unlike a fermion star, nothing here is held up by degeneracy
pressure — the star is supported by the uncertainty principle wearing a
classical disguise, the field's intrinsic gradient energy resisting
collapse. Whether dark matter builds such objects remains an open question,
which is precisely what keeps them interesting.

The model
^^^^^^^^^

The field carries a conserved charge, so a stationary star is not a static
field but a rotating phase: the harmonic ansatz
:math:`\phi(r,t) = \phi(r)e^{-i\omega t}` puts all the time dependence in a
phase the stress-energy never sees. In the polar-areal metric

.. math::

    ds^2 = -\alpha(r)^2 dt^2 + a(r)^2 dr^2 + r^2 d\Omega^2,

the Einstein and Klein-Gordon equations reduce to a first-order ODE system
for :math:`(a, \alpha, \phi, \Phi \equiv \phi')`, with the potential
:math:`V = |\phi|^2 + \tfrac{\lambda}{2}|\phi|^4`, and the frequency
:math:`\omega` is an eigenvalue: only discrete central-lapse values give a
profile that decays into flat space, and the nodeless one is the ground
state. The :class:`pyCE.bosonstars.BosonStar` class finds it by bisection
shooting on the central lapse, exactly as :mod:`pyCE.instantons` shoots on
the core field value — the same numerical instinct, pointed at a different
equation.

Solving a star
^^^^^^^^^^^^^^

Everything happens at instantiation::

    import matplotlib.pyplot as plt
    from pyCE.bosonstars import BosonStar

    star = BosonStar(0.1)      # core field value phi0 = 0.1

    plt.plot(star.r, star.phi)
    plt.xlabel('$r$')
    plt.ylabel(r'$\phi(r)$')
    plt.show()

    print(star.M, star.R, star.omega)

For :math:`\phi_0 = 0.1` you should find M = 0.533 (in units of
:math:`M_{\rm Pl}^2/m`), R = 6.0, and :math:`\omega = 0.937` — a bound
state, since :math:`\omega < 1` means the star sits below the free-particle
mass threshold. The metric functions (``star.a``, ``star.alpha``), the
energy density (``star.rho``), the Misner-Sharp mass function
(``star.mass``), and the configurational entropy machinery (``star.mf``,
``star.k``, ``star.Sc``) all hang off the object. Pass ``lamb`` to switch
on the quartic self-interaction — Monica Colpi, Stuart Shapiro, and Ira
Wasserman showed in 1986 that it inflates these stars from curiosities to
astrophysical contenders. Pass ``plot_progress = True`` to watch the
shooting method work; the failed shots tell you as much about the
eigenvalue problem as the successful one does.

The mass curve
^^^^^^^^^^^^^^

The defining plot of the subject is the mass against the core value::

    import numpy as np

    phi0s = np.linspace(0.05, 0.45, 9)
    stars = [BosonStar(p) for p in phi0s]

    plt.plot(phi0s, [s.M for s in stars], 'o-')
    plt.xlabel(r'$\phi_0$')
    plt.ylabel('$M$')
    plt.show()

The curve rises, peaks at the Kaup limit — M = 0.633 near
:math:`\phi_0 = 0.27` — and turns over; configurations past the peak are
unstable to collapse, in close analogy with the white-dwarf story. Now make
the information-theoretic version of the plot: ``[s.Sc for s in stars]``
against :math:`\phi_0`. How does the configurational entropy behave as the
family approaches the stability boundary? That question is exactly the kind
this library was built to ask — see Gleiser & Sowinski, *Phys. Lett. B*
**727**, 272 (2013) before peeking at the answer.
