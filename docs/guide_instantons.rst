Instantons
==========

In 1977 Sidney Coleman worked out the fate of the false vacuum: a metastable
state decays by nucleating bubbles of true vacuum, and the bubble that
controls the decay rate — the bounce — is an O(d)-symmetric solution of the
Euclidean equations of motion. The :mod:`pyCE.instantons` module finds these
bounces and asks an information-theoretic question of them: how is the energy
of a bounce organized across scales, and how does that organization respond
as the potential is deformed?

The model
^^^^^^^^^

The field lives in the asymmetric double-well potential

.. math::

    V(B) = \frac{1}{2}B^2 - \frac{\alpha}{3}B^3 + \frac{1}{4}B^4,
    \qquad \alpha = \frac{3}{\sqrt{2}}\,(1+g),

where the asymmetry factor :math:`g` controls how far the two vacua are split
— at :math:`g=0` they are degenerate, and the bubble wall becomes thin. The
bounce satisfies

.. math::

    B'' + \frac{d}{r}B' = V'(B),

with :math:`B'(0)=0` and :math:`B\to 0` as :math:`r\to\infty`. This is a
boundary value problem with an unstable answer: start the field a little too
high at the origin and it overshoots, a little too low and it rolls back into
the false vacuum. The module turns this instability into an algorithm — the
shooting method — bisecting on the core value :math:`B(0)` until the profile
decays cleanly into the false vacuum.

Solving a bounce
^^^^^^^^^^^^^^^^

Everything happens at instantiation::

    import matplotlib.pyplot as plt
    from pyCE.instantons import instanton

    inst = instanton(0.2, 3, 3000)   # g = 0.2, d = 3, up to 3000 radial steps

    plt.plot(inst.r, inst.B)
    plt.xlabel('$r$')
    plt.ylabel('$B(r)$')
    plt.show()

The solve takes a few seconds — each bisection step integrates the profile
out with RK4, and overshooting runs are allowed to diverge before being
discarded. Don't be alarmed by overflow warnings during the solve; diverging
is precisely how a failed shot announces itself.

Once the object exists, the physics is sitting on its attributes: the profile
and its derivative (``inst.B``, ``inst.DB``), the potential, gradient, and
total energy densities (``inst.PEdens``, ``inst.GEdens``, ``inst.rho``), the
Euclidean action ``inst.Se``, and the configurational entropy ``inst.Sc``,
computed from the modal fraction of the energy-density power spectrum
(``inst.mf`` on the grid ``inst.k``).

Try sweeping the asymmetry::

    import numpy as np

    gs  = np.linspace(0.05, 0.5, 10)
    Scs = [instanton(g, 3, 3000).Sc for g in gs]

    plt.plot(gs, Scs, 'o-')
    plt.xlabel('$g$')
    plt.ylabel('$S_c$')
    plt.show()

Convince yourself of the trend before you plot it: as :math:`g` grows the
bounce localizes, its power spectrum spreads, and the configurational entropy
responds. For the relationship between :math:`S_c` and the Euclidean action —
and what it suggests about decay rates — see Gleiser & Sowinski, *Phys. Rev.
D* **98**, 056026 (2018).
