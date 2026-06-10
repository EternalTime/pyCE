Polytropes
==========

In 1870 Jonathan Homer Lane, an American astrophysicist working largely
alone, asked what the inside of the Sun must look like if it were a
self-gravitating sphere of gas in hydrostatic equilibrium. Robert Emden
systematized the answer in his 1907 *Gaskugeln*, and the equation that bears
both their names has been a workhorse of stellar structure ever since —
simple enough to solve on a laptop in milliseconds, rich enough that
Chandrasekhar built the white-dwarf mass limit on top of it.

The model
^^^^^^^^^

A polytrope is a fluid with the equation of state
:math:`P = K\rho^{\gamma}`, :math:`\gamma = 1 + 1/n`, where :math:`n` is the
polytropic index. In scaled variables the structure reduces to the
Lane–Emden equation

.. math::

    \theta'' + \frac{2}{r}\theta' = -\theta^n,
    \qquad \theta(0) = 1,\quad \theta'(0) = 0,

with the density given by :math:`\rho = \theta^n` and the stellar surface
sitting at the first zero of :math:`\theta`. Exact solutions exist for
:math:`n = 0`, 1, and 5; everything else is numerics, and
:mod:`pyCE.polytropes` handles the numerics with an RK4 integration that
stops at the surface.

Solving a polytrope
^^^^^^^^^^^^^^^^^^^

::

    import matplotlib.pyplot as plt
    from pyCE.polytropes import polytrope

    p = polytrope(n = 1.5)    # a fully convective star

    plt.plot(p.r, p.rho)
    plt.xlabel('$r$')
    plt.ylabel(r'$\rho/\rho_c$')
    plt.show()

The profile :math:`\theta` is on ``p.theta``, its derivative on ``p.psi``,
the scaled density on ``p.rho``, and the radial grid on ``p.r`` — so the
surface radius is simply ``p.r[-1]``. For :math:`n = 1.5` you should find it
at 3.65, within a grid spacing of the literature value 3.6538; check it. Then
convince yourself the solver is honest by comparing
the :math:`n = 1` output against the exact solution
:math:`\theta = \sin(r)/r`.

Two warnings from the structure of the equation itself. First, the index
:math:`n = 5` marks the boundary of compactness — the famous Schuster–Emden
solution has finite mass but infinite radius — so the surface-finding loop
will never terminate there. Stick to :math:`n < 5` unless you have somewhere
to be tomorrow. Second, the integrator steps slightly past the surface before
stopping, where :math:`\theta < 0` raises a harmless warning for non-integer
:math:`n`; the overshoot is truncated from the returned profile.

With :math:`\theta` in hand, the configurational entropy of a stellar density
profile is the same three-step computation as everywhere else in this library
— transform ``p.rho`` with :func:`pyCE.math.radialFT`, normalize, and
integrate. Compare :math:`S_c` across the index family and you have
reproduced the heart of the configurational-entropy approach to compact
objects: in Gleiser & Sowinski, *Phys. Lett. B* **727**, 272 (2013), this
calculation turns into an information-entropic bound on stability, with the
Chandrasekhar limit emerging from the entropy rather than the pressure.
