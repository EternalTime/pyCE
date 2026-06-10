Oscillons
=========

In 1976 Igor Bogolyubsky and Vladimir Makhankov noticed that a collapsing
bubble of scalar field does not always die — it can settle into a localized,
pulsating lump that persists for thousands of oscillations. Marcelo Gleiser
rediscovered these objects in 1994 and christened them oscillons. They are
not protected by any conserved charge; they survive on dynamics alone, which
is what makes their longevity remarkable and their eventual death worth
watching closely. The :mod:`pyCE.oscillons` module lets you watch.

The setup
^^^^^^^^^

The module evolves a spherically symmetric real scalar field in the same
asymmetric double-well used by :mod:`pyCE.instantons` — the Kolb & Turner
parametrization, with asymmetry factor :math:`\epsilon` setting
:math:`\alpha = \tfrac{3}{\sqrt{2}}(1+\epsilon)`. The numerical scheme has
one trick worth understanding before you run it: the outer region of the
radial grid is boosted outward — a monotonically increasingly boosted (MIB)
coordinate system — so that radiation leaving the oscillon is carried off the
lattice instead of reflecting from the boundary and contaminating the core.
The grid mimics an infinite domain at the cost of a finite one. Time stepping
is iterated Crank–Nicolson, with high-order numerical dissipation damping
grid-scale noise.

Running a simulation
^^^^^^^^^^^^^^^^^^^^

First create the environment, then initialize a field, then evolve it::

    import matplotlib.pyplot as plt
    from pyCE.oscillons import oscillon

    env    = oscillon(asymmetry_factor = 0., dimension = 3)
    fields = env.initialize_field('gaussian', 2.9)   # Gaussian of radius 2.9
    env.simulate_oscillon(fields)

    plt.plot(env.time, env.E)
    plt.xlabel('$t$')
    plt.ylabel('$E$')
    plt.show()

A word of warning before you press enter: the simulation runs until 99% of
the initial energy has radiated through the cap radius, and oscillons are
famous for refusing to die. A long-lived configuration on the default lattice
of 1500 points means a long wait. For a first run, pass
``plot_profile = True`` or ``plot_energy_density = True`` to watch the field
live — seeing the initial Gaussian shed its excess energy and ring down into
the oscillon attractor is worth the slowdown once.

The energy history ``env.E``, the core amplitude history ``env.core``, and
the corresponding times ``env.time`` are stored on the environment when the
run completes. The stress-energy components are available at any moment
through ``env.stress_energy_tensor(fields)``, and ``env.radialFT(y)`` gives
you the radial Fourier transform on the interior grid — the first call builds
the transform matrix, so it is slow once and fast thereafter.

Experiment with the initial radius. Not every Gaussian becomes an oscillon —
too narrow and the field disperses promptly, too wide and the collapse
overshoots. Map out the basin of attraction for yourself; keeping a record of
which radii live and which die is exactly the kind of small journal that pays
off later. And once you have a stable of oscillons, compute the
configurational entropy of their energy densities (``env.radialFT`` is built
for it): the entropy predicts which configurations live longest — see
Gleiser, Stephens & Sowinski, *Phys. Rev. D* **97**, 096007 (2018).
