The Cosmic Microwave Background
===============================

In 1965 Arno Penzias and Robert Wilson could not get rid of a persistent hiss
in their horn antenna — not after pointing it away from New York City, and
not after evicting the pigeons. The hiss was the cosmic microwave background,
and in the decades since, COBE, WMAP, and Planck have turned that noise into
the most precisely measured spectrum in cosmology. The angular power spectrum
:math:`C_\ell` is usually mined for cosmological parameters; the
:mod:`pyCE.cosmology` module asks a different question of it — how much
information does the spectrum carry, and how is that information distributed
across multipoles?

Loading a spectrum
^^^^^^^^^^^^^^^^^^

The nine-year WMAP TT spectrum ships with the package, so this works
offline::

    from pyCE.cosmology import data

    wmap = data.read_power_spectrum(telescope = 'WMAP', ps = 'TT')

The result is a dictionary with the multipoles ``ell``, the spectra ``Dl``
and ``Cl``, and the measurement ``error``. The Planck release-2 spectra are
downloaded from IRSA on demand — same call, ``telescope = 'Planck'`` — and
the Planck best-fit theory curve comes from ``psType = 'fit'``. Both require
a network connection.

From spectrum to entropy
^^^^^^^^^^^^^^^^^^^^^^^^

The information-theoretic measures live in
:mod:`pyCE.cosmology.analysis.aps`. The modal fraction folds in the
:math:`2\ell+1` degenerate :math:`m`-modes and normalizes, turning the
spectrum into a probability distribution over multipoles; the Shannon entropy
and Kullback–Leibler divergence then do what they always do::

    import numpy as np
    from pyCE.cosmology.analysis import aps

    mf = aps.modal_fraction(wmap['ell'], wmap['Cl'].copy())
    print(aps.entropy(mf.copy()))

Pay attention to the ``.copy()`` calls — they are not decoration. These
functions flag invalid entries by writing NaN into their arguments in place,
so hand them copies whenever you intend to reuse a spectrum.

To compare data against theory, take the divergence between the two modal
fractions::

    planck_fit = data.read_power_spectrum(telescope = 'Planck',
                                          psType = 'fit')
    # interpolate onto the WMAP multipoles before comparing
    Cl_fit = np.interp(wmap['ell'], planck_fit['ell'], planck_fit['Cl'])

    p  = aps.modal_fraction(wmap['ell'], wmap['Cl'].copy())
    q  = aps.modal_fraction(wmap['ell'], Cl_fit.copy())
    print(aps.KL_divergence(p.copy(), q.copy()))

The divergence is measured in bits, and it is not symmetric — :math:`D(p\,\|\,q)`
answers how surprised you are by the data when you believed the theory, which
is the direction cosmology usually cares about.

For spectra on a continuous wavenumber grid — matter power spectra — the
parallel measures live in :mod:`pyCE.cosmology.analysis.mps`, with integrals
in place of sums. And for fitting a noisy spectrum without committing to a
parametric model, :func:`pyCE.cosmology.analysis.aps.nonparametric_fit`
implements the shrinkage estimator of Aghamousa, Arjunwadkar & Souradeep
(arXiv:1107.0516): expand the data in an orthogonal basis, shrink, and let an
unbiased risk estimate choose how much.
