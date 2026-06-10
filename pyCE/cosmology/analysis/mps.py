"""Information-theoretic analysis of matter power spectra.

Continuum (integral) analogues of the discrete measures in
:mod:`pyCE.cosmology.analysis.aps`: modal fractions, differential Shannon
entropy, and Kullback-Leibler divergence for power spectra defined on a
continuous wavenumber grid, with the d-dimensional Jacobian factor x**(d-1)
included in the measure. All integrals use the trapezoidal rule.
"""

import numpy as np

#----------------------------------------------------------------------FUNCTIONS

def norm(x,y,d = 3):
    """Normalize y over x with the d-dimensional radial measure.

    Parameters
    ----------
    x : ndarray
        Wavenumber grid.
    y : ndarray
        Spectrum sampled on `x`.
    d : int, optional
        Number of spatial dimensions (default 3).

    Returns
    -------
    ndarray
        y divided by Integral[ y * x**(d-1) dx ].
    """
    return y/np.trapz(y*x**(d-1),x)

def modal_fraction(x,y,d = 3,xmax = np.inf,xmin = 0):
    """Modal fraction of a power spectrum over a restricted k-range.

    Restricts to the modes with xmin <= x <= xmax and y > 0, normalizes the
    spectrum there with the d-dimensional measure, and attaches the Jacobian
    factor x**(d-1) so the result is a probability density in x.

    Parameters
    ----------
    x : ndarray
        Wavenumber grid.
    y : ndarray
        Power spectrum sampled on `x`.
    d : int, optional
        Number of spatial dimensions (default 3).
    xmax, xmin : float, optional
        Bounds of the valid k-range.

    Returns
    -------
    x : ndarray
        The valid wavenumbers.
    mf : ndarray
        The modal fraction on those wavenumbers.
    """
    #finds the valid k-modes
    idx = (x>=xmin)&(x<=xmax)&(y>0)
    x = x[idx]
    y = y[idx]
    #normalize the power spectrum
    mf = norm(x,y,d = d)
    return x,mf*x**(d - 1)

def KL_divergence(x,p,q,xmin = 0,xmax = np.inf):
    """Kullback-Leibler divergence D(p || q) in bits.

    Note that the divergence is not symmetric in its arguments. Only modes
    with xmin <= x <= xmax where both p and q are positive contribute.

    Parameters
    ----------
    x : ndarray
        Wavenumber grid.
    p, q : ndarray
        Modal fractions sampled on `x`.
    xmin, xmax : float, optional
        Bounds of the valid k-range.

    Returns
    -------
    float
        Integral[ p * log2(p/q) dx ] over the valid range.
    """
    #finds the valid k-modes
    idx = (x>=xmin)&(x<=xmax)&(p>0)&(q>0)
    x,p,q = x[idx],p[idx],q[idx]
    return np.trapz(p*np.log2(p/q),x)

def entropy(x,p):
    """Differential Shannon entropy of a modal fraction, in bits.

    Parameters
    ----------
    x : ndarray
        Wavenumber grid.
    p : ndarray
        Modal fraction sampled on `x`.

    Returns
    -------
    float
        -Integral[ p * log2(p) dx ].
    """
    return -np.trapz(p*np.log2(p),x)
