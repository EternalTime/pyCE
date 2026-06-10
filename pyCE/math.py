"""Radial Fourier analysis in d spatial dimensions.

Helper routines shared by the physics modules of pyCE. A function f(r) that
depends only on the radial coordinate in d spatial dimensions has a Fourier
transform that is itself radial in k-space; the angular integrals can be done
analytically, reducing the d-dimensional transform to a one-dimensional
integral against a Bessel kernel,

    ft(k) = sqrt(pi) * 2**(d/2-2) * k**(1-d/2)
            * Integral[ f(r) * J_{d/2-1}(k r) * r**(d/2) dr ].

This module provides that transform (as a function and as a precomputed
matrix), together with d-dimensional radial integration and normalization
utilities.
"""

import numpy as np
import scipy.special as sp

def radialFT(d,f,r):
    """Radial Fourier transform of a radial function in d dimensions.

    Parameters
    ----------
    d : int
        Number of spatial dimensions.
    f : ndarray
        Field values f(r) sampled on the radial grid `r`.
    r : ndarray
        Radial grid. Assumed to start near 0 and be (close to) uniform.

    Returns
    -------
    ft : ndarray
        Radial Fourier transform evaluated on the returned k-grid.
    k : ndarray
        Conjugate radial grid in k-space. It has 5x as many points as `r`
        with spacing pi/(10*r[-1]), i.e. it resolves down to half the
        fundamental mode of the box and extends to 5x the usual band limit.

    Notes
    -----
    For d > 1 the angular integrals are done analytically, leaving the
    one-dimensional Bessel integral quoted in the module docstring, which is
    evaluated with the trapezoidal rule. The kernel k**(1-d/2) is singular at
    k = 0, so ft[0] is instead obtained by extrapolating ft[1:8] with a
    seventh-order one-sided finite-difference (interpolation) stencil. For
    d = 1 the transform reduces to a plain cosine transform.

    The result is rescaled by a constant so that Plancherel's theorem,
    ||f||^2 = ||ft||^2 (with the d-dimensional radial measure), holds exactly
    on the discrete grids.
    """
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    if d>1:
        a      = float(d)/2.0
        ft     = np.zeros(np.shape(k))
        ft[1:] = np.sqrt(np.pi)*(2.0**(a-2))*(k[1:]**(1-a))*np.trapz(f*sp.jv(a-1,np.outer(k[1:],r))*(r**a),r)
        #This uses finite differences to get the value at k=0
        ft[0]  = sum(np.array([287/48.0, -(61/4.0), 1033/48.0, -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])*ft[1:8])
    else:
        ft = np.trapz(np.cos(np.outer(k,r))*f,r)
    #normalizes to ensure Plancheral's theorem holds
    ft = ft*np.sqrt(radial_integrate(r,np.abs(f)**2,d)/radial_integrate(k,np.abs(ft)**2,d))
    return ft,k

def radialFT_mat(d,r):
    """Matrix form of the radial Fourier transform.

    Builds the dense matrix F such that ``ft = F @ f`` approximates the
    transform computed by :func:`radialFT` (without the final Plancherel
    rescaling, which depends on f). Useful when many transforms are needed on
    the same grid, e.g. at every output step of a simulation.

    Parameters
    ----------
    d : int
        Number of spatial dimensions.
    r : ndarray
        Radial grid the matrix will act on.

    Returns
    -------
    F : ndarray, shape (5*len(r), len(r))
        Transform matrix. Trapezoidal weights (the half-weight on the first
        point and the grid spacings) are folded into the columns, and the
        k = 0 row implements the same seven-point extrapolation used in
        :func:`radialFT`.
    k : ndarray
        Conjugate radial grid in k-space (same convention as
        :func:`radialFT`).
    """
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    F = np.zeros([len(k),len(r)])
    dr0 = [np.diff(r)[0]/2]
    if d>1:
        a           = float(d)/2.0
        #Bessel kernel with the integration weights baked into each column
        F[1:,:]     = (np.sqrt(np.pi)*(2.0**(a-2))
                        *np.tile(k[1:]**(1-a),(len(r),1)).T
                        *(sp.jv(a-1,np.outer(k[1:],r))*(r**a))
                        *np.tile(np.array(dr0+list(np.diff(r))),(len(k)-1,1)))
        #F0 patches the singular k=0 row with the extrapolation stencil
        F0          = np.eye(len(k))
        F0[0,0:8]   = np.array([0, 287/48.0, -(61/4.0), 1033/48.0,
                                -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])
        F = np.dot(F0,F)
    else:
        F = np.cos(np.outer(k,r))
    return F,k

def sphere_solid_angle(d):
    """Total solid angle of the unit (d-1)-sphere, 2*pi**(d/2)/Gamma(d/2).

    E.g. 2 for d = 1, 2*pi for d = 2, 4*pi for d = 3.
    """
    return 2*np.pi**(d/2.0)/sp.gamma(d/2.0)

def fourier_factor(d):
    """Symmetric Fourier normalization constant, (2*pi)**(-d/2)."""
    return 1.0/(2*np.pi)**(d/2.0)

def radial_integrate(r,y,d):
    """Integrate a radial function over all of d-dimensional space.

    Computes Omega_d * Integral[ y(r) * r**(d-1) dr ] with the trapezoidal
    rule, where Omega_d is the solid angle from :func:`sphere_solid_angle`.

    Parameters
    ----------
    r : ndarray
        Radial grid.
    y : ndarray
        Integrand sampled on `r`.
    d : int
        Number of spatial dimensions.

    Returns
    -------
    float
        The d-dimensional volume integral of y.
    """
    return sphere_solid_angle(d)*np.trapz(y*r**(d-1),r)

def normalize(r,y,d):
    """Rescale y so its d-dimensional radial integral equals one."""
    return y/radial_integrate(r,y,d)
