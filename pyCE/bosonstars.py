"""Generation and configurational-entropy analysis of boson stars.

Solves for spherically symmetric, ground-state solutions of the
Einstein-Klein-Gordon system for a complex scalar field with the harmonic
ansatz phi(r, t) = phi(r) e^(-i omega t) and the potential::

    V(|phi|) = |phi|**2 + (lamb/2)|phi|**4

In the standard 3+1 metric ansatz, ds^2 = -alpha(r)^2 dt^2 + a(r)^2 dr^2
+ r^2 dOmega^2, the field and metric functions obey a coupled ODE system in
r with a frequency eigenvalue omega: only a discrete set of central lapse
values yields a profile that decays into flat space, and the nodeless member
of that set is the ground state. The solver finds it by bisection shooting
on the central lapse, integrating outward with RK4.

For reviews see Schunck & Mielke (gr-qc/0410040) and Liebling & Palenzuela
(arXiv:1202.5809). From the solution, the class computes the ADM mass, the
effective radius, and the configurational entropy of the energy density.
"""

import numpy as np
from pyCE.math import radialFT, radial_integrate

class BosonStar:
    """Ground-state boson star and its configurational entropy.

    Instantiation does all the work: it shoots for the central lapse,
    integrates the Einstein-Klein-Gordon system outward, and extracts the
    physical features and the configurational entropy.

    Parameters
    ----------
    phi0 : float
        Core value of the scalar field.
    lamb : float, optional
        Quartic self-interaction strength (default 0, the mini boson star).
    alpha_range : [float, float], optional
        Initial bracket for the central lapse used by the shooting method
        (default [0, 1]). The supplied list is copied, never modified.
    dr : float, optional
        Radial step size (default .01). For small phi0 the gradients are
        gentle but the star is large; for large phi0 the reverse — lower
        `dr` if the shooting method stalls.
    r_max : float, optional
        Maximum integration radius (default 50). Too small and the shot
        converges to a profile that is not yet the ground state; too large
        and the integration stops at `phi_tol` before reaching the tail.
    phi_tol : float, optional
        Field amplitude at which the tail is declared converged
        (default 1e-6).
    plot_progress : bool, optional
        Plot each shooting iterate live (default False).

    Attributes
    ----------
    r : ndarray
        Radial grid.
    phi, Phi : ndarray
        Scalar field profile and its radial derivative.
    a, alpha : ndarray
        Radial metric function and lapse, with the lapse rescaled so
        alpha -> 1 at the outer edge (asymptotic flatness).
    omega : float
        Frequency eigenvalue of the harmonic ansatz; omega < 1 signals a
        gravitationally bound configuration.
    rho : ndarray
        Energy density.
    mass : ndarray
        Misner-Sharp mass within radius r, m(r) = (r/2)(1 - 1/a**2).
    M : float
        Total (ADM) mass, mass[-1].
    R : float
        Effective radius: the first moment of r over the proper energy
        distribution.
    denFT, k : ndarray
        Radial Fourier transform of rho and its k-grid.
    mf : ndarray
        Modal fraction ``|denFT|**2 / max|denFT|**2``.
    Sc : float
        Configurational entropy, -Integral[ mf * ln(mf) ] over k-space
        with the d = 3 radial measure.

    Examples
    --------
    >>> from pyCE.bosonstars import BosonStar
    >>> star = BosonStar(0.1)
    >>> star.M, star.R, star.omega    # doctest: +SKIP
    (0.5326, 6.004, 0.9367)
    """

    def __init__(self, phi0,
                 lamb=0,
                 alpha_range=None,
                 dr=.01,
                 r_max=50,
                 phi_tol=10**-6,
                 plot_progress=False):
        self.dr = dr
        self.lamb = lamb
        alpha_range = list(alpha_range) if alpha_range is not None else [0, 1]
        self.generate_profile(phi0, alpha_range, r_max, phi_tol,
                              plot_progress)
        self.extract_physical_features()
        self._get_entropy()

    def extract_physical_features(self):
        """Energy density, mass function, ADM mass, and effective radius."""
        #Misner-Sharp mass; its asymptotic value is the ADM mass
        self.mass = .5*self.r*(1 - 1.0/self.a**2)
        self.M = self.mass[-1]
        self.rho = (.5*(1 + .5*self.lamb*self.phi**2 + 1/self.alpha**2)
                    *self.a*self.phi**2 + .5*self.Phi**2/self.a)
        #first moment of r over the proper energy distribution
        self.R = (np.trapezoid(self.a*self.alpha*self.rho*self.r**3, self.r)
                  /np.trapezoid(self.a*self.alpha*self.rho*self.r**2, self.r))

    def generate_profile(self, phi0, alpha_range, r_max, phi_tol,
                         plot_progress=False):
        """Bisection shooting on the central lapse.

        Each iterate integrates outward from the midpoint of the current
        bracket. A shot whose field diverges upward without crossing zero
        sat too high in the bracket; anything else (a crossing, or downward
        divergence) sat too low. The bracket halves each pass, and the loop
        ends when the tail amplitude drops below `phi_tol` or the bracket
        is exhausted at machine precision.
        """
        if plot_progress:
            import matplotlib.pyplot as plt
            plt.clf()

        self.r = [self.dr]
        self.a = [1]
        self.phi = [phi0]
        self.Phi = [0]

        d_alpha = alpha_range[1] - alpha_range[0]
        while (np.abs(self.phi[-1]) > phi_tol) and (d_alpha > 10**-16):

            self.r = [self.dr]
            self.a = [1]
            self.phi = [phi0]
            self.Phi = [0]

            alpha0 = np.mean(alpha_range)
            self.alpha = [alpha0]

            self._RK4_(r_max)

            if ((np.sign(self.phi[-2]) == 1)
                    and (sum(np.abs(np.diff(np.sign(self.phi[::-3])))) < 2)):
                alpha_range[1] = alpha_range[1] - .5*d_alpha
            else:
                alpha_range[0] = alpha_range[0] + .5*d_alpha
            d_alpha = .5*d_alpha

            if plot_progress:
                plt.plot(self.r[:-1], self.phi[:-1], linewidth=.2,
                         color='red')
                plt.ylim([-phi0, 2*phi0])
                plt.pause(.001)
        if plot_progress:
            plt.clf()
            plt.plot(self.r, self.phi)
            plt.pause(.001)

    def _get_entropy(self):
        """Configurational entropy of the energy-density power spectrum.

        Fourier transforms rho, normalizes the power spectrum by its peak
        to form the modal fraction mf, and integrates -mf*ln(mf) over
        k-space. The machine epsilon inside the log regularizes mf = 0
        modes.
        """
        self.denFT, self.k = radialFT(3, self.rho, self.r)
        f = np.abs(self.denFT)**2
        self.mf = f/max(f)
        self.Sc = -radial_integrate(
            self.k, self.mf*np.log(np.finfo(float).eps + self.mf), 3)

    def _RK4_(self, r_max):
        """Integrate the Einstein-Klein-Gordon system outward with RK4.

        The unknowns are (a, alpha, phi, Phi); the right-hand sides below
        are the standard polar-areal-gauge equations with the harmonic
        ansatz, the frequency absorbed into the lapse normalization.
        Integration stops at `r_max` or when any unknown leaves a
        generous bounding box (a diverging shot announcing itself). On
        exit, the lapse is rescaled by its asymptotic value so that
        alpha -> 1 at the edge, and that rescaling factor is the
        frequency eigenvalue omega.
        """
        c = 0.16666666666666666666667

        d       = lambda k1,k2,k3,k4: c*(k1 + 2.0*k2 + 2.0*k3 + k4)
        F_a     = lambda r,a,alpha,phi,Phi: .5*a*( -(a**2-1)/r
                    + r*( (1.0/alpha**2 + 1 + .5*self.lamb*phi**2)*a**2*phi**2
                    + Phi**2))*self.dr
        F_alpha = lambda r,a,alpha,phi,Phi: .5*alpha*( (a**2-1)/r
                    + r*( (1.0/alpha**2 - 1 - .5*self.lamb*phi**2)*a**2*phi**2
                    + Phi**2))*self.dr
        F_phi   = lambda r,a,alpha,phi,Phi: Phi*self.dr
        F_Phi   = lambda r,a,alpha,phi,Phi: (-(1 + a**2 - (a*r*phi)**2)*Phi/r
                    - (1.0/alpha**2 - 1 - self.lamb*phi**2)*a**2*phi)*self.dr

        dr      = self.dr
        r       = self.r
        a       = self.a
        alpha   = self.alpha
        phi     = self.phi
        phi0    = phi[0]
        Phi     = self.Phi

        while ((r[-1] < r_max)
            and (np.abs(a[-1]) < max(100*phi0,100))
            and (np.abs(alpha[-1]) < max(100*phi0,100))
            and (np.abs(phi[-1]) < max(10*phi0,100))
            and (np.abs(Phi[-1]) < max(10*phi0,100))):

            k1_a      = F_a( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_alpha  = F_alpha( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_phi    = F_phi( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_Phi    = F_Phi( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])

            k2_a      = F_a(    r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_alpha  = F_alpha(r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_phi    = F_phi(  r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_Phi    = F_Phi(  r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k3_a      = F_a(    r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_alpha  = F_alpha(r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_phi    = F_phi(  r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_Phi    = F_Phi(  r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)

            k4_a      = F_a(    r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_alpha  = F_alpha(r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_phi    = F_phi(  r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_Phi    = F_Phi(  r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)

            r.append(r[-1] + dr)
            a.append(a[-1] + d(k1_a,k2_a,k3_a,k4_a))
            alpha.append(alpha[-1] + d(k1_alpha,k2_alpha,k3_alpha,k4_alpha))
            phi.append(phi[-1] + d(k1_phi,k2_phi,k3_phi,k4_phi))
            Phi.append(Phi[-1] + d(k1_Phi,k2_Phi,k3_Phi,k4_Phi))

        self.omega = 1.0/(a[-1]*alpha[-1])
        self.r = np.array(r)
        self.a = np.array(a)
        self.alpha = np.array(alpha)*self.omega
        self.phi = np.array(phi)
        self.Phi = np.array(Phi)
