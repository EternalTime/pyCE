"""Polytropic stellar models via the Lane-Emden equation.

A polytrope is a self-gravitating fluid sphere with equation of state
P = K * rho**gamma, gamma = 1 + 1/n. In scaled variables the structure is
governed by the Lane-Emden equation

    theta'' + (2/r) theta' = -theta**n,

with theta(0) = 1 and theta'(0) = 0; the density is rho = theta**n and the
(first) zero of theta marks the stellar surface.
"""

import numpy as np


class polytrope:
    """Polytropic stellar model from the Lane-Emden equation.

    Creates a polytrope with index n, which is related to the adiabatic index
    via gamma = 1 + 1/n. Uses an RK4 method to solve the Lane-Emden equation.

    Parameters
    ----------
    n : float, optional
        Polytropic index (default 1.5).
    dr : float, optional
        Radial step size (default .01).

    Attributes
    ----------
    theta : ndarray
        Lane-Emden profile.
    psi : ndarray
        Radial derivative of theta.
    rho : ndarray
        Scaled density profile.
    r : ndarray
        Radial distance array in scaled lengths.
    """
    def __init__(self,n = 1.5,dr = .01):
        self.n = n
        self.gamma = 1+1.0/n
        self.dr = dr
        self.__solve_Lane_Emden__()
        self.rho = self.theta**self.n

    def __solve_Lane_Emden__(self):
        """Integrate the Lane-Emden equation outward with classic RK4.

        The second-order equation is split into the first-order system
        theta' = psi and psi' = -(2/r) psi - theta**n. Initial data sit at
        r = eps with theta = 1 and psi = -eps (an infinitesimal inward slope
        avoids the coordinate singularity at the origin). Integration stops
        once theta drops to 1e-5, i.e. at the stellar surface, or on
        OverflowError (n >= 5 polytropes have no finite surface). The
        profile is then truncated to theta > 0 and stored in self.theta,
        self.psi, self.r.
        """
        n  = 0
        F1 = lambda r,theta,psi: psi
        F2 = lambda r,theta,psi: -2.0*psi/r - theta**self.n
        theta  = [1.0, 1.0]
        psi = [-np.finfo(np.float64).eps,-np.finfo(np.float64).eps]
        r  = [np.finfo(np.float64).eps,self.dr]
        while theta[-1] > 10**-5:
            try:
                n   = n + 1
                k31 = F1(r[-1],theta[-1],psi[-1])
                k41 = F2(r[-1],theta[-1],psi[-1])

                k32 = F1(r[-1]+.5*self.dr,theta[-1] + .5*self.dr*k31,psi[-1] + .5*self.dr*k41)
                k42 = F2(r[-1]+.5*self.dr,theta[-1] + .5*self.dr*k31,psi[-1] + .5*self.dr*k41)

                k33 = F1(r[-1]+.5*self.dr,theta[-1] + .5*self.dr*k32,psi[-1] + .5*self.dr*k42)
                k43 = F2(r[-1]+.5*self.dr,theta[-1] + .5*self.dr*k32,psi[-1] + .5*self.dr*k42)

                k34 = F1(r[-1]+self.dr,theta[-1]+self.dr*k33,psi[-1]+self.dr*k43)
                k44 = F2(r[-1]+self.dr,theta[-1]+self.dr*k33,psi[-1]+self.dr*k43)

                dth,dps = (self.dr/6.)*(k31+2.*k32+2.*k33+k34),(self.dr/6.)*(k41+2.*k42+2.*k43+k44)
                theta.append(theta[-1] + dth)
                psi.append(psi[-1] + dps)
                r.append(r[-1]+self.dr)

            except OverflowError:
                print("Overflow occured at n =", n)
                break
        idx = np.array(theta) > 0.0
        self.theta  = np.array(theta)[idx]
        self.psi = np.array(psi)[idx]
        self.r  = np.array(r)[idx]
