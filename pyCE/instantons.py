"""Generation and configurational-entropy analysis of instantons.

Solves for the O(d)-symmetric bounce ("bubble") profile of a real scalar
field in the asymmetric double-well potential

    V(B) = (1/2) B**2 - (alpha/3) B**3 + (1/4) B**4,

with asymmetry parametrized by alpha = (3/sqrt(2)) * (1 + g). The bounce
satisfies the Euclidean equation of motion

    B'' + (d/r) B' = V'(B) = B - alpha B**2 + B**3,

with B'(0) = 0 and B -> 0 (the false vacuum) as r -> infinity, and is found
by a shooting method: bisect on the core value B(0) until the profile decays
into the false vacuum instead of over- or under-shooting.

From the profile, the class computes the energy densities, the Euclidean
action, and the configurational entropy of the energy-density power spectrum.
"""

import numpy as np
from pyCE.math import radialFT,radial_integrate

class instanton:
    """O(d)-symmetric bounce profile and its configurational entropy.

    Instantiation does all the work: it solves for the bounce by shooting,
    builds the energy profiles, and computes the Euclidean action and the
    configurational entropy.

    Parameters
    ----------
    asymmetry_factor : float
        Asymmetry g of the potential; alpha = (3/sqrt(2)) * (1 + g).
        g = 0 corresponds to the degenerate (thin-wall) limit.
    dimension : int
        Number of SPATIAL dimensions d.
    N : int
        Maximum number of radial integration steps (grid spacing dr = 0.01).

    Attributes
    ----------
    r : ndarray
        Radial grid on which the bounce was accepted.
    B, DB : ndarray
        Bounce profile B(r) and its radial derivative B'(r).
    B0 : float
        Core value B(0) found by the shooting method.
    Bmin, Bmax : float
        Initial bracket for the shooting method.
    PEdens, GEdens, rho : ndarray
        Potential, gradient, and total energy densities.
    PE, GE, E : float
        Radially integrated potential, gradient, and total energies.
    Se : float
        Euclidean action: rho integrated in d+1 dimensions (the O(d+1)
        bounce interpretation of the static d-dimensional profile).
    denFT, k : ndarray
        Radial Fourier transform of rho and its k-grid.
    mf : ndarray
        Modal fraction ``|denFT|**2 / max|denFT|**2``.
    Sc : float
        Configurational entropy, -Integral[ mf * ln(mf) ] over k-space
        with the d-dimensional radial measure.
    """
    #Remember d is the number of SPATIAL dimensions
    def __init__(self,asymmetry_factor,dimension,N):
        eps     = np.finfo(np.longdouble).eps
        self.d  = dimension
        self.g  = np.longdouble(asymmetry_factor)
        self.dr = np.longdouble(.01)
        self.dk = self.dr
        alpha   = np.longdouble(3/np.sqrt(2)*(1+asymmetry_factor))
        self.alpha = alpha

        self.__generate_bubble_profile__(alpha,N)

        self.__get_energy_profiles__(alpha)

        self.__get_euclidean_action__()

        self.__get_entropy__()


    def __generate_bubble_profile__(self,alpha,N):
        """Bracket the core value and run the shooting method.

        Bmax is the true-vacuum field value (the larger root of V'(B)/B = 0);
        starting above it overshoots. Bmin is the value where the inverted
        potential -V has dropped enough that the field cannot escape the
        false vacuum; starting below it undershoots. The bounce core value
        B(0) lies between the two.
        """
        #Set up the bounds for the shooting method
        self.Bmax = np.longdouble((alpha+np.sqrt(alpha**2-4.))/2.)
        self.Bmin = np.longdouble((2.*alpha-np.sqrt(4.*alpha**2-18.))/3.)
        self.__shootFor__(N,self.d)

    def __get_energy_profiles__(self,alpha):
        """Compute potential/gradient/total energy densities and energies."""
        self.PEdens = .5*self.B**2 - (alpha*self.B**3)/3.0 + .25*self.B**4
        self.GEdens = .5*self.DB**2
        self.PE     = radial_integrate(self.r,self.PEdens,self.d)
        self.GE     = radial_integrate(self.r,self.GEdens,self.d)
        self.rho    = self.PEdens + self.GEdens
        self.E      = self.PE+self.GE

    def __get_euclidean_action__(self):
        """Euclidean action: the energy density integrated in d+1 dimensions."""
        self.Se = radial_integrate(self.r,self.rho,self.d+1)

    def __get_entropy__(self):
        """Configurational entropy of the energy-density power spectrum.

        Fourier transforms rho, normalizes the power spectrum by its peak to
        form the modal fraction mf, and integrates -mf*ln(mf) over k-space.
        The machine epsilon inside the log regularizes mf = 0 modes.
        """
        self.denFT, self.k = radialFT(self.d,self.rho,self.r)
        f       = np.abs(self.denFT)**2
        self.mf = f/max(f)
        self.Sc = -radial_integrate(self.k,self.mf*np.log(np.finfo(np.longdouble).eps+self.mf),self.d)

    def __shootFor__(self,N,d):
        """Bisection shooting for the bounce core value B(0).

        Starting from the midpoint of [Bmin, Bmax], integrate outward with
        RK4 and inspect the last finite value of B: its sign says whether the
        shot over- or undershot, and B0 is corrected by a bisection step
        (deltaB * 2**-iteration). Iterate until the tail decays below 1e-18
        or the result stops changing (breakTag guards against stagnation at
        machine precision). Finally, keep only the radii where both B and
        -DB are positive and finite (where the log is not NaN), i.e. the
        monotonically decaying part of the profile, and demote the long-double
        working arrays to float64.
        """
        isnotnan  = lambda x: ~np.isnan(x)
        deltaB    = np.longdouble(.5*(self.Bmax - self.Bmin))
        self.B0   = np.longdouble(self.Bmin + deltaB)
        increment = np.longdouble(1.0)
        self.B    = np.array([np.longdouble(10)])
        lastB     = self.B[isnotnan(self.B)][-1]
        breakTag  = 0
        while np.abs(lastB)>1e-18:
            self.__RK4__(N,d)
            lastBnew = self.B[isnotnan(self.B)][-1]
            #print(self.B0,lastBnew)
            if lastBnew == lastB:
                breakTag = breakTag + 1
                if breakTag > 3:
                    break
            else:
                breakTag = 0
            lastB     = lastBnew
            increment = increment/np.longdouble(2.)
            self.B0   = self.B0 + np.longdouble(np.sign(lastB))*deltaB*increment
        idx = isnotnan(np.log(self.B)) & isnotnan(np.log(-self.DB+np.finfo(np.longdouble).eps))
        self.B     = np.asarray(self.B[idx], dtype=float)
        self.DB    = np.asarray(self.DB[idx], dtype=float)
        self.r     = np.asarray(self.r[idx], dtype=float)
        self.r[0]  = 0.0
        self.DB[0] = 0.0

    def __RK4__(self,N,d):
        """Integrate the bounce ODE outward from r ~ 0 with classic RK4.

        The second-order equation is split into the first-order system
        B' = DB and DB' = -(d/r) DB + V'(B). Initial data sit at r = eps
        with B = B0 and DB = -eps (an infinitesimal inward slope avoids the
        coordinate singularity at the origin). Integration runs for at most
        N steps of size dr; an OverflowError (the shot diverging at
        long-double range) ends the run early, and the profile collected so
        far is stored in self.B, self.DB, self.r.
        """
        n  = 0
        B1 = lambda r,b,b1: b1
        B2 = lambda r,b,b1: -d*b1/r + b - self.alpha*b**2 + b**3
        B  = [self.B0, self.B0]
        DB = [-np.finfo(np.longdouble).eps,-np.finfo(np.longdouble).eps]
        r  = [np.finfo(np.longdouble).eps,self.dr]
        while n < N-2:
            try:
                n   = n + 1
                k31 = B1(r[-1],B[-1],DB[-1])
                k41 = B2(r[-1],B[-1],DB[-1])

                k32 = B1(r[-1]+.5*self.dr,B[-1] + .5*self.dr*k31,DB[-1] + .5*self.dr*k41)
                k42 = B2(r[-1]+.5*self.dr,B[-1] + .5*self.dr*k31,DB[-1] + .5*self.dr*k41)

                k33 = B1(r[-1]+.5*self.dr,B[-1] + .5*self.dr*k32,DB[-1] + .5*self.dr*k42)
                k43 = B2(r[-1]+.5*self.dr,B[-1] + .5*self.dr*k32,DB[-1] + .5*self.dr*k42)

                k34 = B1(r[-1]+self.dr,B[-1]+self.dr*k33,DB[-1]+self.dr*k43)
                k44 = B2(r[-1]+self.dr,B[-1]+self.dr*k33,DB[-1]+self.dr*k43)

                r.append(r[-1]+self.dr)
                B.append(B[-1] + self.dr/6.*(k31+2.*k32+2.*k33+k34))
                DB.append(DB[-1] + self.dr/6.*(k41+2.*k42+2.*k43+k44))
            except OverflowError:
                print("Overflow occured at n =", n)
                break
        self.B  = np.array(B)
        self.DB = np.array(DB)
        self.r  = np.array(r)
