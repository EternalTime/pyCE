import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
import tqdm
import scipy.special as sp
from pyCE.math import sphere_solid_angle,radial_integrate,radialFT_mat


class polytrope:
    """
    ----------------------------------------------------------------------------
    POLYTROPE CLASS
    ----------------------------------------------------------------------------
    Creates a polytrope with index n, which is related to the adiabatic index
    via gamma = 1 + 1/n. Uses an RK4 method to solve the Lane-Emden equation.

    ----------------------------------------------------------------------------
    KWARGS: n               -- polytropic index                     DEFAULT: 1.5
    ----------------------------------------------------------------------------
    ASPECTS:
            theta           -- Lane-Emden profile
            rho             -- scaled density profile
            pressure        -- scaled pressure profile
            r               -- radial distance array in scaled lengths
            R               -- radius of polytrope (not that polytropes with
                                n>=5 have infinite radius)


    ----------------------------------------------------------------------------
    """
    def __init__(self,n = 1.5,dr = .01):
        self.n = n
        self.gamma = 1+1.0/n
        self.dr = dr
        self.__solve_Lane_Emden__()
        self.rho = self.theta**self.n

    def __solve_Lane_Emden__(self):
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
				print "Overflow occured at n = ", n
				break
        idx = np.array(theta) > 0.0
        self.theta  = np.array(theta)[idx]
        self.psi = np.array(psi)[idx]
        self.r  = np.array(r)[idx]
