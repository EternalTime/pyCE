import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
import tqdm
import scipy.special as sp

class oscillon:
    def __init__(asymmetry_factor = 0.,dimension = 3,N = 1000,
                    radius_max = 20, radius_MIB = 17, delta_MIB = .5,
                    radius_cap = 15, tol = 10**-8, dissipation = 1.0,
                    courant_factor = .5):
        self.d  = dimension
        self.N  = N
        self.alpha = 3./np.sqrt(2)*(1+asymmetry_factor)

        self.r  = np.linspace(0,rmax,N)
        self.dr = np.mean(np.diff(r))
        self.dt = courant_factor*self.dr

        self._dissipation_factor = dissipation
        self._rmax = radius_max
        self._rcap = radius_cap

    def _boost_factor(self,radius_MIB,delta_MIB):
        r = (self.r - radius_MIB)/delta_MIB
        boost = .5*np.tanh(r)
        boost = boost_factor - boost_factor[0]
        d_boost = .5/(delta_MIB*np.cosh(r)**2)
        self._f = boost
        self._df = d_boost

    def _potential(self,phi):
        phi2 = phi**2
        phi3 = phi2*phi
        phi4 = phi2*phi2
        return .5*phi2 - self.alpha*phi3/3. + .25*phi4

    def _gradient_potential(self,phi):
        phi2 = phi**2
        phi3 = phi2*phi
        return phi - self.alpha*phi2 + phi3

    def simulate_oscillon(self,printTag = True,saveTag = False):

    def _timestep(self,fields):
        error = 1.0
        fields_old  = fields*1.0
        fields_0    = fields_old + self.dt*( self._dissipation(fields_old)
                                + .5*self._F(fields_old))
        while error > self.tol:
            fields_new  = fields_0 + .5*self.dt*self._F(fields_old)
            error       = self._L2_norm(fields_new[:,1:],fields_old[:,1:])
            fields_old  = 1.*fields_new

        self._rt        = self._rt + self.f*self.dt
        self._rtd       = self.rt**(self.d-1)
        self._a         = self._a + self._df*self.dt
        self._b         = self._f/self._a
        self._drd       = np.convolve(self._rt**self.d,[.5,0,-.5],'same')
        self._drd[0]    = self.dr**self.d
        self._drd[-3:]  = np.convolve(rt[(N-6):N]**d,
                            np.array([-11.,18.,-9.,2.])/6.,
                            'valid')
        return fields_new

    def _F(self,fields):
        f0 = fields[0]
        f1 = fields[1]
        f2 = fields[2]

        dF      = np.zeros([3,self.N])
        dF[0]   = f2/self._a+self._b*f1

        dF[1]       = np.convolve(dF[0],[.5,0,-.5],'same')/self.dr
        dF[1,0]     = 0
        dF[1,-1]    = 0

        temp        = self._b*f2+f1/self._a)*self._rtd
        dF[2]       = np.convolve(temp,[.5,0,-.5],'same')/self._drd
        dF[2]       = self._a*(self.d*dF[2] - self._gradient_potential(f0))
        dF[2,1:]    = dF[2,1:] - (self.d-1)*self._f[1:]/self._rt[1:]*f2[1:]
        dF[2,0]     = dF[2,1]
        dF[2,-1]    = dF[2,-2]

        return dF

    def _dissipation(self,fields):
        dF = np.array(map(lambda f:
                            np.convolve(f,[1.,-4.,6.,-4.,1.],'same'),fields))
        dF[:,0:1]   = 0
        dF[:,1:2]   = 0
        dF[:,-2:-1] = 0
        dF[:,-1:]   = 0
        return -self._dissipation_factor*dF*courant_factor**4

def initialize_profile(profile_type = 'thick', r0, delta = 0.0):
    if profile_type = 'thick':
        radius = r0
        np.exp(-(self.r/r0)**2)
    elif profile_type = 'thin':
        radius = r0
    else:
        print('Unsupported profile type, please try again.')
