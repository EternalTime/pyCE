import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
import tqdm
import scipy.special as sp

class oscillon:
    """
    ----------------------------------------------------------------------------
    OSCILLON CLASS
    ----------------------------------------------------------------------------
    Creates an environment in which to simulate an oscillon. Sets up the lattice
    with an MIB coordinate system. Set's up the potential, using the Kolb and
    Turner
    ----------------------------------------------------------------------------
    KWARGS: asymmetry_factor-- K&T dimensionless potential, epsilon DEFAULT: 0
            dimension       -- number of spatial dimensions         DEFAULT: 3
            N               -- size of spatial lattice              DEFAULT: 1000
            radius_max      -- maximum radius of spatial lattice    DEFAULT: 30
            radius_MIB      -- radius at which coordinates get boosted
                                DEFAULT: 29
            delta_MIB       -- width of boost region                DEFAULT: .5
            radius_cap      -- set boundary of oscillon             DEFAULT: 20
            tol             -- tolerance of iterative scheme in timestep
                                                                    DEFAULT: 10**-8
            dissipation     -- strength of dissipation              DEFAULT: 1.0
            courant_factor  -- ratio of dt/dx                       DEFAULT: 0.5
    ----------------------------------------------------------------------------
    """
    def __init__(self,asymmetry_factor = 0.,dimension = 3,N = 1000,
                    radius_max = 30, radius_MIB = 29, delta_MIB = .5,
                    radius_cap = 20, tol = 10**-8, dissipation = 1.0,
                    courant_factor = .5):
        self.d  = dimension
        self.N  = N
        self.alpha = 3./np.sqrt(2)*(1+asymmetry_factor)

        self.r  = np.linspace(0,radius_max,N)
        self.dr = np.mean(np.diff(self.r))
        self.dt = courant_factor*self.dr

        self._courant_factor = courant_factor
        self._dissipation_factor = dissipation
        self._rmax = radius_max
        self._rcap = radius_cap
        self._define_boost_factor(radius_MIB,delta_MIB)
        self._Ncap = int(sum(self.r<=radius_cap))
        self._tol = tol

    def initialize_field(self,field_type = 'gaussian', *params):
        if field_type == 'gaussian':
            r0 = params[0]
            A  = .5*(self.alpha+np.sqrt(self.alpha**2-4))
            f0 = A*np.exp(-(self.r/r0)**2)
            f1 = -2*self.r/r0**2*f0
            f2 = -self._f*f1
        elif field_type == 'tanh':
            print('not yet available')
            return 0
        else:
            print('field_type not understood')
            return 0
        return np.array([f0,f1,f2])

    def _define_boost_factor(self,radius_MIB,delta_MIB):
        r = (self.r - radius_MIB)/delta_MIB
        boost = .5*np.tanh(r)
        boost = boost - boost[0]
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

    def _timestep(self,fields):
        error = 1.0
        fields_old  = fields*1.0
        fields_0    = fields_old + self.dt*( self._dissipation(fields_old)
                                + .5*self._F(fields_old))
        while error > self._tol:
            fields_new  = fields_0 + .5*self.dt*self._F(fields_old)
            error       = _L2_norm(fields_new[:,1:],
                                    fields_old[:,1:])
            fields_old  = 1.*fields_new

        return fields_new

    def _timestep_start(self):
        self._rt        = self.r
        self._rtd       = self.r**(self.d-1)
        self._a         = np.ones(len(self.r))
        self._b         = self._f/self._a
        self._drd       = np.convolve(self._rt**self.d,[.5,0,-.5],'same')
        self._drd[0]    = self.dr**self.d
        self._drd[-3:]  = np.convolve(self._rt[(self.N-6):self.N]**self.d,
                            np.array([-11.,18.,-9.,2.])/6.,
                            'valid')

    def _timestep_update(self):
        self._rt        = self._rt + self._f*self.dt
        self._rtd       = self._rt**(self.d-1)
        self._a         = self._a + self._df*self.dt
        self._b         = self._f/self._a
        self._drd       = np.convolve(self._rt**self.d,[.5,0,-.5],'same')
        self._drd[0]    = self.dr**self.d
        self._drd[-3:]  = np.convolve(self._rt[(self.N-6):self.N]**self.d,
                            np.array([-11.,18.,-9.,2.])/6.,
                            'valid')

    def _F(self,fields):
        f0 = fields[0]
        f1 = fields[1]
        f2 = fields[2]

        dF      = np.zeros([3,self.N])
        dF[0]   = f2/self._a+self._b*f1

        dF[1]       = np.convolve(dF[0],[.5,0,-.5],'same')/self.dr
        dF[1,0]     = 0
        dF[1,-1]    = 0

        temp        = (self._b*f2+f1/self._a)*self._rtd
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
        return -self._dissipation_factor*dF*self._courant_factor**4

    def stress_energy_tensor(self,fields):
        V   = self._potential(fields[0])
        rho = (fields[2]**2+fields[1]**2)/(2.*self._a**2) + V
        j   = -fields[1]*fields[2]/self._a**3 - self._b*rho
        P   = ((1-self._f**2)*rho - 2*V)/self._a**2 - 2*self._b*j
        return np.array([rho,j,P])

    def energy(self,T):
        E = radial_integrate(self._rt[0:self._Ncap],T[0,0:self._Ncap],self.d)
        dE = (sphere_solid_angle(self.d)*self._rt[self._Ncap]**(self.d-1)*
                T[1,self._Ncap]*self.dt)
        return E,dE

    def simulate_oscillon(self,fields,printTag = True,saveTag = False):
        self._timestep_start()
        t = [0]
        SET = self.stress_energy_tensor(fields)
        E0,dE = self.energy(SET)
        E = 1.*E0
        energy_history = [E0]
        n = 0
        while E > .01*E0:
            fields = self._timestep(fields)
            self._timestep_update()
            if np.mod(n,30) == 0:
                SET = self.stress_energy_tensor(fields)
                E,dE = self.energy(SET)
                energy_history.append(E)
                t.append(t[-1]+100*self.dt)
                if printTag == True:
                    plt.clf()
                    plt.plot(self._rt,fields[0],',',markersize = .2)
                    plt.ylim([-1,2])
                    plt.xlim([0,50])
                    plt.draw()
                    plt.pause(.00000000001)
            n = n+1
        self.E = np.array(energy_history)
        self.time = np.array(t)

def sphere_solid_angle(d):
    return 2*np.pi**(d/2.0)/np.math.gamma(d/2.0)

def radial_integrate(x,y,d):
    return sphere_solid_angle(d)*np.trapz(y*x**(d-1),x)

def _L2_norm(y1,y2):
    return np.sqrt(np.mean((y1-y2)**2))
