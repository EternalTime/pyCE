r"""Simulation of oscillons in d spatial dimensions.

Evolves a spherically symmetric real scalar field in the asymmetric
double-well (Kolb & Turner) potential

    V(phi) = (1/2) phi**2 - (alpha/3) phi**3 + (1/4) phi**4,

with alpha = (3/sqrt(2)) * (1 + asymmetry_factor). The field is evolved as
the first-order system (phi, Phi, Pi), where Phi is the radial gradient and
Pi the generalized momentum, on a radial grid in monotonically increasingly
boosted (MIB) coordinates: an outer region of the grid is smoothly boosted
outward so that outgoing radiation is carried off the lattice instead of
reflecting back onto the oscillon, mimicking an infinite domain. Time
stepping is an iterated Crank-Nicolson scheme with Kreiss-Oliger-style
dissipation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyCE.math import sphere_solid_angle,radial_integrate,radialFT_mat

class oscillon:
    r"""Oscillon environment class

    Creates an environment in which to simulate an oscillon. Sets up the
    lattice with an MIB coordinate system and the potential, using the Kolb
    and Turner parametrization.

    First create an environment::

      osc_env = oscillon(KWARGS)

    Then initialize a field::

      fields = osc_env.initialize_field(KWARGS)

    Run the simulation::

      osc_env.simulate_oscillon(fields,KWARGS)


    Parameters
    ----------
    asymmetry_factor : float
        Asymmetry epsilon of the K&T dimensionless potential. DEFAULT: 0
    dimension : int
        Number of SPATIAL dimensions. DEFAULT: 3
    N : int
        Size of the spatial lattice. DEFAULT: 1500
    radius_max : float
        Maximum radius of the spatial lattice. DEFAULT: 45
    radius_MIB : float
        Radius at which coordinates get boosted. DEFAULT: 44
    delta_MIB : float
        Width of the boost region. DEFAULT: .5
    radius_cap : float
        Boundary of the oscillon; energy is integrated within it. DEFAULT: 42
    tol : float
        Tolerance of the iterative scheme in the timestep. DEFAULT: 1e-8
    dissipation : float
        Strength of the numerical dissipation. DEFAULT: 0.8
    courant_factor : float
        Ratio :math:`dt/dx`. DEFAULT: 0.5

    Attributes
    ----------
    E : ndarray
        Energy history of the oscillon (set by `simulate_oscillon`).
    core : list
        Core value history, phi(r=0,t) (set by `simulate_oscillon`).
    time : ndarray
        Time values for the energy/core histories (set by
        `simulate_oscillon`).

    Notes
    -----
    The value of :math:`\omega` is larger than 5.
    """
    def __init__(self,asymmetry_factor = 0.,
                    dimension = 3,
                    N = 1500,
                    radius_max = 45,
                    radius_MIB = 44,
                    delta_MIB = .5,
                    radius_cap = 42,
                    tol = 10**-8,
                    dissipation = 0.8,
                    courant_factor = .5):
        self.d      = dimension
        self.N      = N
        self.alpha  = 3./np.sqrt(2)*(1+asymmetry_factor)

        self.r      = np.linspace(0,radius_max,N)
        self.dr     = np.mean(np.diff(self.r))
        self.dt     = courant_factor*self.dr

        self._courant_factor = courant_factor
        self._dissipation_factor = dissipation
        self._rmax  = radius_max
        self._rcap  = radius_cap
        self._Ncap  = int(sum(self.r<=radius_cap))
        self._tol   = tol
        self._define_boost_factor(radius_MIB,delta_MIB)
        #This tag turns true the first time a fourier transform is called. Then
        # the fourier matrix is precomputed, which means the first call will
        # always take a bit longer. Once the fourier matrix exists, fourier
        # transforms are much quicker ( still O(N^2)... it would be nice to
        # implement a FFT of O(NlogN)... Max? Michelle?)
        self._FT_tag = False


    def initialize_field(self,field_type = 'gaussian', *params):
        """Build the initial field vector (phi, Phi, Pi).

        Parameters
        ----------
        field_type : str
            Initial profile shape. Currently only 'gaussian' is implemented:
            a Gaussian of radius ``params[0]`` (note radius = sqrt(2)*sigma)
            with amplitude at the true-vacuum field value
            A = (alpha + sqrt(alpha**2 - 4))/2.
        *params
            Shape parameters; for 'gaussian', the radius r0.

        Returns
        -------
        ndarray, shape (3, N)
            Stacked initial data [phi, Phi, Pi]. Pi is initialized to
            -f*Phi, i.e. the time derivative of phi is zero in the boosted
            coordinates. Returns 0 if `field_type` is not recognized.
        """
        if field_type == 'gaussian':
            r0 = params[0]
            A  = .5*(self.alpha+np.sqrt(self.alpha**2-4))
            f0 = A*np.exp(-(self.r/r0)**2)
            f1 = -2*self.r/r0**2*f0
            f2 = -self._f*f1
        elif field_type == 'tanh':
            print('not yet available')
            return 0
        #elif field_type ==
        else:
            print('field_type not understood')
            return 0
        return np.array([f0,f1,f2])

    def _define_boost_factor(self,radius_MIB,delta_MIB):
        """Set up the MIB shift profile f(r) and its derivative df/dr.

        f is a smoothed step (tanh of width delta_MIB centered on
        radius_MIB) rising from 0 in the interior to ~1 in the outer region,
        so the inner grid is static while the outer grid is boosted outward.
        """
        r = (self.r - radius_MIB)/delta_MIB
        boost = .5*np.tanh(r)
        boost = boost - boost[0]
        d_boost = .5/(delta_MIB*np.cosh(r)**2)
        self._f = boost
        self._df = d_boost

    def _potential(self,phi):
        """K&T potential V(phi) = phi**2/2 - alpha*phi**3/3 + phi**4/4."""
        phi2 = phi**2
        phi3 = phi2*phi
        phi4 = phi2*phi2
        return .5*phi2 - self.alpha*phi3/3. + .25*phi4

    def _gradient_potential(self,phi):
        """Potential gradient V'(phi) = phi - alpha*phi**2 + phi**3."""
        phi2 = phi**2
        phi3 = phi2*phi
        return phi - self.alpha*phi2 + phi3

    def _timestep(self,fields):
        """Advance the field vector by one step of iterated Crank-Nicolson.

        A predictor half-step (plus dissipation) seeds the iteration; each
        pass re-evaluates the RHS at the running average and reapplies the
        boundary conditions (regularity of Phi at the origin via a one-sided
        stencil, Phi(0) = 0, even Pi at the origin, and zero-gradient outer
        boundary). Iteration stops when successive iterates differ by less
        than `tol` in the L2 norm. The grid quantities are then advanced by
        `_timestep_update`.
        """
        error = 1.0
        fields_old  = fields*1.0
        fields_0    = fields_old + self.dt*(.5*self._F(fields_old)
                                        + self._dissipation(fields))
        while error > self._tol:
            #iteration step
            fields_new  = fields_0 + .5*self.dt*self._F(fields_old)
            #implementing boundary conditions
            fields_new[1,1] = (10*fields_new[1,2]-10*fields_new[1,3]
                                +5*fields_new[1,4]-fields_new[1,5])/5.0
            fields_new[1,0] = 0
            fields_new[2,0] = fields_new[2,1]
            fields_new[1,-1] = fields_new[1,-2]
            fields_new[2,-1] = fields_new[2,-2]
            error       = _L2_norm(fields_new[:,1:],
                                    fields_old[:,1:])
            fields_old  = 1.*fields_new
        self._timestep_update()
        return fields_new

    def _simulation_start(self):
        """Initialize the time-dependent grid quantities.

        r~ (the boosted radial coordinate), the metric-like factors
        a (initially 1) and b = f/a, and d(r~^d) (the volume-element
        differences used by the flux-conservative derivative in `_F`,
        with one-sided stencils at the origin and outer edge).
        """
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
        """Advance the MIB grid quantities by dt.

        The boosted radius moves as dr~/dt = f and the factor a grows as
        da/dt = df/dr; b and the volume elements are recomputed to match.
        """
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
        """Right-hand side of the first-order MIB evolution system.

        With fields = (phi, Phi, Pi), a the MIB factor, and b = f/a:

        - d(phi)/dt = Pi/a + b*Phi
        - d(Phi)/dt = the radial derivative of d(phi)/dt (centered
          differences in the bulk, one-sided stencils at the boundaries,
          and a regularity-preserving extrapolation at the first interior
          point)
        - d(Pi)/dt  = the flux-conservative radial derivative of the
          momentum flux (b*Pi + Phi/a) * r~^(d-1), divided by the volume
          element, minus a*V'(phi), with the geometric source term
          -(d-1)*(f/r~)*Pi; the origin and outer edge are patched with
          one-sided extrapolation stencils.

        Commented-out stencils record alternatives that did not converge.

        Parameters
        ----------
        fields : ndarray, shape (3, N)

        Returns
        -------
        ndarray, shape (3, N)
            Time derivatives of (phi, Phi, Pi).
        """
        f0 = fields[0]
        f1 = fields[1]
        f2 = fields[2]

        dF      = np.zeros([3,self.N])
        dF[0]   = f2/self._a+self._b*f1

        dF[1,1:(self.N-1)] = np.convolve(dF[0],[.5,0,-.5],'valid')/self.dr
        dF[1,-1]    = -dF[1,-5]+4*dF[1,-4]-6*dF[1,-3]+4*dF[1,-2]
                    #finite difference stencil not converging
                    #(3*f0[-5] - 16*f0[-4] + 36*f0[-3]
                    #    -48*f0[-2] + 25*f0[-1])/(12*self.dr)
        dF[1,0]     = (-25*dF[0,0] + 48*dF[0,1] - 36*dF[0,2]
                        + 16*dF[0,3] - 3*dF[0,4])/(12*self.dr)
                    #fitting to the origin seems to break the regularity
                    #4*dF[1,1] - 6*dF[1,2] + 4*dF[1,3] - dF[1,4]
                    #finite difference stencil not converging
                    #
        dF[1,1]     = (10*dF[1,2]-10*dF[1,3]+5*dF[1,4]-dF[1,5])/5.0

        temp        = (self._b*f2 + f1/self._a)*self._rtd
        dF[2,1:(self.N-1)]  = (np.convolve(temp,[.5,0,-.5],'valid')/
                                    self._drd[1:(self.N-1)])
        dF[2,-1]    = -dF[2,-5]+4*dF[2,-4]-6*dF[2,-3]+4*dF[2,-2]
                    #finite difference stencil not converging
                    #(3*temp[-5] - 16*temp[-4] + 36*temp[-3]
                    #    -48*temp[-2] + 25*temp[-1])/(12*self._drd[-1])
        dF[2]       = self._a*(self.d*dF[2] - self._gradient_potential(f0))
        dF[2,1:]    = dF[2,1:] - (self.d-1)*self._f[1:]/self._rt[1:]*f2[1:]
        dF[2,0]     = 4*dF[2,1] - 6*dF[2,2] + 4*dF[2,3] - dF[2,4]
        dF[2,-1]    = -dF[2,-5]+4*dF[2,-4]-6*dF[2,-3]+4*dF[2,-2]

        return dF

    def _dissipation(self,fields):
        """Fourth-derivative (Kreiss-Oliger-style) numerical dissipation.

        Applies the five-point fourth-difference stencil [1,-4,6,-4,1] to
        each field component to damp grid-scale noise, scaled by the
        dissipation factor and courant_factor**4. The two points at each
        boundary are zeroed (commented-out one-sided stencils preserved for
        reference).
        """
        dF = np.array(list(map(lambda f:
                            np.convolve(f,[1.,-4.,6.,-4.,1.],'same'),
                            fields)))/self.dr
        dF[:,0:1]   = 0#(35*fields[:,0:1] - 186*fields[:,1:2] + 411*fields[:,2:3]
                    #    - 484*fields[:,3:4] + 321*fields[:,4:5]
                    #    - 114*fields[:,5:6] + 17*fields[:,6:7])/6.0
        dF[:,1:2]   = 0#(17*fields[:,0:1] - 84*fields[:,1:2] + 171*fields[:,2:3]
                    #    - 184*fields[:,3:4] + 11*fields[:,4:5]
                    #    - 36*fields[:,5:6]  + 5*fields[:,6:7])/6.0
        dF[:,-2:-1] = 0#(5*fields[:,-7:-6] - 36*fields[:,-6:-5] + 111*fields[:,-5:-4]
                    #    - 184*fields[:,-4:-3] + 171*fields[:,-3:-2]
                    #    - 84*fields[:,-2:-1] + 17*fields[:,-1:])/6.0
        dF[:,-1:]   = 0# (17*fields[:,-7:-6] - 114*fields[:,-6:-5] + 321*fields[:,-5:-4]
                    #    - 484*fields[:,-4:-3] + 411*fields[:,-3:-2]
                    #    - 186*fields[:,-2:-1] + 35*fields[:,-1:])/6.0
        return -self._dissipation_factor*dF*self._courant_factor**4

    def stress_energy_tensor(self,fields):
        """Components of the stress-energy in the MIB coordinates.

        Parameters
        ----------
        fields : ndarray, shape (3, N)

        Returns
        -------
        ndarray, shape (3, N)
            [rho, j, P]: energy density, radial momentum density (flux),
            and radial pressure.
        """
        V   = self._potential(fields[0])
        rho = (fields[2]**2+fields[1]**2)/(2.*self._a**2) + V
        j   = -fields[1]*fields[2]/self._a**3 - self._b*rho
        P   = ((1-self._f**2)*rho - 2*V)/self._a**2 - 2*self._b*j
        return np.array([rho,j,P])

    def energy(self,T):
        """Energy inside radius_cap and the flux through its surface.

        Parameters
        ----------
        T : ndarray
            Stress-energy components from `stress_energy_tensor`.

        Returns
        -------
        E : float
            Energy density integrated out to radius_cap.
        dE : float
            Energy carried through the cap surface in one timestep
            (surface area * momentum flux * dt).
        """
        E = radial_integrate(self._rt[0:self._Ncap],T[0,0:self._Ncap],self.d)
        dE = (sphere_solid_angle(self.d)*self._rt[self._Ncap]**(self.d-1)*
                T[1,self._Ncap]*self.dt)
        return E,dE

    def simulate_oscillon(self,fields,
                            plot_profile = False,
                            plot_energy_density = False,
                            plot_energy_shells = False,
                            saveTag = False,
                            num_E_steps = 10):
        """Evolve the field until the oscillon has radiated away.

        Time-steps the initial data until the energy inside radius_cap
        drops below 1% of its initial value, recording the energy and core
        amplitude every `num_E_steps` steps. Histories are stored on the
        instance as `E`, `core`, and `time`.

        Parameters
        ----------
        fields : ndarray, shape (3, N)
            Initial data from `initialize_field`.
        plot_profile, plot_energy_density, plot_energy_shells : bool
            Live diagnostic plots updated at each recording step.
        saveTag : bool
            Currently unused.
        num_E_steps : int
            Number of timesteps between recorded samples.
        """
        self._simulation_start()
        t = [0]
        SET = self.stress_energy_tensor(fields)
        E0,dE = self.energy(SET)
        E = 1.*E0
        energy_history = [E0]
        core = [fields[0][0]]
        n = 0
        while E > .01*E0:
            fields = self._timestep(fields)
            if np.mod(n,num_E_steps) == 0:
                SET = self.stress_energy_tensor(fields)
                E,dE = self.energy(SET)
                energy_history.append(E)
                core.append(fields[0][0])
                t.append(t[-1]+num_E_steps*self.dt)
                if plot_profile:
                    self._plot_profile(fields,3)
                if plot_energy_density:
                    self._plot_energy_density(SET)
                if plot_energy_shells:
                    self._plot_energy_shells(SET)
            n = n + 1
        self.E = np.array(energy_history)
        self.time = np.array(t)
        self.core = core

    def _plot_profile(self,fields,num_fields):
        """Live semilog plot of |phi|, |Phi|, |Pi| against r."""
        x = self.r#[0:50]
        plt.figure(1)
        plt.clf()
        for n in range(num_fields):
            plt.subplot(num_fields,1,n+1)
            y = np.abs(fields[n])#[0:50]
            plt.semilogy(x,y,'.',markersize = .5)
            #plt.ylim([-1,2.5])
            plt.xlim([0,self.r[-1]])
            #plt.xlabel('$r$')
            #plt.ylabel('$\phi$',rotation = 0)
        plt.draw()
        plt.pause(.0000000000001)

    def _plot_energy_density(self,T):
        """Live semilog plot of the energy density rho(r)."""
        x = self.r
        y = np.abs(T[0])
        plt.figure(2)
        plt.clf()
        plt.semilogy(x,y,',')
        plt.ylim([10**-12,100])
        plt.xlim([0,self.r[-1]])
        plt.xlabel('$r$')
        plt.ylabel('$\\rho$',rotation = 0)
        plt.draw()
        plt.pause(.0000000000001)

    def _plot_energy_shells(self,T):
        """Live semilog plot of the energy per radial shell, rho * a * r~^(d-1)."""
        x = self.r
        y = np.abs(T[0])*self._a*self._rtd
        plt.figure(3)
        plt.clf()
        plt.semilogy(x,y,',')
        plt.ylim([10**-20,100])
        plt.xlim([0,self.r[-1]])
        plt.xlabel('$r$')
        plt.ylabel('$\\rho r^d$',rotation = 0)
        plt.draw()
        plt.pause(.0000000000001)

    def radialFT(self,y):
        """Radial Fourier transform of y on the interior grid (r < radius_cap).

        The transform matrix is built lazily on the first call (see
        :func:`pyCE.math.radialFT_mat`) and cached, so the first call is
        slow and subsequent calls are a single matrix-vector product. The
        k-grid is stored as `self.k`.
        """
        if self._FT_tag:
            return np.dot(self._FTmat,y)
        else:
            self._FT_tag = True
            self._FTmat,self.k = radialFT_mat(self.d,self.r[1:self._Ncap])
            return np.dot(self._FTmat,y[1:self._Ncap])

def _L2_norm(y1,y2):
    """Root-mean-square difference between two arrays."""
    return np.sqrt(np.mean((y1-y2)**2))
