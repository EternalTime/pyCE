import numpy as np
import scipy.special as sp

class critical_bubble:
    #Remember d is the number of SPATIAL dimensions
    def __init__(self,asymmetry_factor,dimension,N):
        eps     = np.finfo(np.float128).eps
        self.d  = dimension
        self.g  = np.float128(asymmetry_factor)
        self.dr = np.float128(.01)
        self.dk = self.dr
        alpha   = np.float128(3/np.sqrt(2)*(1+asymmetry_factor))
        self.alpha = alpha

        self.__generate_bubble_profile__(alpha,N)

        self.__get_energy_profiles__(alpha)

        self.__get_euclidean_action__()

        self.__get_entropy__()


    def __generate_bubble_profile__(self,alpha,N):
        #Set up the bounds for the shooting method
        self.Bmax = np.float128((alpha+np.sqrt(alpha**2-4.))/2.)
        self.Bmin = np.float128((2.*alpha-np.sqrt(4.*alpha**2-18.))/3.)
        self.__shootFor__(N,self.d)

    def __get_energy_profiles__(self,alpha):
        self.PEdens = .5*self.B**2 - (alpha*self.B**3)/3.0 + .25*self.B**4
        self.GEdens = .5*self.DB**2
        self.PE     = radial_integrate(self.r,self.PEdens,self.d)
        self.GE     = radial_integrate(self.r,self.GEdens,self.d)
        self.rho    = self.PEdens + self.GEdens
        self.E      = self.PE+self.GE

    def __get_euclidean_action__(self):
        self.Se = radial_integrate(self.r,self.rho,self.d+1)

    def __get_entropy__(self):
        self.denFT, self.k = radialFT(self.d,self.rho,self.r)
        f       = np.abs(self.denFT)**2
        self.mf = f/max(f)
        self.Sc = -radial_integrate(self.k,self.mf*np.log(np.finfo(np.float128).eps+self.mf),self.d)

    def __shootFor__(self,N,d):
        isnotnan  = lambda x:map(lambda y:not(y),np.isnan(x))
        deltaB    = np.float128(.5*(self.Bmax - self.Bmin))
        self.B0   = np.float128(self.Bmin + deltaB)
        increment = np.float128(1.0)
        self.B    = np.array([np.float128(10)])
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
            increment = increment/np.float128(2.)
            self.B0   = self.B0 + np.float128(np.sign(lastB))*deltaB*increment
        idx = map(lambda x,y:x&y,isnotnan(np.log(self.B)),isnotnan(np.log(-self.DB+np.finfo(np.float128).eps)))
        self.B     = np.array(map(float,self.B[idx]))
        self.DB    = np.array(map(float,self.DB[idx]))
        self.r     = np.array(map(float,self.r[idx]))
        self.r[0]  = 0.0
        self.DB[0] = 0.0

    def __RK4__(self,N,d):
        n  = 0
        B1 = lambda r,b,b1: b1
        B2 = lambda r,b,b1: -d*b1/r + b - self.alpha*b**2 + b**3
        B  = [self.B0, self.B0]
        DB = [-np.finfo(np.float128).eps,-np.finfo(np.float128).eps]
        r  = [np.finfo(np.float128).eps,self.dr]
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
				print "Overflow occured at n = ", n
				break
        self.B  = np.array(B)
        self.DB = np.array(DB)
        self.r  = np.array(r)

    #def __FourierTransform__(self):
    #    c = sphere_solid_angle(self.d)*fourier_factor(self.d)
    #    self.dk = .05*np.pi/self.r[-1]
    #    k = np.array(range(len(self.r)))*self.dk
    #    ft = c*np.trapz(
    #            np.sin(np.outer(k,self.r))*(self.r**(self.d-2)*self.rho)
    #            ,k)/k
    #    ft[0] = c*np.trapz(self.rho*self.r**(self.d-1),self.r)
    #    self.denFT = ft
    #    self.k = k

def radialFT(d,rho,r):
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    if d>1:
        a      = float(d)/2.0
        ft     = np.zeros(np.shape(k))
        ft[1:] = np.sqrt(np.pi)*(2.0**(a-2))*(k[1:]**(1-a))*np.trapz(rho*sp.jv(a-1,np.outer(k[1:],r))*(r**a),r)
        #This uses finite differences to get hte value at k=0
        ft[0]  = sum(np.array([287/48.0, -(61/4.0), 1033/48.0, -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])*ft[1:8])
    else:
        ft = np.trapz(np.cos(np.outer(k,r))*rho,r)
    #normalizes to ensure Plancheral's theorem holds
    ft = ft*np.sqrt(radial_integrate(r,np.abs(rho)**2,d)/radial_integrate(k,np.abs(ft)**2,d))
    return ft,k

def sphere_solid_angle(d):
    return 2*np.pi**(d/2.0)/np.math.gamma(d/2.0)

def fourier_factor(d):
    return 1.0/(2*np.pi)**(d/2.0)

def radial_integrate(x,y,d):
    return sphere_solid_angle(d)*np.trapz(y*x**(d-1),x)
