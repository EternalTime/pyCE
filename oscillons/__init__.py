import numpy as np

class bubble:
    def __init__(self,alpha,d,N,bubtype = 'quantum'):
        eps = np.finfo(float).eps
        s = lambda d:2*np.pi**(d/2.0)/np.math.gamma(d/2.0)
        self.d = d
        self.alpha = np.float128(alpha)
        self.dx = np.float128(.005)

        self.Bmax = (self.alpha+np.sqrt(self.alpha**2-4.))/2.
        self.Bmin = .5*self.Bmax#0*(2.*self.alpha-np.sqrt(4.*self.alpha**2-18.))/3.
        if bubtype == 'quantum':
            self.shootFor(N,d)
        elif bubtype == 'thermal':
            self.shootFor(N,d-1)
        else:
            print('bubtype must be either (quantum) or (thermal)')
            return None
        idx = self.xi > self.xi[np.abs(self.B)==min(np.abs(self.B))][0]
        self.B[idx] = self.B[idx]*0+eps

        self.PEdens = .5*self.B**2 - (self.alpha*self.B**3)/3.0 + .25*self.B**4
        self.GEdens = .5*self.DB**2
        self.rho = self.PEdens + self.GEdens
        self.PE = s(d)*np.trapz(self.PEdens*self.xi**(d-1),self.xi)
        self.GE = s(d)*np.trapz(self.GEdens*self.xi**(d-1),self.xi)
        self.E  = self.PE+self.GE
        self.S4 = s(d+1)*np.trapz(self.rho*self.xi**d,self.xi)
        self.FourierTransform()
        f = np.abs(self.denFT)**2
        self.mf = f/(s(d)*np.trapz(f*self.ki**(self.d-1),self.ki))
        self.Sc = -s(d)*np.trapz(self.ki**(d-1)*self.mf*np.log(eps+self.mf),self.ki)

    def shootFor(self,N,d):
        deltaB = np.float128(.5*(self.Bmax - self.Bmin))
        self.B0 = np.float128(self.Bmin + deltaB)
        increment = np.float128(1.0)
        self.B = np.array([np.float128(10)])
        lastB = self.B[map(lambda x:not(x),np.isnan(self.B))][-1]
        while np.abs(lastB)>1e-8:
            self.RK4(N,d)
            lastBnew = self.B[map(lambda x:not(x),np.isnan(self.B))][-1]
            #print(self.B0,lastBnew)
            if lastBnew == lastB:
                breakTag = breakTag + 1
                if breakTag > 3:
                    break
            else:
                breakTag = 0
            lastB = lastBnew
            increment = increment/np.float128(2.)
            self.B0 = self.B0 + np.float128(np.sign(lastB))*deltaB*increment

    def RK4(self,N,d):
        n = 0
        B1 = lambda xi,b,b1: b1
        B2 = lambda xi,b,b1: -d*b1/xi + b - self.alpha*b**2 + b**3
        B = [self.B0, self.B0]
        DB = [np.float128(0.0),np.float128(0.0)]
        xi = [np.float128(0.0),self.dx]
        while n < N-2:
			try:
				n = n + 1
				k31 = B1(xi[-1],B[-1],DB[-1])
				k41 = B2(xi[-1],B[-1],DB[-1])

				k32 = B1(xi[-1]+.5*self.dx,B[-1] + .5*self.dx*k31,DB[-1] + .5*self.dx*k41)
				k42 = B2(xi[-1]+.5*self.dx,B[-1] + .5*self.dx*k31,DB[-1] + .5*self.dx*k41)

				k33 = B1(xi[-1]+.5*self.dx,B[-1] + .5*self.dx*k32,DB[-1] + .5*self.dx*k42)
				k43 = B2(xi[-1]+.5*self.dx,B[-1] + .5*self.dx*k32,DB[-1] + .5*self.dx*k42)

				k34 = B1(xi[-1]+self.dx,B[-1]+self.dx*k33,DB[-1]+self.dx*k43)
				k44 = B2(xi[-1]+self.dx,B[-1]+self.dx*k33,DB[-1]+self.dx*k43)

				xi.append(xi[-1]+self.dx)
				B.append(B[-1] + self.dx/6.*(k31+2.*k32+2.*k33+k34))
				DB.append(DB[-1] + self.dx/6.*(k41+2.*k42+2.*k43+k44))
			except OverflowError:
				print "Overflow occured at n = ", n
				break
        self.B = np.array(B)
        self.DB = np.array(DB)
        self.xi = np.array(xi)

    def FourierTransform(self):
        c = 2*np.pi**(self.d/2.0)/np.math.gamma(self.d/2.0)/(2*np.pi)**(self.d/2.0)
        f = self.rho
        r = self.xi
        k = r
        ft = c*np.trapz(np.sin(np.outer(k,r))*(r**(self.d-2)*f),k)/k
        ft[0] = c*np.trapz(f*r**(self.d-1),r)
        self.denFT = ft
        self.ki = k
