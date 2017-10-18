import numpy as np

class bubble:
    def __init__(self,alpha,d):
        self.d = d
        self.alpha = alpha
        self.dx = .01
        self.Vmax = (self.alpha+np.sqrt(self.alpha**2-4.))/2.
        self.Vmin = .5*self.Vmax#0*(2.*self.alpha-np.sqrt(4.*self.alpha**2-18.))/3.
        self.shootFor(3000,100)
        self.PEdens = .5*self.B**2 - self.alpha*self.B**3/3. + .25*self.B**4
        self.GEdens = (d-2)*.5*self.DB**2/d
        self.rho = self.PEdens + self.GEdens
        self.PE = sum(self.PEdens*self.xi**(d-1))*self.dx
        self.GE = sum(self.GEdens*self.xi**(d-1))*self.dx
        self.E  = self.PE+self.GE
        self.FourierTransform()

    def shootFor(self,N,lim):
        deltaV = .5*(self.Vmax - self.Vmin)
        self.B0 = self.Vmin + deltaV
        increment = 1
        m = 0
        while m < lim:
            m = m + 1
            self.RK4(N)
            if self.B[-1] > 0:
                delta = 0
            else:
                delta = 1
            increment = increment/2.
            self.B0 = self.B0 + ((-1)**delta)*deltaV*increment

    def RK4(self,N):
        n = 0
        B1 = lambda xi,b,b1: b1
        B2 = lambda xi,b,b1: -(self.d-1)*b1/xi + b - self.alpha*b**2 + b**3
        B = [self.B0, self.B0]
        DB = [0.0,0.0]
        xi = [0.0,self.dx]
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
				#print "Overflow occured at n = ", n
				break
        self.B = np.array(B)
        self.DB = np.array(DB)
        self.xi = np.array(xi)

    def FourierTransform(self):
		f, domain, dx = self.rho, self.xi, self.dx
		maxK = np.pi/(self.dx*10)
		N = len(domain)
		dk = float(maxK)/float(N)
		kdomain = np.array([dk*n for n in range (0,N)])
		matValues = domain*f*np.sin(np.outer(kdomain,domain))
		ftransform = [sum(domain**(self.d-1)*f)]
		for n in range(1,len(kdomain)):
			ftransform.append(sum(matValues[n]))
		ftransform[1:] = ftransform[1:]/kdomain[1:]
		self.dk = dk
		self.denFT = np.array(ftransform)
		self.ki = np.array(kdomain)
