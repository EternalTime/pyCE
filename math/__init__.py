import numpy as np
import scipy.special as sp

def radialFT(d,f,r):
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    if d>1:
        a      = float(d)/2.0
        ft     = np.zeros(np.shape(k))
        ft[1:] = np.sqrt(np.pi)*(2.0**(a-2))*(k[1:]**(1-a))*np.trapz(f*sp.jv(a-1,np.outer(k[1:],r))*(r**a),r)
        #This uses finite differences to get the value at k=0
        ft[0]  = sum(np.array([287/48.0, -(61/4.0), 1033/48.0, -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])*ft[1:8])
    else:
        ft = np.trapz(np.cos(np.outer(k,r))*f,r)
    #normalizes to ensure Plancheral's theorem holds
    ft = ft*np.sqrt(radial_integrate(r,np.abs(f)**2,d)/radial_integrate(k,np.abs(ft)**2,d))
    return ft,k

def sphere_solid_angle(d):
    return 2*np.pi**(d/2.0)/np.math.gamma(d/2.0)

def fourier_factor(d):
    return 1.0/(2*np.pi)**(d/2.0)

def radial_integrate(r,y,d):
    return sphere_solid_angle(d)*np.trapz(y*r**(d-1),r)

def normalize(r,y,d):
    return y/radial_integrate(r,y,d)
