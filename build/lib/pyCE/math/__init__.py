import numpy as np
import scipy.special as sp
from numpy.matlib import repmat

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

def radialFT_mat(d,r):
    k = np.array(range(5*len(r)))*np.pi/(10*r[-1])
    F = np.zeros([len(k),len(r)])
    dr0 = [np.diff(r)[0]/2]
    if d>1:
        a           = float(d)/2.0
        F[1:,:]     = (np.sqrt(np.pi)*(2.0**(a-2))
                        *repmat(k[1:]**(1-a),len(r),1).T
                        *(sp.jv(a-1,np.outer(k[1:],r))*(r**a))
                        *repmat(np.array(dr0+list(np.diff(r))),len(k)-1,1))
        F0          = np.eye(len(k))
        F0[0,0:8]   = np.array([0, 287/48.0, -(61/4.0), 1033/48.0,
                                -(109/6.0), 147/16.0, -(31/12.0), 5/16.0])
        F = np.dot(F0,F)
    else:
        F = np.cos(np.outer(k,r))
    return F,k

def sphere_solid_angle(d):
    return 2*np.pi**(d/2.0)/np.math.gamma(d/2.0)

def fourier_factor(d):
    return 1.0/(2*np.pi)**(d/2.0)

def radial_integrate(r,y,d):
    return sphere_solid_angle(d)*np.trapz(y*r**(d-1),r)

def normalize(r,y,d):
    return y/radial_integrate(r,y,d)
