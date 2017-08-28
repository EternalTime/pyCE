import numpy as np

def norm(p):
    return p/np.nansum(p)

def modal_fraction(ell,Cl):
    nCl = (2*ell+1)*Cl
    nCl[nCl < 0] = np.nan
    return norm(nCl)

def KL_divergence(p,q):
    q[q==0] = np.nan
    pq = p/q
    return np.nansum(p*np.log2(pq))

def entropy(p):
    p[p<=0] = np.nan
    return np.nansum(-p*np.log2(p))
