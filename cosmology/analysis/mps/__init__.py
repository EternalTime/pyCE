import numpy as np

def norm(x,y,dimension = 3):
    z = np.trapz(y*x**(dimension-1),x)
    return y/z

def modal_fraction(k,power_spectrum,dimension = 3,kmax = np.nan,full=True):
    #if there is a max k, then the arrays are truncated
    if not(np.isnan(kmax)):
        k = k[k <= kmax]
        power_spectrum = power_spectrum[0:len(k)]
    #normalizze the power spectrum
    mf = norm(k,power_spectrum,dimension = dimension)
    #if full=True, returns f(vec(k)), the power spectrum over directions
    #otherwise returns the radial power spectrum
    if full:
        return k,mf
    else:
        return k,mf*k**(dimension - 1 )

def KL_divergence(k,p,q,full = True,dimension = 3):
    q[q==0] = np.nan
    if full:
        return np.trapz(k**(dimension-1)*p*np.log2(p/q),k)
    else:
        return np.trapz(p*np.log2(p/q),k)

def entropy(k,p):
    return np.trapz(-p*np.log2(p))
