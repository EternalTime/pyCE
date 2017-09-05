import numpy as np

def norm(p):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   p = norm(p)
    ----------------------------------------------------------------------------
    INPUT:      p (array)
    ----------------------------------------------------------------------------
    OUTPUT:     p (array)
                Returns an L1-normalized array.
    ----------------------------------------------------------------------------
    """
    return p/np.nansum(p)

def modal_fraction(ell,Cl):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   mf = modal_fraction(ell,Cl)
    ----------------------------------------------------------------------------
    INPUT:      ell (integer array)
                Cl (array)
    ----------------------------------------------------------------------------
    OUTPUT:     mf (array)
                Calculates the modal fraction of an angular power spectrum.
                Incorporates the 2*ell+1 Jacobian factor coming from the sum
                over all the m's.
    ----------------------------------------------------------------------------
    """
    nCl = (2.*ell+1.)*Cl
    nCl[nCl < 0] = np.nan
    return norm(nCl)

def KL_divergence(p,q):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   kl = KL_divergence(p,q)
    ----------------------------------------------------------------------------
    INPUT:      p (array) a modal fraction
                q (array) a modal fraction
    ----------------------------------------------------------------------------
    OUTPUT:     kl (positive real number)
                Returns the Kullback-Liebler divergence from q to p. Note that
                divergence is not symmetric. For more on the information measure
                see: add website
                Integration uses the trapezoidal rule implemented in NumPy.
    ----------------------------------------------------------------------------
    """
    q[q==0] = np.nan
    pq = p/q
    return np.nansum(p*np.log2(pq))

def entropy(p):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   h = entropy(p)
    ----------------------------------------------------------------------------
    INPUT:      p (array) a modal fraction
    ----------------------------------------------------------------------------
    OUTPUT:     h (positive real number)
                Returns the Shannon Entropy of the p distribution. 
    ----------------------------------------------------------------------------
    """
    p[p<=0] = np.nan
    return np.nansum(-p*np.log2(p))
