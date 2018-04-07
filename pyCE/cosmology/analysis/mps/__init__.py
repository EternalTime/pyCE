import numpy as np

#----------------------------------------------------------------------FUNCTIONS

def norm(x,y,d = 3):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   y = norm(x,y,[d])
    ----------------------------------------------------------------------------
    INPUT:      x (array)
                y (array)
            [optional arguments]
                d (positive integer)
    ----------------------------------------------------------------------------
    OUTPUT:     y (array)
    ----------------------------------------------------------------------------
    Integrates y over x with a measure coming from the Jacobian in d dimensions.
    Uses the result to normalize y, and returns it with the Jacobian factor.
    Integration is done using the trapezoidal rule implemented by numpy.
    ----------------------------------------------------------------------------
    """
    return y/np.trapz(y*x**(d-1),x)

def modal_fraction(x,y,d = 3,xmax = np.inf,xmin = 0):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   x,y = modal_fraction(x,y,[d,xmax,xmin])
    ----------------------------------------------------------------------------
    INPUT:      x (array)
                y (array)
            [optional arguments]
                d (positive integer)
                xmax (positive real number)
                xmin (positive real number)
    ----------------------------------------------------------------------------
    OUTPUT:     x,y (arrays)
    ----------------------------------------------------------------------------
    Normalizes y over the domain set by xmin and xmax, and returns both the
    valid x and y. Integration is done using the trapezoidal rule implemented
    by numpy.
    ----------------------------------------------------------------------------
    """
    #finds the valid k-modes
    idx = (x>=xmin)&(x<=xmax)&(y>0)
    x = x[idx]
    y = y[idx]
    #normalize the power spectrum
    mf = norm(k,y,d = d)
    return x,y*x**(d - 1 )

def KL_divergence(x,p,q,xmin = 0,xmax = np.inf):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   kl = KL_divergence(x,p,q,[xmin,xmax])
    ----------------------------------------------------------------------------
    INPUT:      x (array)
                p (array) a modal fraction
                q (array) a modal fraction
            [optional arguments]
                xmin (positive real number)
                xmax (positive real number)
    ----------------------------------------------------------------------------
    OUTPUT:     kl (positive real number)
    ----------------------------------------------------------------------------
    Returns the Kullback-Liebler divergence from q to p. Note that divergence is
    not symmetric. For more on the information measure see:
    add website
    Integration uses the trapezoidal rule implemented in NumPy.
    ----------------------------------------------------------------------------
    """
    #finds the valid k-modes
    idx = (x>=xmin)&(x<=xmax)&(p>0)&(q>0)
    x,p,q = x[idx],p[idx],q[idx]
    return np.trapz(p*np.log2(p/q),x)

def entropy(x,p):
    """
    ----------------------------------------------------------------------------
    FUNCTION:   h = entropy(k,p)
    ----------------------------------------------------------------------------
    INPUT:      x (array)
                p (array) a modal fraction
    ----------------------------------------------------------------------------
    OUTPUT:     h (positive real number)
    ----------------------------------------------------------------------------
    Calculates the differential entropy of the distribution,integrating using
    the trapezoidal rule implemented in NumPy.
    ----------------------------------------------------------------------------
    """
    return -np.trapz(p*np.log2(p),x)
