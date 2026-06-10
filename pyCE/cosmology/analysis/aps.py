"""Information-theoretic analysis of angular power spectra.

Tools for CMB angular power spectra C_ell: forming the modal fraction (the
normalized distribution over multipoles, including the 2*ell+1 degeneracy),
Shannon entropy and Kullback-Leibler divergence of such distributions, and a
nonparametric (shrinkage) fit of a noisy spectrum to an orthogonal basis
following Aghamousa, Arjunwadkar & Souradeep (arXiv:1107.0516).
"""

import numpy as np
import tqdm

#----------------------------------------------------------------------FUNCTIONS

def norm(p):
    """L1-normalize an array (NaN-safe).

    Parameters
    ----------
    p : ndarray

    Returns
    -------
    ndarray
        p divided by its nansum, so the finite entries sum to one.
    """
    return p/np.nansum(p)

def modal_fraction(ell,Cl):
    """Modal fraction of an angular power spectrum.

    Weights each multipole by its 2*ell+1 degenerate m-modes and normalizes,
    giving the fraction of the total power carried by each ell. Negative
    entries (unphysical, e.g. noise-dominated estimates) are set to NaN.

    Parameters
    ----------
    ell : ndarray
        Multipole moments.
    Cl : ndarray
        Angular power spectrum C_ell. Modified in place where negative.

    Returns
    -------
    ndarray
        Normalized modal fraction over ell.
    """
    nCl = (2.*ell+1.)*Cl
    nCl[nCl < 0] = np.nan
    return norm(nCl)

def KL_divergence(p,q):
    """Kullback-Leibler divergence D(p || q) in bits.

    Note that the divergence is not symmetric in its arguments.

    Parameters
    ----------
    p : ndarray
        A modal fraction.
    q : ndarray
        A modal fraction. Zero entries are set to NaN in place and dropped
        from the sum.

    Returns
    -------
    float
        sum_i p_i * log2(p_i / q_i) over finite entries.
    """
    q[q==0] = np.nan
    pq = p/q
    return np.nansum(p*np.log2(pq))

def entropy(p):
    """Shannon entropy of a modal fraction, in bits.

    Parameters
    ----------
    p : ndarray
        A modal fraction. Non-positive entries are set to NaN in place and
        dropped from the sum.

    Returns
    -------
    float
        -sum_i p_i * log2(p_i) over finite entries.
    """
    p[p<=0] = np.nan
    return np.nansum(-p*np.log2(p))

#----------------------------------------------------------------NONPARAMETRIC FITTING

def basis_cos(j,x):
    """j-th orthonormal cosine basis function on [0, 1].

    Returns 1 for j = 0 and sqrt(2)*cos(j*pi*x) otherwise.
    """
    if j == 0:
        return np.ones(np.shape(x))
    else:
        return 1.4142135623730951*np.cos(j*np.pi*x)

def basis_leg(j,x):
    """j-th orthonormal Legendre polynomial rescaled to [0, 1]."""
    J = np.zeros(j+1)
    J[-1] = 1
    return np.sqrt((2*j+1.))*np.polynomial.legendre.Legendre(J,[0,1])(x)

def npf_makeU(x,basis):
    """Orthogonal design matrix for a nonparametric fit.

    Parameters
    ----------
    x : ndarray
        Sample points in [0, 1].
    basis : callable
        Basis function basis(j, x), e.g. `basis_cos` or `basis_leg`.

    Returns
    -------
    ndarray, shape (len(x), len(x))
        Column i holds basis(i, x)/sqrt(N), so that U is (approximately)
        orthogonal: U^T U ~ identity.
    """
    N = len(x)
    U = np.ones([N,N])
    for i in range(0,N):
        U[:,i] = basis(i,x)
    return U/np.sqrt(N)

def nonparametric_fit(data,error,U,lType = 'NSS',JRange = [1,100]):
    """Nonparametric shrinkage fit of noisy data to an orthogonal basis.

    Expands the data in the basis U, shrinks the coefficient vector, and
    selects the amount of shrinkage by minimizing an unbiased estimate of
    the risk. The algorithm follows arXiv:1107.0516v2.

    Parameters
    ----------
    data : ndarray
        Data vector of length N.
    error : ndarray
        One-sigma errors on the data. Used both for the inverse-variance
        weights and as a diagonal stand-in for the covariance matrix (the
        off-diagonal terms are set to zero until a full covariance is
        available).
    U : ndarray, shape (N, N)
        Orthogonal basis matrix from `npf_makeU`.
    lType : str
        Shrinkage scheme: 'NSS' keeps the first j coefficients (Nested
        Subset Selection); 'Fractional' applies monotone fractional
        weights 2**-i.
    JRange : list of int
        [minJ, maxJ], the range of shrinkage parameters scanned.

    Returns
    -------
    dict
        'nbf'  : the nonparametric best fit evaluated at the data points,
        'Risk' : the risk estimate at each J in the scan,
        'EDoF' : the effective degrees of freedom at each J.
    """
    minJ = JRange[0]
    maxJ = JRange[1]
    N = len(data)
    Y = data
    E = np.diag(1/error**2) #initialize inverse variance
    B = np.diag(error**2)/N #this should actually be the covariance matrix, but
    # until we get that we'll settle for just the variance and leave the off-
    # diagonal terms = 0
    x = (2.*np.array(list(range(N)))+1.)/(2.*N)
    #Use the U matrix in a few places to make the B, W, and Z matrices
    # (see the paper if this is confusing)
    B = np.dot(np.dot(np.transpose(U),B),U)
    Z = np.dot(np.transpose(U),Y)/np.sqrt(N)
    W = np.dot(np.transpose(U),np.dot(E,U))

    # implements the Nested Subset Selection choice for shrinkage
    # TODO: implement some other shrinkage choices (exponential?)
    print('\n---------------- Calculating Risk ---------------')
    if lType in ['NSS']:
        print('Shrinkage Method:         Nested Subset Selection\n')
        myD = lambda j:np.diag([1]*j+[0]*(N-j))
    elif lType in ['Fractional']:
        print('Shrinkage Method:         Fractional Monotone Shrinkage\n')
        myD = lambda j:np.diag([2.0**-i for i in range(N)])

    R = np.zeros(maxJ-minJ+1)
    EDoF = np.zeros(np.shape(R))

    for j in tqdm.tqdm(range(minJ,maxJ+1)):
        D = myD(j)
        Db = np.eye(N)-D
        temp = [np.dot(np.dot(np.dot(np.dot(np.transpose(Z),Db),W),Db),Z),
                np.trace(np.dot(np.dot(np.dot(D,W),D),B)),
                -np.trace(np.dot(np.dot(np.dot(Db,W),Db),B))]
        R[j-minJ] = sum(temp)
        EDoF[j-minJ] = j#sum(np.diagonal(D))
    J = list(R).index(min(R[1::]))+minJ

    print('\nNPfit optimized at J = ' + str(J))
    return {'nbf':np.sqrt(N)*np.dot(U,([1]*J+[0]*(N-J))*Z),'Risk':R,'EDoF':EDoF}
