import tqdm
import numpy as np
import matplotlib.pyplot as plt

def basis_cos(j,x):
    if j == 0:
        return np.ones(np.shape(x))
    else:
        return 1.4142135623730951*np.cos(j*np.pi*x)

def basis_leg(j,x):
    J = np.zeros(j+1)
    J[-1] = 1
    return np.sqrt((2*j+1.))*np.polynomial.legendre.Legendre(J,[0,1])(x)

def npf_makeU(x,basis):
    N = len(x)
    U = np.ones([N,N])
    for i in range(0,N):
        U[:,i] = basis(i,x)
    return U/np.sqrt(N)

def nonparametric_fit(data,error,U,lType = 'NSS',JRange = [1,100]):
    """
    ----------------------------------------------------------------------------
    FUNCTION:       dic = nonparametric_fit(data, error, basis,[lType, JRange])
    ----------------------------------------------------------------------------
    INPUT:          data    (array)
                    error   (array)
                    U       (array) orthogonal matrix made from basis
    ----------------------------------------------------------------------------
    OUTPUT:         dic     (dictionary, keys: nbf, Risk, EDoF)
                    Outputs a dictionary containing the nonparametric best fit
                    (nbf) of the data with respect to the basis. Also returns
                    the risk function (Risk) and the effective degrees of
                    freedom (EDoF). The algorithm follows 1107.0516v2.
    ----------------------------------------------------------------------------
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
        R = np.zeros(maxJ-minJ+1)
        EDoF = np.zeros(np.shape(R))

        for j in tqdm.tqdm(range(minJ,maxJ+1)):
            D = np.diag([1]*j+[0]*(N-j))
            Db = np.eye(N)-D
            temp = [np.dot(np.dot(np.dot(np.dot(np.transpose(Z),Db),W),Db),Z),
                    np.trace(np.dot(np.dot(np.dot(D,W),D),B)),
                    -np.trace(np.dot(np.dot(np.dot(Db,W),Db),B))]
            R[j-minJ] = sum(temp)
            EDoF[j-minJ] = j#sum(np.diagonal(D))
        J = list(R).index(min(R[1::]))+minJ

        print('\nNPfit optimized at J = ' + str(J))
        return {'nbf':np.sqrt(N)*np.dot(U,([1]*j+[0]*(N-j))*Z),'Risk':R,'EDoF':EDoF}
