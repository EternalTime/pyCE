import tqdm
import numpy as np
import matplotlib.pyplot as plt

def nonparametric_fit(data,error,basis,lType = 'NSS',maxJ = 100):
    """
    ----------------------------------------------------------------------------
    FUNCTION:       dic = nonparametric_fit(data, error, basis,[lType, maxJ])
    ----------------------------------------------------------------------------
    INPUT:          data    (array)
                    error   (array)
                    basis   (function: arguments(j,x))
    ----------------------------------------------------------------------------
    OUTPUT:         dic     (dictionary, keys: nbf, Risk, EDoF)
                    Outputs a dictionary containing the nonparametric best fit
                    (nbf) of the data with respect to the basis. Also returns
                    the risk function (Risk) and the effective degrees of
                    freedom (EDoF). The algorithm follows 1107.0516v2.
    ----------------------------------------------------------------------------
    """
    N = len(data)
    Y = data
    E = np.diag(1/error**2) #initialize inverse variance
    B = np.diag(error**2)/N #this should actually be the covariance matrix, but
    # until we get that we'll settle for just the variance and leave the off-
    # diagonal terms = 0
    x = (2.*np.array(list(range(N)))+1.)/(2.*N)
    # Make the Unitary matrix from the basis. if the basis is wrong then the
    # resulting matrix will not be orthogonal and the algorithm will output
    # bullshit. TODO: add a check to ensure U is orthogonal
    U = np.ones([N,N])
    for i in range(1,N):
        U[:,i] = basis(i,x)
    U = U/np.sqrt(N)
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
        R = np.zeros(maxJ)
        EDoF = np.zeros(np.shape(R))

        for j in tqdm.tqdm(range(1,maxJ+1)):
            D = np.diag([1]*j+[0]*(N-j))
            Db = np.eye(N)-D
            temp = [np.dot(np.dot(np.dot(np.dot(np.transpose(Z),Db),W),Db),Z),
                    np.trace(np.dot(np.dot(np.dot(D,W),D),B)),
                    -np.trace(np.dot(np.dot(np.dot(Db,W),Db),B))]
            R[j-1] = sum(temp)
            EDoF[j-1] = sum(np.diagonal(D))
        J = list(R).index(min(R[1::]))+1

        print('\nNPfit optimized at J = ' + str(J))
        return {'nbf':np.sqrt(N)*np.dot(U,([1]*j+[0]*(N-j))*Z),'Risk':R,'EDoF':EDoF}


def basis_cos(j,x):
    if j == 0:
        return np.ones(np.shape(x))
    else:
        return 1.4142135623730951*np.cos(j*np.pi*x)
