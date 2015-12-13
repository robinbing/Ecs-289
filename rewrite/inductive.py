import numpy as np
from numpy.linalg import svd
from CompResidual import CompResidual
from scipy.sparse.linalg import svds
# rewrite inductive_mc function
def inductive(X, S, kx, nCluster, lamBda, maxIt):
    # X is feature matrix
    # S is sparse matrix

    UX, SX, VX = svds(X, kx)
    U = np.zeros((kx, 1))
    maxiter = maxIt
    D = np.zeros((1, 1))

    for i in xrange(maxiter):
        # V = U*D
        if i == 0:
            R = CompResidual(S, UX.T, UX.dot(U).dot(D).dot(U.T).T)
        grad = -UX.T.dot(R.A).dot(UX)
        uu, ss, vv = svd(U.dot(D).dot(U.T) - grad)
        U = uu

        imax = np.vectorize(max)
        D = np.diag(imax(ss - lamBda, 0))
        for i in range(nCluster, kx):
            D[i, i] = 0
        R = CompResidual(S, UX.T, UX.dot(U).dot(D).dot(U.T).T)
    U = UX.dot(U)
    return [U, D]

