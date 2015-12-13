from scipy import sparse
import numpy as np

def CompResidual(A, Z, E):
    # the first matrix(original similarity matrix 'S') is a sparse matrix
    # the second matrix is a dense one ('Z' in paper)
    # the third...
    res = []
    row0, col0, entry = sparse.find(A)
    m, n = A.get_shape()
    for i in range(len(entry)):
        row = row0[i]
        col = col0[i]
        res.append(entry[i] - np.dot(Z[:, row], E[:, col]))
    A = sparse.coo_matrix((res, (row0, col0)), shape=(m, n)).tocsr()
    return A
