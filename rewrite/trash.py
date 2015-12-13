__author__ = 'admin'
import numpy as np
def sparseMat(mat):
    # transform a matrix to sparse matrix format
    # pr record all the non-zero entry
    # ir the index of the row
    # jc[j+1] - j[j] indicate the # of non-zero in a j th column
    row, col = mat.shape
    pr = []
    ir = []
    jc = []
    # traversal, find non-zero entry
    num_entry = 0
    for i in xrange(col):
        jc.append(num_entry)
        for j in xrange(row):
            if mat[j, i] != 0:
                pr.append(mat[j, i])
                ir.append(j)
                num_entry += 1
    jc.append(num_entry)
    return [pr, ir, jc]

# X = np.matrix('0,0,2,1,0;0,1,0,0,0;1,0,0,0,1;1,1,0,0,1;2,3,0,0,0')
# a, b, c = sparseMat(X)

def CompResidual(A, Z, E):
    # the first matrix(original similarity matrix 'S') is a sparse matrix
    # the second matrix is a dense one ('Z' in paper)
    # the third... is kind of evil stuff in code, better ignore it
    pr, ir, jc = A
    row1, col1 = Z.shape
    row2, col2 = E.shape

    k = row1

    for



import scipy.sparse
ha = scipy.sparse(X)


##################################
import pandas
S = zeros([9, 9])
index = pandas.Series([41,48,59,9,7,36,52,63,39,38,53,23,35,32,75,58,49,55,14,1,78,22,13,
                       37,51,56,17,67,16,2,6,26])-1
r_index = []
l_index = []
for item in index:
    r_index.append(item // 9)
    l_index.append(item % 9)
S[r_index,l_index] = groundTruth[r_index,l_index]

S = sparse.csr_matrix(S)
####################################3

S = zeros([n_total, n_total])
r_index, l_index = randindex(n_total, rate, -1)
S[r_index, l_index] = groundTruth[r_index, l_index]
# function varctorization
sign = func
sign = np.vectorize(sign)
S = sign(S + S.T)
S = sparse.csr_matrix(S)
#
U, D = idc.inductive(X, S, kx, ncluster, lamBda, 50)
est_S = U.dot(D).dot(U.T)
queryNum = math.floor(query_rate * n_total)
seed = -1
r_query, l_query = query_func(est_S, queryNum, r_index, l_index, seed)
sign = func
sign = np.vectorize(sign)
S = sign((S + S.T).A)
S = sparse.csr_matrix(S)
#
U, D = idc.inductive(X, S, kx, ncluster, lamBda, 50)


# kmeans
Xresult = matrix(U[:, 0:ncluster])
Xresult = Xresult / (matlib.repmat(np.sqrt(np.square(Xresult).sum(axis=1)),
                                   1,
                                   ncluster) * 1.0)
label = KMeans(n_clusters=ncluster).fit_predict(Xresult)
label = array(label)
predictA = - ones([n_total, n_total])
#
for i in range(ncluster):
    pos = np.where(label == i)[0]
    for j in pos:
        for k in pos:
            predictA[j, k] = 1
    #
accbias = sum(predictA != groundTruth).sum() / float(np.product(groundTruth.shape))
print 'sample rate: ', rate, "  ", "err: ", accbias


S = zeros([n_total, n_total])
r_index, l_index = randindex(n_total, rate, -1)
S[r_index, l_index] = groundTruth[r_index, l_index]
# function varctorization
sign = func
sign = np.vectorize(sign)
S = sign(S + S.T)
S = sparse.csr_matrix(S)
#
U, D = idc.inductive(X, S, kx, ncluster, lamBda, 50)

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
la = zip(a,b)

lamBda=0.00001, ncluster=n_cluster, kx=50, rate=0.001, func=sign,
query_func = query.queryZero, queryNum=50