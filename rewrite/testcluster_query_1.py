import query
import math

import os
import readProblem as rp
import inductive as idc
import numpy as np
from numpy import ones, zeros, hstack, array, matlib, matrix
from scipy import sparse
from sklearn.cluster import KMeans
import random
import time
# add you own path
#os.chdir('D:/fall 2015/ecs 289/rewrite')
#os.chdir('/home/ybluo/rewrite')

yorigin, xorigin = rp.svm_read_problem('mushrooms')
yorigin = array(yorigin)
xorigin = rp.buildXmatix(xorigin)

# number of class 1&2 separately
ny1 = sum(yorigin == 1)
ny2 = sum(yorigin == 2)
# total length
n_total = len(yorigin)

# create constraints matrix
Y = np.concatenate((yorigin[yorigin == 1], yorigin[yorigin == 2]))
X = sparse.hstack((xorigin[yorigin == 1, ].T,
                   xorigin[yorigin == 2, ].T), format='csr')
X = X.T

# constraints matrix for all, namely S
groundTruth = hstack((hstack((ones([ny1, ny1]), -ones([ny1, ny2]))).T,
                      hstack((-ones([ny2, ny1]), ones([ny2, ny2]))).T))

# random create incomplete matrix
# sign function same to matlab one
def sign(entry):
    if entry > 0:
        return 1
    if entry < 0:
        return -1
    return 0

# randomly select the entry index
def randindex(num, rate, seed):
    if seed != -1:
        random.seed(seed)
    r_index = []
    l_index = []
    index = random.sample(range(num * num), int(num * num * rate))
    for item in index:
        r_index.append(item // num)
        l_index.append(item % num)
    return [r_index, l_index]

def testcluster_query(lamBda=0.00001, ncluster=2, kx=50, rate=0.001, func=sign, query_func = query.queryZero, query_rate = 0.001 ):
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
    #
    # active query
    ### ADD: how to calculate the estimated similarity matrix?
    est_S = U.dot(D).dot(U.T)
    queryNum = math.floor(query_rate * n_total)
    seed = -1
    r_query, l_query = query_func(est_S, queryNum, r_index, l_index, seed)
    S[r_index + r_query , l_index + l_query] = groundTruth[r_index + r_query, l_index + l_query]

    # function varctorization
    sign = func
    sign = np.vectorize(sign)
    S = sign(S + S.T)
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
    print 'sample rate: ', rate,'query rate',query_rate*rate, "err: ", accbias
    # return accbias

start = time.time()
print 'query zero'
rlist = [0.001, 0.00001, 0.000005, 0.000001]
rlist2 = [100, 10, 1, 0.1, 0.01]
for i in range(3):
	for x in rlist:
		for y in rlist2:
			testcluster_query(rate=x, query_func = query.queryZero, query_rate = y)

#[testcluster_query(rate=x, query_func = query.queryRand, query_rate = 0.001) for x,y in com]
print 'elapse time:', time.time() - start
