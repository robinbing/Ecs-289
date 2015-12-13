import query
import math

import os
import pandas as pd
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
dataname = 'diabetes'

yorigin, xorigin = rp.svm_read_problem(dataname)
yorigin = array(yorigin)
xorigin = rp.buildXmatix(xorigin)

n_cluster = len(set(yorigin))
n_total = len(yorigin)


# constraints matrix for all, namely S

def groundT(yorigin, l = n_total):
    groundTruth = -ones([l, l])
    dic = {}
    yClass = set(yorigin)
    for i in range(len(yClass)):
        cl = yClass.pop()
        dic[cl] = np.where(yorigin == cl)[0]
    for clas, index in dic.items():
        for i in index:
            groundTruth[[i], index] = 1
    return groundTruth

groundTruth = groundT(yorigin)

# create constraints matrix
Y = yorigin
X = xorigin

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

def testcluster_query(lamBda=0.00001, ncluster=2, kx=5, rate=0.001, func=sign,
                      query_func = query.queryZero, queryNum=50):
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
    #queryNum = math.floor(query_rate * n_total)
    seed = -1
    r_query, l_query = query_func(est_S, queryNum, r_index, l_index, seed)
    S = zeros([n_total, n_total])
    S[r_index +  r_query, l_index +  l_query] = groundTruth[r_index +  r_query, l_index +  l_query ]

    # function varctorization
    sign = func
    sign = np.vectorize(sign)
    S = sign((S + S.T))
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
    #for i in range(ncluster):
     #   pos = np.where(label == i)[0]
      #  for j in pos:
       #     for k in pos:
        #        predictA[j, k] = 1

    for i in range(n_total):
        for j in range(n_total):
            if label[i] == label[j]:
                predictA[i, j] = 1
    #
    accbias = sum(predictA != groundTruth).sum() / float(np.product(groundTruth.shape))
    print('sample rate: ', rate, "  ", "query rate:", queryNum, "err: ", accbias)
    return accbias



start = time.time() 
# mushroom rate=[0.01,0.0001,0.000001]
# queryNum = 16 32 64 128 326
rates = [0.01,0.0001,0.000001]
nums = [20, 40, 60, 80, 200, 400, 800, 2000, 4000, 6000, 8000]
result = []
for i in range(5):
    for rate in rates:
        for num in nums:
            err1 = testcluster_query(rate=rate, query_func=query.queryZero, ncluster=n_cluster,
                                     queryNum=num)
            err2 = testcluster_query(rate=rate, query_func=query.queryRand, ncluster=n_cluster,
                                     queryNum=num)
            temp = pd.DataFrame([i,rate, num, err1, err2], index=['iter','rate', 'num', 'not-rand', 'rand']).T
            result.append(temp)
pd.concat(result).to_csv(dataname+'.csv',header = True)