import numpy as np
import random


def queryZero(est_S, queryNum, r_known, c_known, seed = None):
    """
    query some entries from the groud truth matrix, and feed to the 
    @est_S: estimated similarity matrix
    @queryNum: number of total queries from groundTruth matrix
    @r_known: list of int. row indox of known constraints
    @c_known: list of int. col indox of known constraints

    @return:
    @tuple of two elements, each element is a list
    @r_query: row index in the query
    @c_query: column index in the query
    """
    n, m = est_S.shape
    order = np.argsort(abs(est_S), axis = None)
#    topN = order[:queryNum]
#    groundTruth[topN]
    r_query = []
    c_query = []
    q_count = 0
    i = 0
    #
    while q_count < queryNum and i < m*n:
        r = order[i] // m
        c = order[i] % m
#
        if (r, c) not in zip(r_known, c_known):
            r_query.append(r)
            c_query.append(c)
            q_count += 1
#
        i += 1
#
    if q_count < queryNum: # not enough query left
        print("Warning: # of queries is less than requested.")
#
#
    return (r_query, c_query)


def queryRand(est_S, queryNum, r_known, c_known, seed = -1):
    """
    query some entries from the groud truth matrix, and feed to the 
    @est_S: estimated similarity matrix
    @queryNum: number of total queries from groundTruth matrix
    @r_known: list of int. row indox of known constraints
    @c_known: list of int. col indox of known constraints
    @seed: int, seed for random sampling
#
    @return:
    @tuple of two elements, each element is a list
    @r_query: row index in the query
    @c_query: column index in the query
    """
    if seed != -1:
        random.seed(seed)
#
    n, m = est_S.shape
    # overgenerate the random index, b/c may lose some in filtering
    index = random.sample(range(m*n), m*n)
#
    r_query = []
    c_query = []
    q_count = 0
    i = 0
#
    while q_count < queryNum and i < m*n:
#
        r = index[i] // m
        c = index[i] % m
#
        if (r, c) not in zip(r_known, c_known):
            r_query.append(r)
            c_query.append(c)
            q_count += 1
#
        i += 1
#
    if q_count < queryNum: # not enough query left 
        print("Warning: # of queries is less than requested.")
#
    return (r_query, c_query)

if __name__ == "__main__":  # just for testing
    est_S= np.array([[0,4], [-3,0.1], [-1, 11]])
    r_known = [1,2]
    c_known = [0,1]
    res1 = queryZero(est_S, 2, r_known, c_known)
    res2 = queryRand(est_S, 2, r_known, c_known, 1)


            
