__author__ = 'admin'
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
os.chdir('D:/fall 2015/ecs 289/rewrite')
#os.chdir('/home/ybluo/rewrite')

yorigin, xorigin = rp.svm_read_problem('mushrooms_10')
yorigin = array(yorigin)
xorigin = rp.buildXmatix(xorigin)


def groundTruth(yorigin):
    l = len(yorigin)
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

indx = np.where(yorigin == yClass.pop())

matrix