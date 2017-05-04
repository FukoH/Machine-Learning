# coding=utf-8
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def compute_grad(theta, X, y):

    #print theta.shape

    theta.shape = (1, 3)

    grad = np.zeros(3)

    h = sigmoid(X.dot(theta.T))

    delta = h - y

    l = grad.size

    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1

    theta.shape = (3,)

    return  grad

def sigmoid(x):
    return 1/(1+math.exp(-x))