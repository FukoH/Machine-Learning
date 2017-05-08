# coding=utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def load():
    pathx = r'C:\Users\Administrator\PycharmProjects\Machine-Learning\dataset\linear regression II\ex3x.txt'
    pathy = r'C:\Users\Administrator\PycharmProjects\Machine-Learning\dataset\linear regression II\ex3y.txt'
    X = np.genfromtxt(pathx)
    Y = np.genfromtxt(pathy)
    x1 = X[:,0]
    x2 = X[:,1]
    Y = Y.reshape((len(Y)), 1)
    #X = np.column_stack((X, np.ones_like(Y)))
    return x1,x2,Y

def normal_equation(X,y):
    #np.linalg.inv((X.T.dot(X)))
    theta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
    return theta

x1,x2,y = load()
X = np.column_stack((np.ones_like(y),x1,x2))
theta = normal_equation(X,y)
print theta