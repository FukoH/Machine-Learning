# coding=utf-8
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

def load_data():
    pathx = 'D:\ml\logistic_regression\ex4x.txt'
    pathy = 'D:\ml\logistic_regression\ex4y.txt'
    X = np.genfromtxt(pathx,delimiter=',')
    Y = np.genfromtxt(pathy)
    Y = Y.reshape((len(Y)),1)
    return X,Y

def sigmoid(x):
    return 1/(1+math.exp(-x))

def cost_function(theta,X,Y):
    m = X.shape[0]
    
    J = (1/m)*(-Y.dot(np.log(sigmoid(X.dot(theta))))
               -(np.ones_like(Y)-Y).dot(np.log(np.ones_like(sigmoid(X.dot(theta)))-sigmoid(X.dot(theta)))))
    #grad = (1/m)*X.T.dot((sigmoid(X.dot(theta))-Y))
    return J

def compute_grad(theta,X,Y):
    m = X.shape[0]
    grad = (1 / m) * X.T.dot((sigmoid(X.dot(theta)) - Y))
    return grad

X,y = load_data()
theta = np.array([0,0,0])
theta = np.reshape(theta,(len(theta),1))
result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=compute_grad, args=(X, y))
print (cost_function(result[0], X, y))

print (result)

    # theta.shape =(3,1)
    # grad = np.zeros(3)
    # hyposis = sigmoid(X.dot(theta.T))
    # delta = hyposis - y
    # l = grad.size
    #for i in range(l):

# def compute_grad(theta, X, y):
#
#     #print theta.shape
#
#     theta.shape = (1, 3)
#
#     grad = np.zeros(3)
#
#     h = sigmoid(X.dot(theta.T))
#
#     delta = h - y
#
#     l = grad.size
#
#     for i in range(l):
#         sumdelta = delta.T.dot(X[:, i])
#         grad[i] = (1.0 / m) * sumdelta * - 1
#
#     theta.shape = (3,)
#
#     return  grad
