# coding=utf-8
from __future__ import division
# import math
import numpy as np


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.optimize as opt

def load_data():
    pathx = 'D:\ml\logistic_regression\ex4x.txt'
    pathy = 'D:\ml\logistic_regression\ex4y.txt'
    X = np.genfromtxt(pathx)
    Y = np.genfromtxt(pathy)
    Y = Y.reshape((len(Y)), 1)
    X = np.column_stack((X, np.ones_like(Y)))
    return X, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(theta, X, Y):
    m = X.shape[0]

    J = (1 / m) * (-Y.dot(np.log(sigmoid(X.dot(theta))))
                   - (np.ones_like(Y) - Y).dot(np.log(np.ones_like(sigmoid(X.dot(theta))) - sigmoid(X.dot(theta)))))
    # grad = (1/m)*X.T.dot((sigmoid(X.dot(theta))-Y))
    return J


def compute_grad(theta, X, Y):
    m = X.shape[0]
    h = X.dot(theta)
    loss = sigmoid(h) - Y
    grad = (1 / m) * X.T.dot(loss)
    return grad


def BGD(alpha, theta, X, Y, iter_nums):
    for i in range(iter_nums):
        theta = compute_grad(theta, X, Y)


def batch_gradient_descent(alpha, theta, x, y, num_iters):
    #    x_trans = np.transpose(x)
    #    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        #        hyposis = x.dot(theta)
        #        loss = hyposis - y
        #        gradient = np.dot(x_trans,loss)/m
        gradient = compute_grad(theta, x, y)
        theta = theta - alpha * gradient
        J_history[i] = cost_function(theta, x, y)
    return theta, J_history


def test():
    x, y = load_data()
    #    x = np.column_stack((x,np.ones_like(y)))
    y = np.array(y).reshape((len(y), 1))
    iteration = 1500
    alpha = 0.01
    theta = np.zeros((3, 1))
    theta, J_history = batch_gradient_descent(alpha, theta, x, y, iteration)
    #    drwa_J(theta0_value,theta1_value,J_history)
    print (theta)
    print (J_history)


if __name__ == '__main__':
    test()


# X,y = load_data()
# theta = np.array([0,0,0])
# theta = np.reshape(theta,(len(theta),1))
# result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=compute_grad, args=(X, y))
# print (cost_function(result[0], X, y))
#
# print (result)

# theta.shape =(3,1)
# grad = np.zeros(3)
# hyposis = sigmoid(X.dot(theta.T))
# delta = hyposis - y
# l = grad.size
# for i in range(l):

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
