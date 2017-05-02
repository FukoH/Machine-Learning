# coding=utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x = data[:,0]
    y = data[:,1]
    return (x,y)

def cost_function(theta, x, y):
    m = len(y)
    J = (1/(2*m)) * np.sum((x.dot(theta)-y)**2)
    return J

def batch_gradient_descent(alpha, theta, x, y,num_iters):
    x_trans = np.transpose(x)
    m = len(y)
    J_history = np.zeros(num_iters)
    theta0_value = np.zeros(num_iters)
    theta1_value = np.zeros(num_iters)
    for i in range(num_iters):
        hyposis = x.dot(theta)
        loss = hyposis - y
        gradient = np.dot(x_trans,loss)/m
        theta = theta - alpha*gradient
        theta0_value[i] = theta[0]
        theta1_value[i] = theta[1]
        J_history[i] = cost_function(theta,x,y)
    return theta,J_history,theta0_value,theta1_value

def drwa_J(theta0_value,theta1_value,J_history):
    X,Y = np.meshgrid(theta0_value,theta1_value)
    Z = J_history
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    plt.xlim(0,2)
    plt.ylim(-4,0)
    #ax.contourf(X, Y, Z, cmap='rainbow')
    plt.show()


def test():
    x, y = load_data("D:\machine learning\linear regression\ex1data1.txt")
    x = np.column_stack((x,np.ones_like(y)))
    y = np.array(y).reshape((len(y),1))
    iteration = 1500
    alpha = 0.01
    theta = np.zeros((2,1))
    theta,J_history,theta0_value,theta1_value = batch_gradient_descent(alpha,theta,x,y,iteration)
    drwa_J(theta0_value,theta1_value,J_history)
    print theta
    print J_history

if __name__=='__main__':
    test()