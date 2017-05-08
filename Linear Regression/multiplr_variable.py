# coding=utf-8

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.genfromtxt(path, delimiter=",")
    x1 = data[:,0]
    x2 = data[:,1]
    y = data[:,2]
    return (x1,x2,y)

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

def cost_function(theta, x, y):
    m = len(y)
    J = (1/(2*m)) * np.sum((x.dot(theta)-y)**2)
    return J

def batch_gradient_descent(alpha, theta, x, y,num_iters):
    x_trans = np.transpose(x)
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        hyposis = x.dot(theta)
        loss = hyposis - y
        gradient = np.dot(x_trans,loss)/m
        theta = theta - alpha*gradient
        J_history[i] = cost_function(theta,x,y)
    return theta,J_history



x1,x2,y = load()
#X = np.column_stack((x1,x2))
#mean = np.mean(X)
f_x1 = (x1 - np.mean(x1))/np.std(x1)
f_x2 = (x2 - np.mean(x2))/np.std(x2)
x = np.column_stack((np.ones_like(y),f_x1,f_x2))
#y = (y - np.mean(y))/np.std(y)
#y = np.array(y).reshape((len(y),1))
iteration = 100
alphas = [0.01,0.03,0.1,0.3,1,1.3]
plotstyle = ['r','b','k','g','r--','b--']
#theta = np.zeros((3,1))

#theta,J_history= batch_gradient_descent(alpha,theta,x,y,iteration)
converge_theta = None
theta = np.zeros((3, 1))
for i in range(len(alphas)):
    result_theta, J_history =batch_gradient_descent(alphas[i], theta, x, y, iteration)
    plt.plot(range(50),J_history[0:50],plotstyle[i])
    if(alphas[i]==1):
        converge_theta = result_theta
plt.legend(['0.01','0.03','0.1', '0.3', '1', '1.3'])
plt.xlabel("number of iterations")
plt.ylabel("cost J")

print converge_theta
#print J_history
plt.show()

#make prediction
predict = np.array((1,(1650-np.mean(x1))/np.std(x1),(3-np.mean(x2))/np.std(x2)))
result = predict.dot(converge_theta)
print result

