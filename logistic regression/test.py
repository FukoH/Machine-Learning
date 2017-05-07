# coding=utf-8
#import math
#rom __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.genfromtxt(r'C:\Users\fuko\PycharmProjects\exercise\ex4x.txt')
Y = np.genfromtxt(r'C:\Users\fuko\PycharmProjects\exercise\ex4y.txt')
Y = Y.reshape((len(Y), 1))

pos = np.where(Y==1)
neg = np.where(Y==0)

plt.scatter(X[pos,0],X[pos,1],marker='o',c='r')
plt.scatter(X[neg,0],X[neg,1],marker='x',c='g')

plt.xlabel('Exam1 Score')
plt.ylabel('Exam2 Score')
plt.show()
