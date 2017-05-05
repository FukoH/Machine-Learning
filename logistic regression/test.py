# coding=utf-8
from __future__ import division
# import math
import numpy as np


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.optimize as opt


pathx = 'D:\ml\logistic_regression\ex4x.txt'
pathy = 'D:\ml\logistic_regression\ex4y.txt'
X = np.genfromtxt(pathx)
Y = np.genfromtxt(pathy)
Y = Y.reshape((len(Y)), 1)
X = np.column_stack((X, np.ones_like(Y)))
print (5 - Y)


