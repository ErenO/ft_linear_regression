#!/usr/bin/env python
# coding: utf-8

import os
from numpy import loadtxt, zeros, ones, array, linspace, logspace, genfromtxt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


# def computeCost(X, y, theta):
#     inner = np.power(((X * theta.T) - y), 2)
#     return np.sum(inner) / (2 * len(X))
def computeCost(X, y, theta):
    m = y.size
    pred = X.dot(theta).flatten()
    sqErrors = (pred - y) ** 2
    J = (1.0 / (2 * m)) * sqErrors.sum()
    return (J)

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def main():
    # path = os.getcwd() + '/data2.txt'
    # data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # data.head()
    # data.describe()
    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    # append a ones column to the front of the data set
    # data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)

    # X = np.matrix(X.values)
    # y = np.matrix(y.values)
    # theta = np.matrix(np.array([0,0]))
    # alpha = 0.01
    # iters = 1000

    data = genfromtxt('data2.txt', delimiter=',', skip_header=0)
    # cols = data.shape[1]
    # X = data.iloc[:,0:cols-1]
    # y = data.iloc[:,cols-1:cols]
    X = data[:, 0]
    y = data[:, 1]
    # print y
    # X = np.delete(data, 1, 1)
    # y = np.delete(data, 0, 1)
    m = y.size
    # it = ones((m, 2))
    # it[:, 1] = X
    # theta = zeros((2, 1))
    theta = np.matrix(np.array([0,0]))
    #Some gradient descent settings
    iters = 1000
    alpha = 0.01
    # print theta
    # print it
    print y
    # perform gradient descent to "fit" the model parameters
    g, cost = gradientDescent(X, y, theta, alpha, iters)

if __name__ == '__main__':
    main()
