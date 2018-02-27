#!/usr/bin/env python
# coding: utf-8

from numpy import loadtxt, zeros, ones, array, linspace, logspace, genfromtxt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    m = y.size

    pred = X.dot(theta).flatten()

    sqErrors = (pred - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        pred = X.dot(theta).flatten()

        err1 = (pred - y) * X[:, 0]
        err2 = (pred - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * err1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * err2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

def main():
    data = genfromtxt('data2.txt', delimiter=',', skip_header=0)

    X = data[:, 0]
    y = data[:, 1]

    m = y.size

    it = ones((m, 2))
    it[:, 1] = X
    theta = zeros((2, 1))

    iterations = 1500
    alpha = 0.01
    theta, J = gradient_descent(it, y, theta, alpha, iterations)
    print (theta)
    print (J)
    y_hat = theta[1][0] * X + theta[0][0]
    plt.scatter(X, y)
    plt.plot(X, y_hat)
    plt.ylabel("Profit")
    plt.xlabel("km")
    plt.show()

if __name__ == '__main__':
    main()
