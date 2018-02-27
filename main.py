#!/usr/bin/env python
# coding: utf-8

from numpy import loadtxt, zeros, ones, array, linspace, logspace, genfromtxt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
# import pandas as pd


def compute_cost(X, y, theta):
    m = y.size
    pred = X.dot(theta).flatten()
    sqErrors = (pred - y) ** 2
    J = (1.0 / (2 * m)) * sqErrors.sum()
    return (J)

def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = zeros((num_iters, 1))
    print (X.dot(theta).flatten())
    for i in range(num_iters):
        pred = X.dot(theta).flatten()
        err1 = (pred - y) * X[:, 0]
        err2 = (pred - y) * X[:, 1]

        theta[0][0] = theta[0][0] * alpha * (1.0/m) * err1.sum()
        theta[1][0] = theta[1][0] * alpha * (1.0/m) * err2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)
    return theta, J_history

def main():
    #Load the dataset
    # data = loadtxt('data.csv', delimiter=',')
    # data = pd.read_csv('data.csv', name = ['km', 'price'])
    data = genfromtxt('data2.txt', delimiter=',', skip_header=1)
    # print data
    #Plot the data
    # scatter(data[:, 0], data[:, 1], marker='km', c='price')
    # title('Profits distribution')
    # xlabel('Population of City in 10,000s')
    # ylabel('Profit in $10,000s')
    # show()

    X = data[:, 0]
    y = data[:, 1]

    # print y
    #number of training samples
    m = y.size
    # print m
    #Add a column of ones to X (interception data)
    it = ones((m, 2))
    # print (it)
    it[:, 1] = X
    # print (it)
    #Initialize theta parameters
    theta = zeros((2, 1))

    #Some gradient descent settings
    iterations = 1500
    alpha = 0.01
    theta, J = gradient_descent(it, y, theta, alpha, iterations)
    # print (theta)
    # print (J)
if __name__ == '__main__':
    main()
