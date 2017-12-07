# cost function for one vs. all

import numpy as np

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def lrCostFunction(theta, X, y, lambda_):
    m, n = X.shape
    theta = np.matrix(theta).reshape(n, 1)
    X = np.matrix(X)
    y = np.matrix(y)
    hx = sigmoid(X.dot(theta))
    cost1 = np.sum(np.multiply(~y, np.log(hx)))
    cost2 = np.sum(np.multiply(1-y, np.log(1-hx)))
    #reg = lambda_ * np.sum(np.power(theta, 2))
    reg = (lambda_ * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)))
    J = (cost1 + cost2)/m + reg/(2*m)

    return J

def gradient(theta, X, y, lambda_):
    m, n = X.shape
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    grads = np.zeros(n)
    hx = sigmoid(X * theta.T)
    error = hx - y

    for i in range(n):
        grad = np.sum(np.multiply(error, X[:,i]) / m)
        if i == 0:
            grads[i] = grad
        else:
            reg = lambda_ * theta[:, i] / m
            grads[i] = grad + reg


    return grads