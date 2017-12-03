# Optimize the theta
import numpy as np

from scipy import optimize as opt

def sigmoid(z):

    return 1/(1+np.exp(-z))

def costFunction(theta, X, y):
    #X = np.matrix(X)
    #y = np.matrix(y)
    m, n = X.shape
    theta = np.matrix(theta).reshape(n,1)
    hx = sigmoid(X.dot(theta))
    cost1 = np.multiply(-y,np.log(hx))
    cost2 = np.multiply((1 - y),(np.log(1 - hx)))
    J = np.sum(cost1 - cost2)/m

    return J

def gradient(theta, X, y):
    m, n = X.shape
    theta = np.matrix(theta).reshape(n,1)
    hx = sigmoid(X.dot(theta))
    error = hx - y
    grads = np.zeros(n)
    for i in range(n):
        grad = np.sum(np.multiply(error,X[:,i]))/m
        grads[i] = grad

    return grads

def logisticRegression(X, y, theta):

    result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y))

    return result