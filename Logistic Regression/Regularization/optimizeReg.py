# Optimize the theta
import numpy as np

from scipy import optimize as opt


def sigmoid(z):

    return 1/(1+np.exp(-z))


def costFunction(theta, X, y, lambda_):
    m, n = X.shape
    theta = np.matrix(theta).reshape(n,1)
    hx = sigmoid(X.dot(theta))
    cost1 = np.multiply(-y,np.log(hx))
    cost2 = np.multiply((1 - y),(np.log(1 - hx)))
    reg = lambda_ * np.sum(np.square(theta))
    J = np.sum(cost1 - cost2)/m + reg/2*m

    return J


def gradient(theta, X, y, lambda_):
    m, n = X.shape
    theta = np.matrix(theta).reshape(n,1)
    hx = sigmoid(X.dot(theta))
    error = hx - y
    grads = np.zeros(n)
    for i in range(n):
        grad = np.sum(np.multiply(error, X[:, i])) / m
        if i == 0:
            #grad = np.sum(np.multiply(error,X[:,i]))/m
            grads[i] = grad
        else:
            reg = lambda_ * theta[i] / m
            #grad = np.sum(np.multiply(error, X[:, i])) / m
            grads[i] = grad + reg

    return grads


def logisticRegression(X, y, theta, lambda_):

    result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y, lambda_))

    return result

def predict(theta, X):
    theta = np.matrix(theta).reshape(X.shape[1], 1)
    hx = sigmoid(X.dot(theta))
    return hx >= 0.5