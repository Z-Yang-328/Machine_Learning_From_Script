# One VS All classification

import numpy as np
import scipy.optimize as opt

from lrCostFunction import lrCostFunction, gradient

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape

    all_theta = np.zeros([num_labels, n + 1])
    all_theta = np.matrix(all_theta)

    X = np.insert(X, 0, 1, axis = 1)
    X = np.matrix(X)
    y = np.matrix(y)

    #initial_theta = np.zeros([n+1, 1])

    for c in range(1, num_labels+1):
        initial_theta = np.zeros([n + 1, 1])
        yc = y == c
        all_theta[c-1,:] = opt.minimize(fun=lrCostFunction, x0=initial_theta, args=(X, yc, lambda_), method='TNC', jac=gradient).x

    return all_theta
