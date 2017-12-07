# Make prediction

import numpy as np
from lrCostFunction import sigmoid

def predictOneVsAll(all_theta, X):
    m, n = X.shape
    num_labels = all_theta.shape[0]

    X = np.matrix(X)
    X = np.insert(X, 0, 1, axis = 1)
    predictions = np.zeros([m, num_labels])
    predictions = np.matrix(predictions)

    for i in range(num_labels):
        hx = sigmoid(X.dot(all_theta[i].T))
        predictions[:,i] = hx

    preds = np.argmax(predictions, axis = 1) + 1

    return preds

