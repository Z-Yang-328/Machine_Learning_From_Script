# Make predictions

import numpy as np

from optimize import sigmoid

def predict(theta, X):
    theta = np.matrix(theta).reshape(X.shape[1], 1)
    hx = sigmoid(X.dot(theta))

    return hx >= 0.5