# Feature normalization

import numpy as np

def featureNormalize(X):

    mu = np.mean(X)
    X_norm = X - mu

    sigma = np.std(X)
    X_norm = X_norm / sigma
    #X_norm = np.matrix(X_norm)

    return (X_norm, mu, sigma)