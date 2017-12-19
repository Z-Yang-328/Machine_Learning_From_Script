# Implementing PCA

import numpy as np
def pca(X):
    m, n = X.shape

    X = np.matrix(X)
    #sigma = np.cov(X)
    sigma = X.T.dot(X)/m

    U, S, V = np.linalg.svd(sigma)

    return U, S, V