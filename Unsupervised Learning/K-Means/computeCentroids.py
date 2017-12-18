# Compute centroids means

import numpy as np

def computeCentroids(X, idx, K):
    centroids = np.zeros([K,2])
    idx = idx.reshape(-1)
    for i in range(1,K+1):
        centroids[i-1] = np.array([np.mean(X[idx==i][:,0]),np.mean(X[idx==i][:,1])])

    return centroids